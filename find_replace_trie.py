"""
given a large bunch (~1e7) of things to find and replace and a folder of data to clean
and writes the cleaned copied data to the output folder, preserving the relative path

if a tokenizer is used, only matches at token boundaries
"""
import bisect
import collections
import io
import math
import os
import random
import re
import sys
import time
from typing import Any
from typing import AnyStr
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


class Match:
    __slots__ = ('__regs', '__str')

    def __init__(self,
                 start: int,
                 end: int,
                 match: str,
                 ):
        """
        match result (similar to re.Match, but currently using token indices because of how tokenization works)
        todo: normalize to char index
        todo: use a frozen dataclass
        todo: just refer to original string once we have the char indices, for compat

        :param start: index of start TOKEN (not char)
        :param end: index after end token
        :param match: string matched
        """
        self.__regs = ((start, end),)  # mimic the re.Match object
        self.__str = match  # re.Match stores a reference to the ENTIRE ORIGINAL STRING, let's not do that

    @property
    def regs(self) -> Tuple[Tuple[int, int]]:
        return self.__regs

    @property
    def str(self) -> str:
        return self.__str

    def __getitem__(self, group_index: int) -> str:
        if group_index != 0:
            raise IndexError('no such group')
        return self.__str

    def group(self, group_index: int = 0) -> str:
        return self[group_index]

    def start(self, group_index: int = 0) -> int:
        if group_index != 0:
            raise IndexError('no such group')
        return self.__regs[0][0]

    def end(self, group_index: int = 0) -> int:
        if group_index != 0:
            raise IndexError('no such group')
        return self.__regs[0][1]

    def span(self, group_index: int = 0) -> Tuple[int, int]:
        if group_index != 0:
            raise IndexError('no such group')
        return self.__regs[0]

    def __len__(self):
        return self.__regs[0][1] - self.__regs[0][0]

    def __str__(self) -> str:
        return self.__str

    def __repr__(self) -> str:
        return f'<Match object; span={self.__regs[0]}, match={repr(self.__str)}>'


def format_bytes(num):
    """
    string formatting

    :type num: int
    :rtype: str
    """
    num = abs(num)
    if num == 0:
        return '0 Bytes'
    elif num == 1:
        return '1 Byte'
    unit = 0
    while num >= 1024 and unit < 8:
        num /= 1024.0
        unit += 1
    unit = ['Bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'][unit]
    return ('%.2f %s' if num % 1 else '%d %s') % (num, unit)


def format_seconds(num):
    """
    string formatting

    :type num: int | float
    :rtype: str
    """
    num = abs(num)
    if num == 0:
        return '0 seconds'
    elif num == 1:
        return '1 second'
    if num < 1:
        # display 2 significant figures worth of decimals
        return ('%%0.%df seconds' % (1 - int(math.floor(math.log10(abs(num)))))) % num
    unit = 0
    denominators = [60.0, 60.0, 24.0, 7.0]
    while num >= denominators[unit] and unit < 4:
        num /= denominators[unit]
        unit += 1
    unit = ['seconds', 'minutes', 'hours', 'days', 'weeks'][unit]
    return ('%.2f %s' if num % 1 else '%d %s') % (num, unit[:-1] if num == 1 else unit)


_NOTHING = object()  # in Google's pygtrie library this is called `_SENTINEL`
REPLACEMENTS_TYPE = Union[
    Iterable[Tuple[AnyStr, AnyStr]],
    Dict[AnyStr, AnyStr],
    Generator[Tuple[AnyStr, AnyStr], Any, None],
]


class Trie(object):
    __slots__ = ('root', 'tokenizer', 'detokenizer', 'length')

    @staticmethod
    def fromkeys(keys: Iterable[str],
                 default: str = '',
                 case_sensitive: bool = True,
                 verbose: bool = False,
                 ) -> 'Trie':
        _trie = Trie(lowercase=not case_sensitive)
        _trie.update(((key, default) for key in keys), verbose=verbose)
        return _trie

    class Node(dict):
        # build time for 650656 words: 3.8s
        # size for 650656 words: 369133837
        # query time for 650656 words (dist=9): 8.1s
        __slots__ = ('REPLACEMENT',)

        # noinspection PyMissingConstructor
        def __init__(self):
            self.REPLACEMENT = _NOTHING

        # # 3% smaller trie, but as much as 10% slower
        # # build time for 650656 words: 3.2s
        # # size for 650656 words: 357702525
        # # query time for 650656 words (dist=9): 8.9s
        # __slots__ = ()
        #
        # @property
        # def REPLACEMENT(self):
        #     return self.get(_NOTHING, _NOTHING)
        #
        # @REPLACEMENT.setter
        # def REPLACEMENT(self, value):
        #     if value is _NOTHING:
        #         if _NOTHING in self:
        #             del self[_NOTHING]
        #     else:
        #         self[_NOTHING] = value

    def __init__(self,
                 replacements: Optional[REPLACEMENTS_TYPE] = None,
                 tokenizer: Callable[[Union[AnyStr, Iterable[AnyStr]]], Iterable[AnyStr]] = None,
                 detokenizer: Callable[[Iterable[AnyStr]], AnyStr] = None,
                 lowercase: bool = False,
                 ):
        """

        :param replacements:
        :param tokenizer: tokenizer that reads one character at a time and yields tokens
        :param detokenizer: function to combine tokens back into a string
        :param lowercase: if True, lowercase all the things (including output)
        """
        self.root = self.Node()
        self.length = 0

        if tokenizer is None:
            if not lowercase:
                def _list_tokenizer(seq):
                    for elem in seq:
                        yield elem

                self.tokenizer = _list_tokenizer
            else:
                def _lowercase_list_tokenizer(seq):
                    for elem in seq:
                        yield elem.lower()

                self.tokenizer = _lowercase_list_tokenizer
        elif lowercase:
            def _lowercase_wrap_tokenizer(seq):
                for elem in tokenizer(seq):
                    yield elem.lower()

            self.tokenizer = _lowercase_wrap_tokenizer
        else:
            self.tokenizer = tokenizer

        if detokenizer is None:
            def _list_detokenizer(seq):
                return ''.join(seq)

            self.detokenizer = _list_detokenizer
        else:
            self.detokenizer = detokenizer

        if replacements is not None:
            self.update(replacements)

    @property
    def nbytes(self) -> int:
        """
        size of this Trie in bytes
        similar to numpy.array([]).nbytes
        """
        total_bytes = 0
        seen_ids = set()

        for obj in (self, self.length, self.tokenizer, self.detokenizer):
            assert id(obj) not in seen_ids
            total_bytes += sys.getsizeof(obj)
            seen_ids.add(id(obj))

        stack = [self.root]
        while stack:
            node = stack.pop()
            stack.extend(node.values())
            for obj in (node, node.REPLACEMENT, *node.keys()):
                if id(obj) not in seen_ids:
                    total_bytes += sys.getsizeof(obj)
                    seen_ids.add(id(obj))

        return total_bytes

    def _sort_keys(self, ascending=True):
        raise NotImplementedError  # todo

    def __contains__(self, key: AnyStr) -> bool:
        head = self.root
        for token in self.tokenizer(key):
            if token not in head:
                return False
            head = head[token]
        return head.REPLACEMENT is not _NOTHING

    def __len__(self) -> int:
        assert self.length == sum(1 for _ in self.items())  # debugging test, not stress-tested yet
        return self.length

    def _item_slice(self, start, stop, step=None):
        out = []
        for key, value in self.items():
            if key >= stop:
                return out[::step]
            elif key >= start:
                out.append((key, value))
        return out[::step]

    def __getitem__(self, key):
        if type(key) is slice:
            return [value for key, value in self._item_slice(key.start, key.stop, key.step)]
        head = self.root
        for token in self.tokenizer(key):
            if token not in head:
                raise KeyError(key)
            head = head[token]
        if head.REPLACEMENT is _NOTHING:
            raise KeyError(key)
        return head.REPLACEMENT

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key, value):
        head = self.root
        for token in self.tokenizer(key):
            head = head.setdefault(token, self.Node())
        if head.REPLACEMENT is not _NOTHING:
            return head.REPLACEMENT
        self.length += 1
        head.REPLACEMENT = value
        return value

    def __setitem__(self, key, value):
        head = self.root
        for token in self.tokenizer(key):
            head = head.setdefault(token, self.Node())
        if head.REPLACEMENT is _NOTHING:
            self.length += 1
        head.REPLACEMENT = value
        return value

    def pop(self, key=None):
        # empty Trie, so behave like an empty set
        if not self.root.keys():
            raise KeyError(key)

        # pop first item if key not specified
        if key is None:
            key = next(self.keys())

        head = self.root
        breadcrumbs = [(None, head)]

        # trace through trie
        for token in self.tokenizer(key):
            if token not in head:
                raise KeyError(key)
            head = head[token]
            breadcrumbs.append((token, head))

        # key has no value, hence not contained in trie
        if head.REPLACEMENT is _NOTHING:
            raise KeyError(key)

        # store value to be returned later and erase value
        ret_val = head.REPLACEMENT
        head.REPLACEMENT = _NOTHING
        self.length -= 1

        # erase unnecessary nodes backwards, if they have no value and no descendants
        prev_token, _ = breadcrumbs.pop(-1)
        for token, head in breadcrumbs[::-1]:
            if len(head[prev_token]) == 0:
                del head[prev_token]
                prev_token = token
            else:
                break
            if head.REPLACEMENT is not _NOTHING:
                break

        # finally return the popped key & value
        return key, ret_val

    def __delitem__(self, key):
        if isinstance(key, slice):
            for key, value in self._item_slice(key.start, key.stop, key.step):
                self.pop(key)
        elif key is None:
            raise KeyError(None)  # handle None because pop(None) will pop the first item
        else:
            self.pop(key)

    def items(self):
        # todo: special case for empty str?
        _path = []
        _stack = [(self.root, sorted(self.root.keys(), reverse=True))]
        while _stack:
            head, keys = _stack.pop(-1)
            if keys:
                key = keys.pop(-1)
                _stack.append((head, keys))
                head = head[key]
                _path.append(key)
                if head.REPLACEMENT is not _NOTHING:
                    yield self.detokenizer(_path), head.REPLACEMENT
                _stack.append((head, sorted(head.keys(), reverse=True)))
            elif _path:
                _path.pop(-1)
            else:
                assert not _stack

    def to_regex(self,
                 fuzzy_quotes: bool = True,
                 fuzzy_spaces: bool = True,
                 fffd_any: bool = True,
                 simplify: bool = True,
                 boundary: bool = False,
                 ) -> str:
        """
        build a (potentially very very long) regex to find any text in the trie

        :param fuzzy_quotes: unicode quotes also match ascii quotes
        :param fuzzy_spaces: whitespace char matches any unicode whitespace char
        :param fffd_any: lets the \ufffd char match anything
        :param simplify: shorten the output regex via post-processing rules
        :param boundary: enforce boundary at edge of output regex using \b
        :return: regex string
        """
        assert list(self.tokenizer('test-test test')) == list('test-test test'), "shouldn't use a tokenizer"

        _parts = [[], []]
        _stack = [(self.root, sorted(self.root.keys(), reverse=True))]
        while _stack:
            head, keys = _stack.pop(-1)
            if keys:
                key = keys.pop(-1)
                _stack.append((head, keys))
                head = head[key]

                # add new item
                if _parts[-1]:
                    _parts[-1].append('|')

                # allow any whitespace
                # note that this must happen before regex char escaping as different spaces escape differently
                if fuzzy_spaces:
                    key = re.sub(r'\s', ' ', key)  # we'll do the \s replacement later

                # character escaping
                key = re.escape(key)

                # allow ascii quotes
                if fuzzy_quotes:
                    key = key.replace('\\\u2035', "[\\\u2035']")  # reversed prime
                    key = key.replace('\\\u2032', "[\\\u2032']")  # prime
                    key = key.replace('\\\u2018', "[\\\u2018']")  # left quote
                    key = key.replace('\\\u2019', "[\\\u2019']")  # right quote
                    key = key.replace('\\\u0060', "[\\\u0060']")  # grave
                    key = key.replace('\\\u00b4', "[\\\u00b4']")  # acute accent
                    key = key.replace('\\\u201d', '[\\\u201d"]')  # left double quote
                    key = key.replace('\\\u201c', '[\\\u201c"]')  # right double quote
                    key = key.replace('\\\u301d', '[\\\u301d"]')  # reversed double prime quotation mark
                    key = key.replace('\\\u301e', '[\\\u301e"]')  # double prime quotation mark

                # fffd matches any single character
                if fffd_any:
                    key = key.replace('\ufffd', '.')  # unicode replacement character

                _parts[-1].append(key)

                # one level down
                _stack.append((head, sorted(head.keys(), reverse=True)))
                _parts.append([])

            else:
                _current_parts = _parts.pop()
                if _current_parts:
                    if head.REPLACEMENT is not _NOTHING:
                        _parts[-1].append('(?:')
                        _parts[-1].extend(_current_parts)
                        _parts[-1].append(')?')
                    elif len(head) != 1:
                        _parts[-1].append('(?:')
                        _parts[-1].extend(_current_parts)
                        _parts[-1].append(')')
                    else:
                        _parts[-1].extend(_current_parts)

        assert len(_parts) == 1
        _pattern = ''.join(_parts[0])

        if simplify:

            def char_group(match: re.Match) -> str:
                """
                helper function to simplify character groups for regex creation
                used with the regex below to convert '(?:a|b|c|d)' -> '[abcd]'
                """
                out = ['[']
                sep = False
                escaped = False
                unicode = 0
                for char in match.groups()[0]:
                    if unicode:
                        assert not sep
                        out.append(char)
                        unicode -= 1
                        if not unicode:
                            sep = True

                    elif escaped:
                        assert not sep
                        out.append(char)
                        escaped = False
                        if char == '':
                            unicode = 4
                        elif char in '1234567890':
                            unicode = 2
                        else:
                            sep = True
                    elif char == '\\':
                        assert not sep
                        out.append('\\')
                        escaped = True
                    elif char == '|':
                        assert sep
                        sep = False
                    else:
                        assert not sep
                        out.append(char)
                        sep = True
                assert sep
                out.append(']')
                return ''.join(out)

            # this matches a single (possibly escaped) character
            # noinspection RegExpUnnecessaryNonCapturingGroup
            _char = re.compile(r'(?:\\(?:U[0-9a-fA-F]{8}|u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|[0-7]{2,3}|.)|[^\\])')

            # simplify `(?:x|y|z)` -> `[xyz]`
            _pattern = re.sub(r'(?<!\\)\(\?:({C}(?:\|{C})*)\)'.format(C=_char.pattern), char_group, _pattern)

            # simplify `(?:[xyz])` -> `[xyz]`
            # noinspection RegExpRedundantEscape
            _pattern = re.sub(r'(?<!\\)\(\?:(\[[^\[\]]*[^\[\]\\]\])\)', r'\1', _pattern)

            # simplify `(?:[xyz]?)?` -> `[xyz]?`
            # noinspection RegExpRedundantEscape
            _pattern = re.sub(r'(?<!\\)\(\?:(\[[^\[\]]*[^\[\]\\]\]\?)\)\??', r'\1', _pattern)

            # simplify `[.]` -> `.`
            # noinspection RegExpRedundantEscape
            _pattern = re.sub(r'(?<!\\)\[({C})\]'.format(C=_char.pattern), r'\1', _pattern)

            # simplify `(?:.)` -> `.`
            _pattern = re.sub(r'\(\?:({C})\)'.format(C=_char.pattern), r'\1', _pattern)

        # force surrounding brackets, and enforce word boundary
        if _pattern[3:] == '(?:':
            assert _pattern[-1] == ')'
            if boundary:
                _pattern = '(?:\\b%s\\b)' % _pattern[3:-1]
        elif boundary:
            _pattern = '(?:\\b%s\\b)' % _pattern
        else:
            _pattern = '(?:%s)' % _pattern

        if fuzzy_spaces:
            _pattern = _pattern.replace('\\ ', '\\s')
            assert ' ' not in _pattern

            while '\\s\\s' in _pattern:
                if '\\s\\s+' in _pattern:
                    _pattern = _pattern.replace('\\s\\s+', '\\s+')
                else:
                    _pattern = _pattern.replace('\\s\\s', '\\s+')
        else:
            _pattern = _pattern.replace('\\ ', ' ')

        # done
        return _pattern

    def keys(self):
        for key, value in self.items():
            yield key

    def values(self):
        for key, value in self.items():
            yield value

    def update(self,
               replacements: REPLACEMENTS_TYPE,
               verbose: bool = True,
               ):
        if isinstance(replacements, (list, tuple)):
            print_str = f'(%d pairs loaded out of {len(replacements)})'
        elif isinstance(replacements, dict):
            print_str = f'(%d pairs loaded out of {len(replacements)})'
            replacements = replacements.items()
        else:
            print_str = '(%d pairs loaded)'

        for index, (sequence, replacement) in enumerate(replacements):
            if verbose and (index + 1) % 50000 == 0:
                print(print_str % (index + 1))
            self[sequence] = replacement
        return self

    def levenshtein_lookup(self,
                           word: str,
                           distance: int,
                           insertion_cost: Union[int, float] = 1,
                           deletion_cost: Union[int, float] = 1,
                           substitution_cost: Union[int, float] = 1,
                           transposition_cost: Union[int, float] = 2,
                           ) -> Generator[str, Any, None]:
        """
        levenshtein / edit distance based approximate string lookup
        todo: maybe allow returning values not just keys?
        todo: special case for empty str?
        todo: special case for distance = 0
        todo: return distance?
        todo: return node.REPLACEMENT?
        """
        assert list(self.tokenizer('test-test test')) == list('test-test test'), "shouldn't use a tokenizer"
        assert distance >= 0
        assert isinstance(word, str)
        assert len(word) > 0
        assert insertion_cost >= 0
        assert deletion_cost >= 0
        assert substitution_cost >= 0
        # assert transposition_cost >= 0

        _path = []
        _dp_table = [[d * deletion_cost for d in range(len(word) + 1)]]
        _stack = [(self.root, list(self.root.keys()))]
        _word = tuple(enumerate(word))
        _template = [0] * (len(word) + 1)

        # noinspection PyShadowingNames
        def _levenshtein_iter(key):
            nonlocal _path
            nonlocal _word
            nonlocal word
            nonlocal _dp_table

            next_row = _template[:]
            next_row[0] = len(_dp_table) * insertion_cost

            for idx_2, char_2 in _word:
                # [idx_2 + 1] instead of j since _dp_table[-1] and current_row are one character longer than word
                if key == char_2:
                    next_row[idx_2 + 1] = _dp_table[-1][idx_2]
                else:
                    insertions = _dp_table[-1][idx_2 + 1] + insertion_cost
                    deletions = next_row[idx_2] + deletion_cost
                    substitutions = _dp_table[-1][idx_2] + substitution_cost
                    # next_row[idx_2 + 1] = min(insertions, deletions, substitutions)

                    # character transposition
                    if _path and idx_2 and key == word[idx_2 - 1] and _path[-1] == char_2:
                        transpositions = _dp_table[-2][idx_2 - 1] + transposition_cost
                        next_row[idx_2 + 1] = min(insertions, deletions, substitutions, transpositions)
                    else:
                        next_row[idx_2 + 1] = min(insertions, deletions, substitutions)
            return next_row

        while _stack:
            head, keys = _stack[-1]
            if keys:
                key = keys.pop(-1)

                next_head = head[key]
                next_row = _levenshtein_iter(key)

                # early exit?
                if min(next_row) <= distance:
                    _path.append(key)
                    _dp_table.append(next_row)
                    _stack.append((next_head, list(next_head.keys())))
                    if next_row[-1] <= distance and next_head.REPLACEMENT is not _NOTHING:
                        yield self.detokenizer(_path)  # , next_head.REPLACEMENT

            elif _path:
                _path.pop(-1)
                _dp_table.pop(-1)
                _stack.pop(-1)

            else:
                assert len(_dp_table) == 1
                assert len(_stack) == 1
                _stack.clear()

    def damerau_levenshtein_lookup(self,
                                   word: str,
                                   distance: int,
                                   insertion_cost: Union[int, float] = 1,
                                   deletion_cost: Union[int, float] = 1,
                                   substitution_cost: Union[int, float] = 1,
                                   transposition_cost: Union[int, float] = 2,
                                   ) -> Generator[str, Any, None]:
        """
        damerau levenshtein (ie. with transpose) based approximate string lookup
        todo: maybe allow returning values not just keys?
        todo: special case for empty str?
        todo: special case for distance = 0
        """
        assert list(self.tokenizer('test-test test')) == list('test-test test'), "shouldn't use a tokenizer"
        assert distance >= 0
        assert isinstance(word, str)
        assert len(word) > 0
        assert insertion_cost >= 0
        assert deletion_cost >= 0
        assert substitution_cost >= 0
        assert transposition_cost >= 0

        _path = []
        _dp_table = [[d * deletion_cost for d in range(1, len(word) + 1)] + [0]]
        _stack = [(self.root, list(self.root.keys()))]
        _word = tuple(enumerate(word))
        _template = [0] * (len(word) + 1)

        # _out = []

        # noinspection PyShadowingNames
        def _damerau_levenshtein_iter(key):
            nonlocal _path
            nonlocal _word
            nonlocal word
            nonlocal _dp_table

            next_row = _template[:]  # faster than _template.copy()
            next_row[-1] = _dp_table[-1][-1] + insertion_cost  # hack to make -1 an index
            for idx_2, char_2 in _word:
                if key == char_2:
                    next_row[idx_2] = _dp_table[-1][idx_2 - 1]
                else:
                    insertions = _dp_table[-1][idx_2] + insertion_cost
                    deletions = next_row[idx_2 - 1] + deletion_cost
                    substitutions = _dp_table[-1][idx_2 - 1] + substitution_cost
                    # next_row[idx_2] = min(insertions, deletions, substitutions)

                    # character transposition
                    if _path and idx_2 and key == word[idx_2 - 1] and _path[-1] == char_2:
                        transpositions = _dp_table[-2][idx_2 - 2] + transposition_cost
                        next_row[idx_2] = min(insertions, deletions, substitutions, transpositions)
                    else:
                        next_row[idx_2] = min(insertions, deletions, substitutions)
            return next_row

        while _stack:
            head, keys = _stack[-1]
            if keys:
                key = keys.pop(-1)

                next_head = head[key]
                next_row = _damerau_levenshtein_iter(key)

                # early exit?
                if min(next_row) <= distance:
                    _path.append(key)
                    _dp_table.append(next_row)
                    _stack.append((next_head, list(next_head.keys())))
                    if next_row[-2] <= distance and next_head.REPLACEMENT is not _NOTHING:  # order doesn't change speed
                        # _out.append(self.detokenizer(_path))
                        yield self.detokenizer(_path)  # , next_head.REPLACEMENT

            elif _path:
                _path.pop(-1)
                _dp_table.pop(-1)
                _stack.pop(-1)

            else:
                assert len(_dp_table) == 1
                assert len(_stack) == 1
                _stack.clear()

        # return _out

    def _yield_tokens(self,
                      file_path: Union[str, os.PathLike],
                      encoding: str = 'utf8',
                      ) -> Generator[str, None, None]:
        """
        yield tokens from a file given its path

        :param encoding:
        :param file_path: file to read
        """
        with io.open(file_path, mode=('rt', 'rb')[encoding is None], encoding=encoding) as _f:
            # noinspection PyShadowingNames
            for token in self.tokenizer(char for line in _f for char in line):  # make sure to read line by line
                yield token

    def _translate_tokens(self,
                          tokens: Iterable[AnyStr],
                          ) -> Generator[AnyStr, None, None]:
        """
        processes text and yields output one token at a time
        :param tokens: iterable of hashable objects, preferably strings
        """
        output_buffer = collections.deque()  # [(index, token), ...]
        matches = dict()  # {span_start: (span_end + 1, REPLACEMENT), ...} <-- because: match == seq[start:end+1]
        spans = dict()  # positions that are partial matches: {span_start: span_head, ...}
        matches_to_remove = set()  # positions where matches may not start
        spans_to_remove = set()  # positions where matches may not start, or where matching failed

        for index, input_item in enumerate(tokens):
            # append new item to output_buffer
            output_buffer.append((index, input_item))

            # append new span to queue
            spans[index] = self.root

            # reset lists of things to remove
            matches_to_remove.clear()  # clearing is faster than creating a new set
            spans_to_remove.clear()

            # process spans in queue
            for span_start, span_head in spans.items():
                if input_item in span_head:
                    new_head = span_head[input_item]
                    spans[span_start] = new_head
                    if new_head.REPLACEMENT is not _NOTHING:
                        matches[span_start] = (index + 1, new_head.REPLACEMENT)

                        # longest subsequence matching does not allow one match to start within another match
                        matches_to_remove.update(range(span_start + 1, index + 1))
                        spans_to_remove.update(range(span_start + 1, index + 1))

                else:
                    # failed to match the current token
                    spans_to_remove.add(span_start)

            # remove impossible spans and matches from queues
            for span_start in matches_to_remove:
                if span_start in matches:
                    del matches[span_start]
            for span_start in spans_to_remove:
                if span_start in spans:
                    del spans[span_start]

            # get indices of matches and spans
            first_match = min(matches) if matches else index
            first_span = min(spans) if spans else index

            # emit all matches that start before the first span
            while first_match < first_span:
                # take info
                match_end, match_replacement = matches[first_match]
                # emit until match start
                while output_buffer and output_buffer[0][0] < first_match:
                    yield output_buffer.popleft()[1]
                # clear output_buffer until match end
                while output_buffer and output_buffer[0][0] < match_end:  # remember match_end already has the +1
                    output_buffer.popleft()
                # emit replacement
                for token in self.tokenizer(match_replacement):
                    yield token
                # grab next match and retry
                del matches[first_match]
                first_match = min(matches) if matches else index

            # emit until span
            while output_buffer and output_buffer[0][0] < first_span:
                yield output_buffer.popleft()[1]

        # ignore remaining unmatched spans, yield matches only
        for match_start, (match_end, match_replacement) in sorted(matches.items()):
            # emit until match start
            while output_buffer and output_buffer[0][0] < match_start:  # remember match_end already has the +1
                yield output_buffer.popleft()[1]
            # clear output_buffer until match end
            while output_buffer and output_buffer[0][0] < match_end:  # remember match_end already has the +1
                output_buffer.popleft()
            # emit replacement one token at a time
            for token in self.tokenizer(match_replacement):
                yield token

        # emit remainder of output_buffer
        while output_buffer:
            yield output_buffer.popleft()[1]

    def finditer(self,
                 input_sequence: AnyStr,
                 *,
                 allow_overlapping: bool = False,
                 ) -> Generator[Match, Any, None]:
        """
        finds all occurrences within a string

        :param input_sequence: iterable of hashable objects
        :param allow_overlapping: yield all overlapping matches (soar -> so, soar, oar)
        """
        matches = dict()  # {span_start: (span_end + 1, [span_stuff, ...]), ...} <-- because: match == seq[start:end+1]
        spans = dict()  # positions that are partial matches: {span_start: (span_head, [span_stuff, ...]), ...}
        matches_to_remove = set()  # positions where matches may not start
        spans_to_remove = set()  # positions where matches may not start, or where matching failed

        for index, input_item in enumerate(self.tokenizer(input_sequence)):
            # append new span to queue
            spans[index] = (self.root, [])

            # reset lists of things to remove
            matches_to_remove.clear()  # clearing is faster than creating a new set
            spans_to_remove.clear()

            # process spans in queue
            for span_start, (span_head, span_seq) in spans.items():
                if input_item in span_head:
                    new_head = span_head[input_item]
                    span_seq.append(input_item)
                    spans[span_start] = (new_head, span_seq)
                    if new_head.REPLACEMENT is not _NOTHING:
                        matches[span_start] = (index + 1, span_seq[:])

                        # longest subsequence matching does not allow one match to start within another match
                        if not allow_overlapping:
                            matches_to_remove.update(range(span_start + 1, index + 1))
                            # spans_to_remove.update(range(span_start + 1, index + 1))

                else:
                    # failed to match the current token
                    spans_to_remove.add(span_start)

            # remove impossible spans and matches from queues
            for span_start in matches_to_remove.intersection(matches):
                del matches[span_start]
            for span_start in matches_to_remove.intersection(spans):
                del spans[span_start]
            for span_start in spans_to_remove.intersection(spans):
                del spans[span_start]

            # get indices of matches and spans
            first_span = min(spans) if spans else index
            while matches:
                match_start = min(matches)
                if match_start < first_span or allow_overlapping:
                    match_end, match_sequence = matches[match_start]
                    yield Match(match_start, match_end, self.detokenizer(match_sequence))
                    del matches[match_start]
                else:
                    break

        # reached end of string, return all remaining matches
        for match_start, (match_end, match_sequence) in sorted(matches.items()):
            yield Match(match_start, match_end, self.detokenizer(match_sequence))

    def search(self,
               input_sequence: AnyStr,
               *,
               allow_overlapping: bool = False,
               ) -> Union[Match, None]:
        """
        # todo: code a special case since we don't need to track multiple matches?

        :param input_sequence: string to search
        :param allow_overlapping: if enabled, returns first shortest match; otherwise returns first longest match
        """
        for match in self.finditer(input_sequence, allow_overlapping=allow_overlapping):
            return match

    def findall(self,
                input_sequence: AnyStr,
                *,
                allow_overlapping: bool = False,
                ) -> List[AnyStr]:
        return [match.str for match in self.finditer(input_sequence, allow_overlapping=allow_overlapping)]

    def findall_longest(self,
                        input_sequence: AnyStr,
                        ) -> List[AnyStr]:

        def longest_increasing_subsequence(matches: List[Match]) -> List[Match]:
            if not matches:
                return []

            ends = [0]  # position
            values = [(0, None)]  # total length, matches as a nested 2-tuple

            # noinspection PyShadowingNames
            matches = sorted(matches, reverse=True, key=lambda m: (m.end(), m.start()))
            while matches:
                # find the match with an endpoint closest to start
                _end = matches[-1].end()
                assert _end > ends[-1]

                # take all matches with this endpoint
                _matches = []
                while matches and matches[-1].end() == _end:
                    _matches.append(matches.pop(-1))
                assert len(_matches) > 0

                # find the max total length set of matches to reach this endpoint
                best_value = (0, -_end)
                best_match = None
                # noinspection PyShadowingNames
                for _match in _matches:
                    prev_value, prev_match = values[bisect.bisect_right(ends, _match.start()) - 1]
                    new_value = (prev_value + len(_match), -_match.start())
                    if new_value > best_value:
                        best_value = new_value
                        best_match = (_match, prev_match)
                assert best_match is not None
                _matches.clear()

                # append to ends and values if it's better than the last seen value
                prev_value, prev_match = values[-1]
                if best_value[0] > prev_value:
                    ends.append(_end)
                    values.append((best_value[0], best_match))

            _out = []
            _, best_match = values[-1]
            while best_match is not None:
                _out.append(best_match[0])
                best_match = best_match[1]

            # sort and return
            _out.reverse()
            return _out

        out = []
        match_cluster = []
        match_cluster_end = 0
        for match in sorted(self.finditer(input_sequence, allow_overlapping=True),
                            key=lambda m: (m.start(), m.end())):
            if not match_cluster or match.start() < match_cluster_end:
                match_cluster.append(match)
                match_cluster_end = max(match_cluster_end, match.end())
            else:
                _match_cluster, match_cluster = match_cluster, []
                match_cluster.append(match)
                match_cluster_end = max(match_cluster_end, match.end())
                for _match in longest_increasing_subsequence(_match_cluster):
                    out.append(_match.str)
                _match_cluster.clear()

        # remaining cluster
        for match in longest_increasing_subsequence(match_cluster):
            out.append(match.str)
        match_cluster.clear()

        return out

    def translate(self, text: AnyStr) -> str:
        """
        kind of like re.sub, but you don't provide replacements because it's already defined
        >>> Trie({'yellow': 'hello'}).translate('yellow world')
        'hello world'
        """
        return self.detokenizer(token for token in self._translate_tokens(self.tokenizer(text)))

    def process_file(self, input_path, output_path, overwrite=False, encoding='utf8'):
        """
        given a path:
        1. read the file
        2. replace all the things
        3. write the output to another file

        :type input_path: str
        :type output_path: str
        :type overwrite: bool
        :type encoding: str | None
        """

        if os.path.exists(output_path) and not overwrite:
            # skip and log to screen once per thousand files
            print('skipped: %s' % output_path)
        else:
            # recursively make necessary folders
            if not os.path.isdir(os.path.dirname(output_path)):
                assert not os.path.exists(os.path.dirname(output_path))
                os.makedirs(os.path.dirname(output_path))

            # process to temp file
            print('=' * 100)
            print('processing: %s' % input_path)
            print('input size: %s' % format_bytes(os.path.getsize(input_path)))
            temp_path = output_path + '.partial'
            t0 = time.time()

            try:
                with open(temp_path, mode=('wt', 'wb')[encoding is None], encoding=encoding) as _f:
                    for output_chunk in self._translate_tokens(self._yield_tokens(input_path, encoding=encoding)):
                        _f.write(output_chunk)

                print('    output: %s' % temp_path[:-8])

            except Exception:
                os.remove(temp_path)
                print('    failed: %s' % temp_path)
                raise

            # rename to output
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_path, output_path)
            t1 = time.time()
            print('total time: %s' % format_seconds(t1 - t0))

    # I've had enough of my own typos
    from_keys = fromKeys = fromkeys
    find_first = findFirst = find = search
    find_all = findAll = findall
    find_iter = findIter = finditer
    find_longest = findLongest = findall_longest
    replace = translate


def to_regex(list_of_strings,
             case_sensitive=False,
             fuzzy_quotes=True,
             fuzzy_spaces=True,
             fffd_any=True,
             simplify=True,
             boundary=False):
    _trie = Trie.fromkeys(list_of_strings, case_sensitive=case_sensitive)
    return _trie.to_regex(fuzzy_quotes=fuzzy_quotes,
                          fuzzy_spaces=fuzzy_spaces,
                          fffd_any=fffd_any,
                          simplify=simplify,
                          boundary=boundary)


def self_test():
    # regex self-tests
    _spaces = '\t\n\v\f\r \x85\xa0\x1c\x1d\x1e\x1f\ufeff\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006' \
              '\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\u180e\u200b\u200c\u200d\u2060\u2800'
    try:
        assert set(re.sub(r'\s', '', _spaces, flags=re.U)) in [
            set('\u200b\u200c\u200d\u2060\u2800\ufeff'),
            set('\u200b\u200c\u200d\u2060\u2800\ufeff' + '\u180e')]

    except AssertionError:
        print('whatever version of re you have has wrong unicode spaces', repr(re.sub(r'\s', '', _spaces, flags=re.U)))
        raise

    # feed in a list of tuples
    _trie = Trie()
    assert len(_trie) == 0
    _trie.update([('asd', '111'), ('hjk', '222'), ('dfgh', '3333'), ('ghjkl;', '44444'), ('jkl', '!')])
    assert len(_trie) == 5
    assert ''.join(_trie.translate('erasdfghjkll')) == 'er111fg222ll'
    assert ''.join(_trie.translate('erasdfghjkl;jkl;')) == 'er111f44444!;'
    assert ''.join(_trie.translate('erassdfghjkl;jkl;')) == 'erass3333!;!;'
    assert ''.join(_trie.translate('ersdfghjkll')) == 'ers3333!l'

    # fuzz-test regex
    # a-z
    permutations = []
    for a in 'abcde':
        for b in 'abcde':
            for c in 'abcde':
                for d in 'abcde':
                    permutations.append(a + b + c + d)

    # punctuation
    for a in '`~!@#$%^&*()-=_+[]{}\\|;\':",./<>?\0':
        for b in '`~!@#$%^&*()-=_+[]{}\\|;\':",./<>?\0':
            for c in '`~!@#$%^&*()-=_+[]{}\\|;\':",./<>?\0':
                permutations.append(a + b + c)

    # run fuzzer
    for _ in range(1000):
        chosen = set()
        for i in range(10):
            chosen.add(random.choice(permutations))
        _trie = Trie.fromkeys(chosen)
        assert len(_trie) == len(chosen)
        r1 = re.compile(_trie.to_regex(fuzzy_quotes=False))  # fuzzy-matching quotes breaks this test
        for found in r1.findall(' '.join(permutations)):
            assert found in chosen
            chosen.remove(found)
        assert len(chosen) == 0

    # feed in a generator
    _trie = Trie()
    _trie.update({'a': 'b', 'b': 'c', 'c': 'd', 'd': 'a'})
    assert ''.join(_trie.translate('acbd')) == 'bdca'

    # feed in a dict
    _trie = Trie()
    _trie.update({
        'aa':                     '2',
        'aaa':                    '3',
        'aaaaaaaaaaaaaaaaaaaaaa': '~',
        'bbbb':                   '!',
    })
    assert len(_trie) == 4

    assert 'aaaaaaa' not in _trie
    _trie['aaaaaaa'] = '7'
    assert len(_trie) == 5

    assert ''.join(_trie.translate('a' * 12 + 'b' + 'a' * 28)) == '732b~33'
    assert ''.join(_trie.translate('a' * 40)) == '~773a'
    assert ''.join(_trie.translate('a' * 45)) == '~~a'
    assert ''.join(_trie.translate('a' * 25)) == '~3'
    assert ''.join(_trie.translate('a' * 60)) == '~~772'

    del _trie['bbbb']
    assert 'b' not in _trie.root
    assert len(_trie) == 4

    del _trie['aaaaaaa']
    assert len(_trie) == 3
    assert 'aaa' in _trie
    assert 'aaaaaaa' not in _trie
    assert 'aaaaaaaaaaaaaaaaaaaaaa' in _trie

    _trie['aaaa'] = 4
    assert len(_trie) == 4

    del _trie['aaaaaaaaaaaaaaaaaaaaaa']
    assert len(_trie) == 3
    assert 'aaa' in _trie
    assert 'aaaaaaa' not in _trie
    assert 'aaaaaaaaaaaaaaaaaaaaaa' not in _trie

    assert len(_trie.root['a']['a']['a']) == 1
    assert len(_trie.root['a']['a']['a']['a']) == 0

    del _trie['aaa':'bbb']
    assert _trie.to_regex() == '(?:aa)'
    assert len(_trie) == 1

    # fromkeys
    _trie = Trie.fromkeys('mad gas scar madagascar scare care car career error err are'.split())
    assert len(_trie) == 11

    test = 'madagascareerror'
    print(_trie.findall(test))
    assert list(_trie.findall(test)) == ['madagascar', 'error']
    assert list(_trie.findall(test, allow_overlapping=True)) == ['mad', 'gas', 'madagascar',
                                                                 'scar', 'car', 'scare', 'care',
                                                                 'are', 'career', 'err', 'error']

    _trie = Trie.fromkeys('to toga get her here there gather together hear the he ear'.split())
    assert len(_trie) == 12

    test = 'togethere'
    assert list(_trie.findall(test)) == ['together']
    assert list(_trie.findall(test, allow_overlapping=True)) == ['to', 'get', 'the', 'he',
                                                                 'together', 'her', 'there', 'here']

    test = 'togethear'
    assert list(_trie.findall(test)) == ['to', 'get', 'hear']
    assert list(_trie.findall(test, allow_overlapping=True)) == ['to', 'get', 'the', 'he', 'hear', 'ear']

    # test special characters
    _trie = Trie.fromkeys('| \\ \\| |\\ [ () (][) ||| *** *.* **| \\\'\\\' (?:?) \0'.split())
    assert len(_trie) == 14
    assert re.findall(_trie.to_regex(), '***|\\||||') == ['***', '|\\', '|||', '|']

    # test finditer
    _trie = Trie.fromkeys(['asdf'])
    assert len(_trie) == 1
    res = list(_trie.finditer('asdfasdfqweasdf'))
    assert len(res) == 3
    assert res[0].span() == (0, 4)
    assert res[1].span() == (4, 8)
    assert res[2].span() == (11, 15)


# if __name__ == '__main__':
#     self_test()
#
#     import psutil
#
#     # define input/output
#     input_folder = os.path.abspath('test/input')
#     output_folder = os.path.abspath('test/output')
#     file_name_pattern = '*'
#
#     # you can use a generator for the mapping to save memory space
#     mapping = []
#     with open('test/input/english-long.txt', encoding='utf8') as f:
#         for line in f:
#             line = line.strip()
#             mapping.append((line.split()[0], line.split()[-1][::-1]))
#     print('%d pairs of replacements' % len(mapping))
#
#     # parse mapping list into trie with a tokenizer
#     print('parse map to trie...')
#     t_init = datetime.datetime.now()
#     m_init = psutil.virtual_memory().used
#
#     # set tokenizer
#     from tokenizer import unicode_tokenize
#
#     trie = Trie(tokenizer=unicode_tokenize)  # words_only=False, as_tokens=False
#     trie.update(mapping, verbose=True)
#     m_end = psutil.virtual_memory().used
#     t_end = datetime.datetime.now()
#     print('parse completed!', format_seconds((t_end - t_init).total_seconds()))
#     print('memory usage:', format_bytes(m_end - m_init))
#
#     # start timer
#     t_init = datetime.datetime.now()
#     print('processing start...', t_init)
#
#     # process everything using the same tokenizer
#     for path in glob.iglob(os.path.join(input_folder, '**', file_name_pattern), recursive=True):
#         if os.path.isfile(path):
#             new_path = path.replace(input_folder, output_folder)
#             trie.process_file(path, new_path, overwrite=True)
#
#     # stop timer
#     t_end = datetime.datetime.now()
#     print('')
#     print('processing complete!', t_end)
#     print('processing total time:', format_seconds((t_end - t_init).total_seconds()))
#     print('processing total time:', (t_end - t_init))
#
#     # just find all matches, don't replace
#     t = time.time()
#     with open('test/input/kjv.txt') as f:
#         content = f.read()
#     for _ in trie.findall(content):
#         pass
#     print('find_all took this long:', format_seconds(time.time() - t))
#
#     # no tokenizer is better if you want to build a regex
#     # no tokenizer matches and replaces any substring, not just words
#     trie2 = Trie()
#     trie2.update(mapping[:1000], verbose=True)
#
#     # create regex
#     print(trie2.to_regex(boundary=True))
#     print(len(trie2.to_regex(boundary=True)))
#
#     # short code to make regex
#     print(to_regex(['bob', 'bobo', 'boba', 'baba', 'bobi']))
#     print(to_regex(['bob', 'bobo', 'boba', 'baba', 'bobi'], simplify=False))
#     print(to_regex(['pen', 'pineapple', 'apple', 'pencil']))
#     print(to_regex(['pen', 'pineapple', 'apple', 'pencil'], boundary=True))
#
#     # test space in regex
#     print(repr(to_regex(['bo b', 'bo\xa0bo', 'boba', 'ba\t ba', 'bo bi'])))


if __name__ == '__main__':
    words = sorted(line.split(',')[0].strip().lower() for line in open('words_en.txt'))
    trie = Trie.fromkeys(words)
    for i in range(3):
        print('-' * 99)
        print(i)
        t = time.time()
        res = list(trie.levenshtein_lookup('zz', i))
        print(time.time() - t)
        print(len(res), sorted(res)[:25])
