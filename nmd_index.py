from collections import Counter
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from levenshtein import damerau_levenshtein_distance as dld
from levenshtein import edit_distance as ed
from nmd import emd_1d


class ApproxWordList3:
    def __init__(self, n: Union[int, Iterable[int]] = (2, 4), case_sensitive: bool = False):
        if isinstance(n, int):
            self.__n_list = (n,)
        elif isinstance(n, Iterable):
            self.__n_list = tuple(n)
        else:
            raise TypeError(n)

        # vocabulary: word <-> word_index
        self.__vocabulary: List[str] = []  # word_index -> word
        self.__vocab_indices: Dict[str, int] = dict()  # word -> word_index

        # n-gram index (normalized vectors): n_gram -> [(word_index, (loc, loc, ...)), ...]
        self.__n_gram_indices: Dict[str, List[Tuple[int, Tuple[float, ...]]]] = dict()

        # case sensitivity
        if not isinstance(case_sensitive, (bool, int)):
            raise TypeError(case_sensitive)
        self.__case_insensitive = not case_sensitive

    @property
    def vocabulary(self) -> List[str]:
        return sorted(self.vocabulary)

    def _resolve_word_index(self, word: str, auto_add=True) -> Optional[int]:
        if not isinstance(word, str):
            raise TypeError(word)
        if len(word) == 0:
            raise ValueError(word)

        # a bit like lowercase, but more consistent for arbitrary unicode
        if self.__case_insensitive:
            word = word.casefold()

        # return if known word
        if word in self.__vocab_indices:
            return self.__vocab_indices[word]

        # do we want to add it to the vocabulary
        if not auto_add:
            return None

        # add unknown word
        _idx = self.__vocab_indices[word] = len(self.__vocabulary)
        self.__vocabulary.append(word)

        # double-check invariants before returning
        assert len(self.__vocab_indices) == len(self.__vocabulary) == _idx + 1
        assert self.__vocabulary[_idx] == word, (self.__vocabulary[_idx], word)  # check race condition
        return _idx

    def add_word(self, word: str):
        if not isinstance(word, str):
            raise TypeError(word)
        if len(word) == 0:
            raise ValueError(word)

        if self.__case_insensitive:
            word = word.casefold()

        # already contained, nothing to add
        if word in self.__vocab_indices:
            return self

        # i'll be using the STX and ETX control chars as START_TEXT and END_TEXT flags
        assert '\2' not in word and '\3' not in word, word

        word_index = self._resolve_word_index(word)

        for n in set(self.__n_list):
            if n > 1:
                # add START_TEXT and END_TEXT flags
                n_grams = [f'\2{word}\3'[i:i + n] for i in range(len(word) - n + 3)]
            else:
                # do not add START_TEXT and END_TEXT flags for 1-grams
                n_grams = list(word)

            n_gram_locations = dict()  # n_gram -> [loc, loc, ...]
            if len(n_grams) > 1:
                for idx, n_gram in enumerate(n_grams):
                    n_gram_locations.setdefault(n_gram, []).append(idx / (len(n_grams) - 1))
            elif n_grams:
                n_gram_locations.setdefault(n_grams[0], []).append(0)

            for n_gram, locations in n_gram_locations.items():
                self.__n_gram_indices.setdefault(n_gram, []).append((word_index, tuple(locations)))

        return self

    def __lookup(self, word: str, dim: Union[int, float] = 1) -> Counter:
        # count matching n-grams
        matches: Dict[int, List[float]] = dict()
        for n_idx, n in enumerate(self.__n_list):
            n_grams = [f'\2{word}\3'[i:i + n] for i in range(len(word) + 3 - n)]
            n_gram_locations = dict()
            for idx, n_gram in enumerate(n_grams):
                n_gram_locations.setdefault(n_gram, []).append(idx / (len(n_grams) - 1))

            for n_gram, locations in n_gram_locations.items():
                for other_word_index, other_locations in self.__n_gram_indices.get(n_gram, []):
                    word_scores = matches.setdefault(other_word_index, [0 for _ in range(len(self.__n_list))])
                    # should be sum not max, but this is easier to deal with
                    word_scores[n_idx] += max(len(locations), len(other_locations)) - emd_1d(locations, other_locations)

        # normalize scores
        for other_word_index, word_scores in matches.items():
            # should take other word into account too
            norm_scores = [word_scores[n_idx] / (len(word) - n + 3) for n_idx, n in enumerate(self.__n_list)]
            matches[other_word_index] = norm_scores

        # average the similarity scores
        return Counter({word_index: (sum(x ** dim for x in scores) / len(scores)) ** (1 / dim)
                        for word_index, scores in matches.items()})

    def lookup(self, word: str, top_k: int = 10, dim: Union[int, float] = 1):
        if not isinstance(word, str):
            raise TypeError(word)
        if len(word) == 0:
            raise ValueError(word)

        if self.__case_insensitive:
            word = word.casefold()

        assert '\2' not in word and '\3' not in word, word

        # average the similarity scores
        counter = self.__lookup(word, dim)
        _, top_score = counter.most_common(1)[0]

        # return only top_k results if specified (and non-zero), otherwise return all results
        if not top_k or top_k < 0:
            top_k = len(counter)

        # also return edit distances for debugging
        out = [(self.__vocabulary[word_index], round(match_score, 3),
                dld(word, self.__vocabulary[word_index]),
                ed(word, self.__vocabulary[word_index]),
                )
               for word_index, match_score in counter.most_common(top_k * 2)
               if (match_score >= top_score * 0.9) or dld(word, self.__vocabulary[word_index]) <= 1]

        return out[:top_k]


if __name__ == '__main__':
    with open('words_ms.txt', encoding='utf8') as f:
        words = set(f.read().split())
    #
    # wl_1 = ApproxWordList3((1, 2, 3, 4))
    # for word in words:
    #     wl_1.add_word(word)
    #
    # wl_2 = ApproxWordList3((2, 3, 4))
    # for word in words:
    #     wl_2.add_word(word)
    #
    # wl_3 = ApproxWordList3((3, 4))
    # for word in words:
    #     wl_3.add_word(word)

    wl_4 = ApproxWordList3((2, 4))
    for word in words:
        wl_4.add_word(word)

    with open('words_en.txt', encoding='utf8') as f:
        words = set(f.read().split())

    # wl2_1 = ApproxWordList3((1, 2, 3, 4))
    # for word in words:
    #     wl2_1.add_word(word)
    #
    # wl2_2 = ApproxWordList3((2, 3, 4))
    # for word in words:
    #     wl2_2.add_word(word)
    #
    # wl2_3 = ApproxWordList3((3, 4))
    # for word in words:
    #     wl2_3.add_word(word)

    wl2_4 = ApproxWordList3((2, 4))
    for word in words:
        wl2_4.add_word(word)

    # print(wl_4.lookup('bananananaanananananana'))
    # print(wl2_4.lookup('bananananaanananananana'))

    while True:
        word = input('word:\n')
        word = word.strip()
        if not word:
            break
        # print('wl_1', wl_1.lookup(word))
        # print('wl_2', wl_2.lookup(word))
        # print('wl_3', wl_3.lookup(word))
        print('wl_4', wl_4.lookup(word))
        # print('wl2_1', wl2_1.lookup(word))
        # print('wl2_2', wl2_2.lookup(word))
        # print('wl2_3', wl2_3.lookup(word))
        print('wl2_4', wl2_4.lookup(word))
