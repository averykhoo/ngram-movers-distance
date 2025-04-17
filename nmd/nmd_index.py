from collections import Counter
from functools import lru_cache
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

from nmd.nmd_core import emd_1d
from nmd.nmd_core import ngram_movers_distance


@lru_cache(maxsize=0xFFFF)
def get_n_grams(word: str,
                n: int,
                _start: str = '\2',
                _end: str = '\3',
                ) -> List[str]:
    if n > 1:
        word = f'{_start}{word}{_end}'
        return [word[idx:idx + n] for idx in range(len(word) - n + 1)]
    else:
        return list(word)


@lru_cache(maxsize=0xFFFF)
def num_grams(len_word, n, num_flag_chars=2) -> int:
    if n > 1:
        return len_word + num_flag_chars + 1 - n
    else:
        return len_word + num_flag_chars


def mean(vec, dim):
    return (sum(x ** dim for x in vec) / len(vec)) ** (1 / dim)


class ApproxWordListV3:
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
                n_grams = [f'\2{word}\3'[i:i + n] for i in range(num_grams(len(word), n))]
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
            n_grams = [f'\2{word}\3'[i:i + n] for i in range(num_grams(len(word), n))]
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
            norm_scores = [word_scores[n_idx] / (num_grams(len(word), n)) for n_idx, n in enumerate(self.__n_list)]
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
        out = [(self.__vocabulary[word_index],
                match_score,
                # (debugging) damerau_levenshtein_distance(word, self.__vocabulary[word_index]),
                # (debugging) edit_distance(word, self.__vocabulary[word_index]),
                ngram_movers_distance(word, self.__vocabulary[word_index], invert=True, normalize=True),
                )
               for word_index, match_score in counter.most_common(top_k * 2)]

        # print(time.time() - t)
        return out[:top_k]


class ApproxWordListV5:
    def __init__(self,
                 n: Union[int, Iterable[int]] = (2, 4),
                 case_sensitive: bool = False,
                 filter_n: int = 3,
                 ):

        # define n
        if isinstance(n, int):
            assert n > 0
            self.__n_list = (n,)
        elif isinstance(n, Iterable):
            self.__n_list = tuple(sorted(set(n)))
            assert all(isinstance(n, int) and n > 0 for n in self.__n_list)
        else:
            raise TypeError(n)

        # case sensitivity
        if not isinstance(case_sensitive, (bool, int)):
            raise TypeError(case_sensitive)
        self.__case_insensitive = not case_sensitive

        # vocabulary: word <-> word_index
        self.__word_indices: Dict[str, int] = dict()  # word -> word_index
        self.__word_list: List[str] = []  # word_index -> word
        self.__word_lens: List[int] = []  # word_index -> len(word)
        self.__word_num_grams: List[Tuple[int]] = []  # word_index -> [num_grams(len(word), n) for n in self.__n_list]

        # n-gram filter: n_gram -> {word_index, ...}
        self.__filter_n: int = filter_n
        self.__ngram_filter: Dict[str, Set[int]] = dict()

        # n-gram counts: n_gram -> [(word_index, count), ...]
        self.__ngram_counts: Dict[str, List[Tuple[int, int]]] = dict()

        # n-gram normalized positions: n_gram -> [(word_index, (loc, loc, ...)), ...]
        self.__ngram_positions: Dict[str, List[Tuple[int, Tuple[float, ...]]]] = dict()

    @property
    def vocabulary(self) -> List[str]:
        return sorted(self.__word_list)

    def _resolve_word_index(self, word: str, auto_add=True) -> Optional[int]:
        if not isinstance(word, str):
            raise TypeError(word)
        if len(word) == 0:
            raise ValueError(word)

        # casefold is a bit like lowercase, but more consistent for arbitrary unicode
        if self.__case_insensitive:
            word = word.casefold()

        # return if known word
        if word in self.__word_indices:
            return self.__word_indices[word]

        # word not found, do we want to add it to the vocabulary
        if not auto_add:
            return None

        # add unknown word
        _idx = self.__word_indices[word] = len(self.__word_list)
        self.__word_list.append(word)
        self.__word_lens.append(len(word))
        self.__word_num_grams.append(tuple(num_grams(len(word), n) for n in self.__n_list))

        # double-check invariants before returning to make sure we didn't trigger some race condition
        assert len(self.__word_indices) == len(self.__word_list) == _idx + 1
        assert self.__word_list[_idx] == word, (self.__word_list[_idx], word)
        assert self.__word_lens[_idx] == len(word), (self.__word_lens[_idx], word)
        return _idx

    def add_word(self, word: str):
        if not isinstance(word, str):
            raise TypeError(word)
        if len(word) == 0:
            raise ValueError(word)

        if self.__case_insensitive:
            word = word.casefold()

        # already contained, nothing to add
        if word in self.__word_indices:
            return self

        # i'll be using the STX and ETX control chars as START_TEXT and END_TEXT flags
        assert '\2' not in word and '\3' not in word, word

        word_index = self._resolve_word_index(word)

        if self.__filter_n:
            for n_gram in set(get_n_grams(word, self.__filter_n)):
                self.__ngram_filter.setdefault(n_gram, set()).add(word_index)

        for n in set(self.__n_list):
            n_grams = get_n_grams(word, n)

            n_gram_locations = dict()  # n_gram -> [loc, loc, ...]
            if len(n_grams) > 1:
                for idx, n_gram in enumerate(n_grams):
                    n_gram_locations.setdefault(n_gram, []).append(idx / (len(n_grams) - 1))
            elif n_grams:
                n_gram_locations.setdefault(n_grams[0], []).append(0)

            for n_gram, locations in n_gram_locations.items():
                assert len(locations) > 0
                self.__ngram_counts.setdefault(n_gram, []).append((word_index, len(locations)))
                self.__ngram_positions.setdefault(n_gram, []).append((word_index, tuple(locations)))

        return self

    def __lookup_similarity(self,
                            word: str,
                            dim: Union[int, float],
                            top_k: int,
                            normalize: bool,
                            ) -> Counter:
        # todo: use existing indices and calculate cosine distance
        # todo: use existing indices and calculate jaccard

        assert isinstance(top_k, int) and top_k >= 1
        len_word = len(word)

        possible_word_indices = set()
        if self.__filter_n:
            for n_gram in set(get_n_grams(word, self.__filter_n)):
                possible_word_indices.update(self.__ngram_filter.get(n_gram, set()))

        # count matching n-grams
        min_scores: Dict[int, List[int]] = dict()  # word_index -> [count_for_n, ...]
        for n_idx, n in enumerate(self.__n_list):
            for n_gram, count in Counter(get_n_grams(word, n)).items():
                for other_word_index, other_count in self.__ngram_counts.get(n_gram, []):
                    if self.__filter_n and other_word_index not in possible_word_indices:
                        continue
                    if other_word_index not in min_scores:
                        min_scores[other_word_index] = [0] * len(self.__n_list)
                    if normalize:
                        denominator = num_grams(len_word, n) + self.__word_num_grams[other_word_index][n_idx]
                        min_scores[other_word_index][n_idx] += min(count, other_count) / denominator
                    else:
                        min_scores[other_word_index][n_idx] += min(count, other_count)

        # no results, return empty Counter
        if not min_scores:
            return Counter()

        # get min possible score per word
        _scores = [mean(scores, dim=dim) for scores in min_scores.values()]
        _scores.sort(reverse=True)
        min_acceptable_score = _scores[min(top_k, len(_scores)) - 1]

        # filter to possible top_k by max possible score
        # max score per n-gram is exactly 2x min possible score
        possible_word_indices = set()
        for word_index, scores in min_scores.items():
            if mean([x * 2 for x in scores], dim) >= min_acceptable_score:
                possible_word_indices.add(word_index)

        # count matching n-grams
        matches: Dict[int, List[float]] = dict()
        for n_idx, n in enumerate(self.__n_list):
            n_grams = get_n_grams(word, n)
            n_gram_locations = dict()
            for idx, n_gram in enumerate(n_grams):
                n_gram_locations.setdefault(n_gram, []).append(idx / (len(n_grams) - 1))

            for n_gram, locations in n_gram_locations.items():
                for other_word_index, other_locations in self.__ngram_positions.get(n_gram, []):
                    if other_word_index not in possible_word_indices:
                        continue
                    word_scores = matches.setdefault(other_word_index, [0 for _ in range(len(self.__n_list))])
                    word_scores[n_idx] += len(locations) + len(other_locations)
                    word_scores[n_idx] -= emd_1d(locations, other_locations)

        # normalize scores
        for other_word_index, word_scores in matches.items():
            other_len = self.__word_lens[other_word_index]

            # should take other word into account too
            if normalize:
                norm_scores = [word_scores[n_idx] / (num_grams(len_word, n) + num_grams(other_len, n))
                               for n_idx, n in enumerate(self.__n_list)]
                matches[other_word_index] = norm_scores
            else:
                matches[other_word_index] = word_scores

        # average the similarity scores
        return Counter({word_index: mean(scores, dim) for word_index, scores in matches.items()})

    def lookup(self,
               word: str,
               top_k: int = 5,
               dim: Union[int, float] = 1,
               invert: bool = True,
               normalize: bool = False,
               ):
        if not isinstance(word, str):
            raise TypeError(word)
        if len(word) == 0:
            raise ValueError(word)
        if not isinstance(top_k, int):
            raise TypeError(top_k)
        if top_k <= 0:
            raise ValueError(top_k)

        if self.__case_insensitive:
            word = word.casefold()

        assert '\2' not in word and '\3' not in word, word

        # average the similarity scores
        word_scores = self.__lookup_similarity(word, dim, top_k, normalize).most_common(top_k * 2)

        # also return edit distances for debugging
        out = [(self.__word_list[word_index],  # word
                match_score if invert else normalize - match_score,  # lookup result
                # (debugging) damerau_levenshtein_distance(word, self.__word_list[word_index]),
                # (debugging) edit_distance(word, self.__word_list[word_index]),
                # ngram_movers_distance(word, self.__word_list[word_index], invert=invert, normalize=normalize),
                )
               for word_index, match_score in word_scores]

        # print(time.time() - t)
        return sorted(out, key=lambda x: x[1:], reverse=True)


class ApproxWordListV6:
    def __init__(self,
                 n: Union[int, Iterable[int]] = (2, 4),
                 case_sensitive: bool = False,
                 filter_n: int = 3,
                 ):

        # define n
        if isinstance(n, int):
            assert n > 0
            self.__n_list = (n,)
        elif isinstance(n, Iterable):
            self.__n_list = tuple(sorted(set(n)))
            assert all(isinstance(n, int) and n > 0 for n in self.__n_list)
        else:
            raise TypeError(n)

        # case sensitivity
        if not isinstance(case_sensitive, (bool, int)):
            raise TypeError(case_sensitive)
        self.__case_insensitive = not case_sensitive

        # vocabulary: word <-> word_index
        self.__word_indices: Dict[str, int] = dict()  # word -> word_index
        self.__word_list: List[str] = []  # word_index -> word
        self.__word_lens: List[int] = []  # word_index -> len(word)
        self.__word_num_grams: List[Tuple[int]] = []  # word_index -> [num_grams(len(word), n) for n in self.__n_list]

        # n-gram filter: n_gram -> {word_index, ...}
        self.__filter_n: int = filter_n
        self.__ngram_filter: Dict[str, Set[int]] = dict()

        # n-gram counts: n_gram -> [(word_index, count), ...]
        self.__ngram_counts: Dict[str, List[Tuple[int, int]]] = dict()

        # n-gram normalized positions: n_gram -> [(word_index, (loc, loc, ...)), ...]
        self.__ngram_positions: Dict[str, List[Tuple[int, Tuple[float, ...]]]] = dict()

    @property
    def vocabulary(self) -> List[str]:
        return sorted(self.__word_list)

    def _resolve_word_index(self, word: str, auto_add=True) -> Optional[int]:
        if not isinstance(word, str):
            raise TypeError(word)
        if len(word) == 0:
            raise ValueError(word)

        # casefold is a bit like lowercase, but more consistent for arbitrary unicode
        if self.__case_insensitive:
            word = word.casefold()

        # return if known word
        if word in self.__word_indices:
            return self.__word_indices[word]

        # word not found, do we want to add it to the vocabulary
        if not auto_add:
            return None

        # add unknown word
        _idx = self.__word_indices[word] = len(self.__word_list)
        self.__word_list.append(word)
        self.__word_lens.append(len(word))
        self.__word_num_grams.append(tuple(num_grams(len(word), n) for n in self.__n_list))

        # double-check invariants before returning to make sure we didn't trigger some race condition
        assert len(self.__word_indices) == len(self.__word_list) == _idx + 1
        assert self.__word_list[_idx] == word, (self.__word_list[_idx], word)
        assert self.__word_lens[_idx] == len(word), (self.__word_lens[_idx], word)
        return _idx

    def add_word(self, word: str):
        if not isinstance(word, str):
            raise TypeError(word)
        if len(word) == 0:
            raise ValueError(word)

        if self.__case_insensitive:
            word = word.casefold()

        # already contained, nothing to add
        if word in self.__word_indices:
            return self

        # i'll be using the STX and ETX control chars as START_TEXT and END_TEXT flags
        assert '\2' not in word and '\3' not in word, word

        word_index = self._resolve_word_index(word)

        if self.__filter_n:
            for n_gram in set(get_n_grams(word, self.__filter_n)):
                self.__ngram_filter.setdefault(n_gram, set()).add(word_index)

        for n in set(self.__n_list):
            n_grams = get_n_grams(word, n)

            n_gram_locations = dict()  # n_gram -> [loc, loc, ...]
            if len(n_grams) > 1:
                for idx, n_gram in enumerate(n_grams):
                    n_gram_locations.setdefault(n_gram, []).append(idx / (len(n_grams) - 1))
            elif n_grams:
                n_gram_locations.setdefault(n_grams[0], []).append(0)

            for n_gram, locations in n_gram_locations.items():
                assert len(locations) > 0
                self.__ngram_counts.setdefault(n_gram, []).append((word_index, len(locations)))
                self.__ngram_positions.setdefault(n_gram, []).append((word_index, tuple(locations)))

        return self

    def __lookup_similarity(self,
                            word: str,
                            dim: Union[int, float],
                            top_k: int,
                            normalize: bool,
                            ) -> Counter:
        # todo: why am i getting top_k * 2 results? this should be documented somewhere
        # todo: use existing indices and calculate cosine distance
        # todo: use existing indices and calculate jaccard

        assert isinstance(top_k, int) and top_k >= 1
        len_word = len(word)

        possible_word_indices = set()
        if self.__filter_n:
            for n_gram in set(get_n_grams(word, self.__filter_n)):
                possible_word_indices.update(self.__ngram_filter.get(n_gram, set()))

        # count matching n-grams
        min_scores: Dict[int, List[int]] = dict()  # word_index -> [count_for_n, ...]
        for n_idx, n in enumerate(self.__n_list):
            for n_gram, count in Counter(get_n_grams(word, n)).items():
                for other_word_index, other_count in self.__ngram_counts.get(n_gram, []):
                    if self.__filter_n and other_word_index not in possible_word_indices:
                        continue
                    if other_word_index not in min_scores:
                        min_scores[other_word_index] = [0] * len(self.__n_list)
                    if normalize:
                        denominator = num_grams(len_word, n) + self.__word_num_grams[other_word_index][n_idx]
                        min_scores[other_word_index][n_idx] += min(count, other_count) / denominator
                    else:
                        min_scores[other_word_index][n_idx] += min(count, other_count)

        # no results, return empty Counter
        if not min_scores:
            return Counter()

        # get min possible score per word
        _scores = [mean(scores, dim=dim) for scores in min_scores.values()]
        _scores.sort(reverse=True)
        min_acceptable_score = _scores[min(top_k, len(_scores)) - 1]

        # filter to possible top_k by max possible score
        # max score per n-gram is exactly 2x min possible score
        possible_word_indices = set()
        for word_index, scores in min_scores.items():
            if mean([x * 2 for x in scores], dim) >= min_acceptable_score:
                possible_word_indices.add(word_index)

        # count matching n-grams
        matches: Dict[int, List[float]] = dict()
        for n_idx, n in enumerate(self.__n_list):
            n_grams = get_n_grams(word, n)
            n_gram_locations = dict()
            for idx, n_gram in enumerate(n_grams):
                n_gram_locations.setdefault(n_gram, []).append(idx / (len(n_grams) - 1))

            for n_gram, locations in n_gram_locations.items():
                for other_word_index, other_locations in self.__ngram_positions.get(n_gram, []):
                    if other_word_index not in possible_word_indices:
                        continue
                    word_scores = matches.setdefault(other_word_index, [0 for _ in range(len(self.__n_list))])
                    word_scores[n_idx] += len(locations) + len(other_locations)
                    word_scores[n_idx] -= emd_1d(locations, other_locations)

        # normalize scores
        for other_word_index, word_scores in matches.items():
            other_len = self.__word_lens[other_word_index]

            # should take other word into account too
            if normalize:
                norm_scores = [word_scores[n_idx] / (num_grams(len_word, n) + num_grams(other_len, n))
                               for n_idx, n in enumerate(self.__n_list)]
                matches[other_word_index] = norm_scores
            else:
                matches[other_word_index] = word_scores

        # average the similarity scores
        return Counter({word_index: mean(scores, dim) for word_index, scores in matches.items()})

    def lookup(self,
               word: str,
               top_k: int = 5,
               dim: Union[int, float] = 1,
               normalize: bool = False,
               ):
        # todo: maybe avoid re-calculating the nmd score? the approximation should be good enough

        if not isinstance(word, str):
            raise TypeError(word)
        if len(word) == 0:
            raise ValueError(word)
        if not isinstance(top_k, int):
            raise TypeError(top_k)
        if top_k <= 0:
            raise ValueError(top_k)

        if self.__case_insensitive:
            word = word.casefold()

        assert '\2' not in word and '\3' not in word, word

        # also sort by 2-gram nmd as a second level
        counter = Counter()
        for word_index, match_score in self.__lookup_similarity(word, dim, top_k, normalize).most_common(top_k * 2):
            _similarity = ngram_movers_distance(word, self.__word_list[word_index], invert=True, normalize=normalize)
            counter[word_index] = (match_score, _similarity)

        return counter.most_common(top_k)
        return [self.__word_list[word_index] for word_index, _ in counter.most_common(top_k)]


WordList = ApproxWordListV6
