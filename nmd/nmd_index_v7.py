"""
numpy-backed rewrite of the n-gram index

split into a separate file because this needs `numpy`, so (like `nmd_bow`) it isn't
exported from the `nmd` namespace by default

see `nmd.nmd_index.ApproxWordListV6` for the pure-python original, which this is
API-compatible with apart from the return type of `lookup()` (see its docstring)

measured against V6 on a 41.5k-word english vocabulary with n=(2, 4), this is roughly
20x faster per lookup and uses about a tenth of the memory. what changed:

* the candidate-generation pass reads a flat `n_gram -> [word_index, ...]` posting list
  and accumulates with `np.bincount`, instead of a python loop over `(word_index, count)`
  tuples. this needs neither the `__ngram_counts` nor the `__ngram_positions` index
* the score-combining and top-k threshold steps are array ops over the whole vocabulary
  rather than a python loop over every touched word (which was V6's real bottleneck --
  emd was only ~7% of a V6 lookup)
* `mean(2 * v, dim) == 2 * mean(v, dim)`, so the pruning bound is derived from the score
  that was already computed instead of recomputing a mean over a doubled vector
* n-grams that occur exactly once in a word (~98% of postings) are stored as two parallel
  arrays, so their emd reduces to a vectorised `2 - abs(x - y)` with no python loop at all.
  the general dynamic-programming emd only runs for repeated n-grams
* postings live in int32/float64 arrays (12 bytes each) rather than as tuples-of-floats
  inside tuples inside lists, which is where most of V6's ~146 MB went

V6's `filter_n` prefilter is gone: it existed to keep the python candidate loop short, and
once that loop is vectorised it costs more (in memory and build time) than it saves.

the pruning is exact, not lossy. a word sharing an n-gram at counts (c1, c2) has similarity
in [min(c1, c2), 2 * min(c1, c2)], so summed min-counts bracket the true score and the
bound can only discard words that genuinely cannot reach the top k -- `lookup` returns the
same scores an exhaustive scan would (see tests/test_index_v7.py).
"""
import math
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

from nmd.emd_1d import emd_1d_dp

__all__ = ('ApproxWordListV7',)

_START = '\2'
_END = '\3'


def n_grams(word: str, n: int) -> List[str]:
    """
    split a word into n-grams, padding with START_TEXT and END_TEXT flags

    deliberately not `lru_cache`d -- during indexing every (word, n) pair is unique, so the
    cache is pure overhead, and within a lookup we just compute each one once and reuse it
    """
    if n > 1:
        padded = f'{_START}{word}{_END}'
        return [padded[idx:idx + n] for idx in range(len(padded) - n + 1)]
    return list(word)


def num_n_grams(len_word: int, n: int) -> int:
    """
    number of n-grams a word of this length produces -- always equal to `len(n_grams(word, n))`

    note that `nmd.nmd_index.num_grams` disagrees in two cases: it over-counts by 2 at n == 1
    (it adds the START/END flags that `get_n_grams` omits for 1-grams) and it returns a
    negative count when the word is shorter than n - 2
    """
    if n > 1:
        return max(0, len_word + 3 - n)
    return len_word


def _generalized_mean(parts: Sequence[np.ndarray], dim: Union[int, float]) -> np.ndarray:
    """power mean over a list of same-shaped arrays; `dim == 1` is the plain arithmetic mean"""
    if len(parts) == 1:
        return parts[0].copy()
    if dim == 1:
        out = parts[0].copy()
        for part in parts[1:]:
            out += part
        out /= len(parts)
        return out
    out = parts[0] ** dim
    for part in parts[1:]:
        out += part ** dim
    out /= len(parts)
    return out ** (1.0 / dim)


class ApproxWordListV7:
    """
    an index over a set of words, supporting approximate lookup by n-gram mover's distance

    >>> word_list = ApproxWordListV7((2, 4))
    >>> word_list.add_words(['hello', 'yellow', 'mellow', 'bellow'])
    ApproxWordListV7(n=(2, 4), case_sensitive=False) with 4 words
    >>> [word for word, score in word_list.lookup('helo')]
    ['hello', 'bellow', 'mellow', 'yellow']

    words that score exactly the same come back in alphabetical order, so results are
    reproducible run to run (V6 returned ties in posting-list insertion order instead)
    """

    def __init__(self,
                 n: Union[int, Iterable[int]] = (2, 4),
                 case_sensitive: bool = False,
                 idf_exponent: float = 0.0,
                 ):
        """
        :param n: n-gram size(s) to index; combined 2- and 4-grams seem to work best
        :param case_sensitive: if false (the default) words are casefolded on the way in
        :param idf_exponent: weight each n-gram by `idf(n_gram) ** idf_exponent`, where
            `idf(g) = log((len(self) + 1) / document_frequency(g))`, so that rare n-grams count
            for more than common ones. defaults to 0.0, which makes every weight exactly 1.0 and
            keeps the unweighted code path, bit-for-bit.

            idf is a scalar per n-gram type and the emd is computed per type, so weighting cannot
            change any matching decision -- only the aggregation. the two-sided count bound that
            drives pruning stays valid for the same reason: a shared n-gram's contribution is in
            [w * min(c1, c2), 2 * w * min(c1, c2)] for any non-negative w.

            whether it helps is domain-dependent, and the two obvious domains disagree (measured
            in docs/bow-plan.md parts 6-7). matching product names, where a rare n-gram is usually
            the model number, gains ~+10 F1 at 1.0 and ~+17 at 2.0. correcting typos, where a rare
            n-gram is usually the typo itself, is flat at 1.0 and loses ~8 points of hit@1 at 2.0.
            above ~2 ranking degrades on both: the score ends up decided by whichever rare n-gram
            happens to be shared.
        """
        if isinstance(n, int):
            if n < 1:
                raise ValueError(n)
            self._n_list: Tuple[int, ...] = (n,)
        elif isinstance(n, Iterable):
            self._n_list = tuple(sorted(set(n)))
            if not self._n_list:
                raise ValueError(n)
            if not all(isinstance(x, int) and x > 0 for x in self._n_list):
                raise ValueError(n)
        else:
            raise TypeError(n)

        if not isinstance(case_sensitive, (bool, int)):
            raise TypeError(case_sensitive)
        self._case_insensitive = not case_sensitive

        if isinstance(idf_exponent, bool) or not isinstance(idf_exponent, (int, float)):
            raise TypeError(idf_exponent)
        if idf_exponent < 0:
            raise ValueError(idf_exponent)  # a negative exponent would up-weight the commonest n-grams
        self._idf_exponent = float(idf_exponent)
        self._weighted = self._idf_exponent != 0.0

        # vocabulary: word <-> word_index
        self._word_indices: Dict[str, int] = dict()
        self._word_list: List[str] = []
        self._word_lens: List[int] = []

        # n-grams that occur exactly once in a word -- the overwhelming majority (~98% of
        # postings), and the only ones on the vectorised scoring path. these live in numpy:
        # n_gram -> array of word_index, ordered [single-occurrence..., multi-occurrence...]
        self._words: List[Dict[str, np.ndarray]] = [dict() for _ in self._n_list]
        # n_gram -> array of location, index-aligned with the head of the above
        self._locs: List[Dict[str, np.ndarray]] = [dict() for _ in self._n_list]
        # n_gram -> [(word_index, (location, ...)), ...] for words where it occurs 2+ times.
        # kept as python objects: rare, variable-length, and only read by the dp fallback
        self._multi: List[Dict[str, List[Tuple[int, Tuple[float, ...]]]]] = [dict() for _ in self._n_list]

        # writes land here and are merged into the numpy arrays by `_freeze`, so that the
        # python-side posting lists are never retained alongside their numpy equivalent
        self._pending_words: List[Dict[str, List[int]]] = [dict() for _ in self._n_list]
        self._pending_locs: List[Dict[str, List[float]]] = [dict() for _ in self._n_list]
        self._pending_grams: List[set] = [set() for _ in self._n_list]

        self._frozen = True
        self._num_grams_by_word: List[np.ndarray] = [np.empty(0, dtype=np.float64) for _ in self._n_list]

        # idf state. document frequency needs no storage of its own: `_words[n_idx][n_gram]` already
        # holds every word index containing that n-gram, so `df == len(that array)`. both the weights
        # and the per-word weighted totals are derived in `_freeze`, because adding a word changes
        # the vocabulary size and hence every idf, not just the ones it touched
        self._gram_weights: List[Dict[str, float]] = [dict() for _ in self._n_list]
        self._weighted_total_by_word: List[np.ndarray] = [np.empty(0, dtype=np.float64) for _ in self._n_list]

    # ------------------------------------------------------------------ container protocol

    @property
    def vocabulary(self) -> List[str]:
        return sorted(self._word_list)

    @property
    def n_list(self) -> Tuple[int, ...]:
        return self._n_list

    @property
    def case_sensitive(self) -> bool:
        return not self._case_insensitive

    def __len__(self) -> int:
        return len(self._word_list)

    def __iter__(self) -> Iterator[str]:
        return iter(self._word_list)

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        if self._case_insensitive:
            word = word.casefold()
        return word in self._word_indices

    def __repr__(self) -> str:
        return (f'{type(self).__name__}(n={self._n_list}, case_sensitive={self.case_sensitive}) '
                f'with {len(self._word_list)} words')

    # ------------------------------------------------------------------ writes

    def add_word(self, word: str) -> 'ApproxWordListV7':
        """add a single word to the index; adding a word that is already indexed is a no-op"""
        if not isinstance(word, str):
            raise TypeError(word)
        if len(word) == 0:
            raise ValueError(word)
        if _START in word or _END in word:
            raise ValueError(word)  # STX and ETX are used as START_TEXT / END_TEXT flags

        if self._case_insensitive:
            word = word.casefold()
        if word in self._word_indices:
            return self

        word_index = self._word_indices[word] = len(self._word_list)
        self._word_list.append(word)
        self._word_lens.append(len(word))

        for n_idx, n in enumerate(self._n_list):
            grams = n_grams(word, n)
            locations: Dict[str, List[float]] = dict()
            if len(grams) > 1:
                position_divisor = len(grams) - 1
                for idx, n_gram in enumerate(grams):
                    locations.setdefault(n_gram, []).append(idx / position_divisor)
            elif grams:
                locations[grams[0]] = [0.0]

            pending_words = self._pending_words[n_idx]
            pending_locs = self._pending_locs[n_idx]
            pending_grams = self._pending_grams[n_idx]
            multi = self._multi[n_idx]
            for n_gram, locs in locations.items():
                if len(locs) == 1:
                    pending_words.setdefault(n_gram, []).append(word_index)
                    pending_locs.setdefault(n_gram, []).append(locs[0])
                else:
                    multi.setdefault(n_gram, []).append((word_index, tuple(locs)))
                pending_grams.add(n_gram)

        self._frozen = False
        return self

    def add_words(self, words: Iterable[str]) -> 'ApproxWordListV7':
        """
        add many words at once

        prefer this over calling `add_word` in a loop when you also interleave lookups, since
        the numpy view of the index is rebuilt on the first lookup after any write
        """
        for word in words:
            self.add_word(word)
        return self

    def _freeze(self) -> None:
        """
        merge everything written since the last freeze into the numpy arrays

        idempotent and free when there is nothing pending, so `lookup` can just call it.
        only the n-grams actually touched by those writes are rebuilt, but the per-word
        n-gram counts are recomputed over the whole vocabulary, which makes a freeze O(V) --
        hence `add_words` for bulk loading rather than add/lookup/add/lookup
        """
        if self._frozen:
            return

        for n_idx in range(len(self._n_list)):
            pending_words = self._pending_words[n_idx]
            pending_locs = self._pending_locs[n_idx]
            multi = self._multi[n_idx]
            words_by_gram = self._words[n_idx]
            locs_by_gram = self._locs[n_idx]

            for n_gram in self._pending_grams[n_idx]:
                # word indices are laid out [single-occurrence..., multi-occurrence...] so
                # that the locations array stays aligned with the head of the word array
                old_words = words_by_gram.get(n_gram)
                old_locs = locs_by_gram.get(n_gram)
                num_old_single = old_locs.size if old_locs is not None else 0

                word_parts: List[np.ndarray] = []
                loc_parts: List[np.ndarray] = []
                if num_old_single:
                    word_parts.append(old_words[:num_old_single])
                    loc_parts.append(old_locs)
                new_words = pending_words.get(n_gram)
                if new_words:
                    word_parts.append(np.array(new_words, dtype=np.int32))
                    loc_parts.append(np.array(pending_locs[n_gram], dtype=np.float64))
                # the multi-occurrence tail is small, so just rebuild it from scratch
                multi_entries = multi.get(n_gram)
                if multi_entries:
                    word_parts.append(np.fromiter((wi for wi, _ in multi_entries),
                                                  dtype=np.int32, count=len(multi_entries)))

                words_by_gram[n_gram] = (np.concatenate(word_parts) if word_parts
                                         else np.empty(0, dtype=np.int32))
                locs_by_gram[n_gram] = (np.concatenate(loc_parts) if loc_parts
                                        else np.empty(0, dtype=np.float64))

            pending_words.clear()
            pending_locs.clear()
            self._pending_grams[n_idx].clear()

        vocab_size = len(self._word_list)
        self._num_grams_by_word = [
            np.fromiter((num_n_grams(length, n) for length in self._word_lens),
                        dtype=np.float64, count=vocab_size)
            for n in self._n_list
        ]
        if self._weighted:
            self._rebuild_weights(vocab_size)
        self._frozen = True

    def _rebuild_weights(self, vocab_size: int) -> None:
        """
        recompute every n-gram weight and every per-word weighted total

        unlike the rest of `_freeze` this cannot be limited to the n-grams that were touched:
        idf has the vocabulary size in it, so adding a single word moves every weight. that makes
        a weighted freeze O(postings) rather than O(V), which is another reason to bulk-load with
        `add_words` instead of interleaving writes and lookups.
        """
        for n_idx in range(len(self._n_list)):
            words_by_gram = self._words[n_idx]
            locs_by_gram = self._locs[n_idx]
            multi = self._multi[n_idx]

            # idf is log((V + 1) / df), not the textbook log(V / df): an n-gram present in every
            # word would otherwise weigh exactly 0, so a vocabulary whose words all share their
            # n-grams would give every candidate a zero denominator and score nothing at all
            weights = {n_gram: math.log((vocab_size + 1) / word_idxs.size) ** self._idf_exponent
                       for n_gram, word_idxs in words_by_gram.items() if word_idxs.size}
            self._gram_weights[n_idx] = weights

            totals = np.zeros(vocab_size, dtype=np.float64)
            for n_gram, word_idxs in words_by_gram.items():
                weight = weights.get(n_gram)
                if not weight:
                    continue
                locs = locs_by_gram.get(n_gram)
                num_single = locs.size if locs is not None else 0
                if num_single:
                    np.add.at(totals, word_idxs[:num_single], weight)
                for other_word_index, other_locs in multi.get(n_gram, ()):
                    totals[other_word_index] += weight * len(other_locs)
            self._weighted_total_by_word[n_idx] = totals

    def _gram_weight(self, n_idx: int, n_gram: str) -> float:
        """
        weight of one n-gram; an n-gram absent from the index has df 0, and can only ever appear
        in the query's own normalization, where it takes the weight df == 1 would give
        """
        weight = self._gram_weights[n_idx].get(n_gram)
        if weight is not None:
            return weight
        return math.log(len(self._word_list) + 1) ** self._idf_exponent

    def _query_total(self, n_idx: int, query_grams: List[str], query_num_grams: int) -> float:
        """the query side of the denominator: sum of w(g) over its n-grams, counting repeats"""
        if not self._weighted:
            return query_num_grams
        return sum(self._gram_weight(n_idx, n_gram) for n_gram in query_grams)

    # ------------------------------------------------------------------ reads

    def _similarity_vectors(self,
                            word: str,
                            top_k: int,
                            normalize: bool,
                            dim: Union[int, float],
                            position_weight: float = 1.0,
                            denominator: str = 'dice',
                            ) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
        """
        returns (per-n similarity arrays, boolean candidate mask), or None if nothing matched

        the similarity arrays are only meaningful where the candidate mask is set
        """
        self._freeze()
        vocab_size = len(self._word_list)
        if vocab_size == 0:
            return None

        num_n = len(self._n_list)
        query_num_grams = [num_n_grams(len(word), n) for n in self._n_list]
        query_grams = [n_grams(word, n) for n in self._n_list]
        # the query side of the denominator; identical to query_num_grams when unweighted
        query_totals = [self._query_total(n_idx, query_grams[n_idx], query_num_grams[n_idx])
                        for n_idx in range(num_n)]

        # --- pass 1: cheap lower bound on similarity, from n-gram counts alone -------------
        # for a word sharing an n-gram with counts (c1, c2), similarity is in
        # [min(c1, c2), 2 * min(c1, c2)], so summed min-counts bound the final score
        counts: List[np.ndarray] = []
        for n_idx in range(num_n):
            words_by_gram = self._words[n_idx]
            multi = self._multi[n_idx]

            query_counts: Dict[str, int] = dict()
            for n_gram in query_grams[n_idx]:
                query_counts[n_gram] = query_counts.get(n_gram, 0) + 1

            matched = [n_gram for n_gram in query_counts if n_gram in words_by_gram]
            chunks = [words_by_gram[n_gram] for n_gram in matched]
            if not chunks:
                counts.append(np.zeros(vocab_size, dtype=np.float64))
                continue

            if self._weighted:
                # every posting of an n-gram carries that n-gram's weight, so a weighted bincount
                # is the same min-count bound with w(g) folded in. it stays two-sided, because
                # scaling each term by a non-negative constant scales both ends of its range
                posting_weights = np.concatenate(
                    [np.full(words_by_gram[n_gram].size, self._gram_weight(n_idx, n_gram))
                     for n_gram in matched])
                bincount = np.bincount(np.concatenate(chunks), weights=posting_weights,
                                       minlength=vocab_size)
            else:
                bincount = np.bincount(np.concatenate(chunks), minlength=vocab_size).astype(np.float64)
            # bincount credited every posting with 1; correct the few that should count more
            for n_gram, query_count in query_counts.items():
                if query_count > 1:
                    weight = self._gram_weight(n_idx, n_gram) if self._weighted else 1.0
                    for other_word_index, other_locs in multi.get(n_gram, ()):
                        other_count = len(other_locs)
                        bincount[other_word_index] += weight * (min(query_count, other_count) - 1)
            counts.append(bincount)

        if normalize:
            bound_parts = [counts[n_idx] / self._denominator(n_idx, query_totals[n_idx], denominator)
                           for n_idx in range(num_n)]
        else:
            bound_parts = counts
        bounds = _generalized_mean(bound_parts, dim)

        touched = bounds > 0
        if not touched.any():
            return None

        # --- prune to words that could still make the top k --------------------------------
        # the upper bound is exactly twice the lower bound, and
        # mean(2 * v, dim) == 2 * mean(v, dim), so compare against half the k-th best bound
        nonzero_bounds = bounds[touched]
        k = min(top_k, nonzero_bounds.size)
        kth_best = -np.partition(-nonzero_bounds, k - 1)[k - 1]
        candidates = touched & (bounds >= kth_best / 2)

        # --- pass 2: exact emd similarity, for the surviving candidates ---------------------
        scores = [np.zeros(vocab_size, dtype=np.float64) for _ in range(num_n)]
        for n_idx in range(num_n):
            words_by_gram = self._words[n_idx]
            locs_by_gram = self._locs[n_idx]
            multi = self._multi[n_idx]
            score = scores[n_idx]

            grams = query_grams[n_idx]
            # named for what it divides -- an n-gram's index into a 0..1 position. NOT the
            # scoring denominator: that is the `denominator` parameter, which this shadowed
            position_divisor = len(grams) - 1
            query_locations: Dict[str, List[float]] = dict()
            if position_divisor > 0:
                for idx, n_gram in enumerate(grams):
                    query_locations.setdefault(n_gram, []).append(idx / position_divisor)
            elif grams:
                query_locations[grams[0]] = [0.0]

            for n_gram, locations in query_locations.items():
                word_idxs = words_by_gram.get(n_gram)
                if word_idxs is None:
                    continue
                other_locs_arr = locs_by_gram[n_gram]
                n_single = other_locs_arr.size
                # a scalar multiplier on this n-gram type's whole contribution
                weight = self._gram_weight(n_idx, n_gram) if self._weighted else 1.0

                if len(locations) == 1:
                    # 1-vs-1 emd is just abs(x - y) (positions are in [0, 1]), so the
                    # similarity contribution 1 + 1 - emd collapses to 2 - abs(x - y), i.e.
                    # 2 * min(c1, c2) - displacement with min(c1, c2) == 1.
                    # word indices are unique within a posting list, so `+=` on a fancy
                    # index is a well-defined scatter-add
                    if n_single:
                        single_idxs = word_idxs[:n_single]
                        displacement = np.abs(other_locs_arr - locations[0])
                        score[single_idxs] += weight * (2.0 - position_weight * displacement)

                    # 1-vs-m emd is min(min_dist, 2) + (m - 1), so the contribution
                    # 1 + m - emd collapses to 2 - min(min_dist, 2), independent of m
                    location = locations[0]
                    for other_word_index, other_locs in multi.get(n_gram, ()):
                        min_dist = min(abs(location - y) for y in other_locs)
                        displacement = min_dist if min_dist < 2.0 else 2.0
                        score[other_word_index] += weight * (2.0 - position_weight * displacement)
                else:
                    # the query n-gram repeats: same collapse with the roles swapped
                    n_locations = len(locations)
                    if n_single:
                        single_idxs = word_idxs[:n_single]
                        min_dist = np.abs(other_locs_arr[:, None] - np.asarray(locations)).min(axis=1)
                        np.minimum(min_dist, 2.0, out=min_dist)
                        score[single_idxs] += weight * (2.0 - position_weight * min_dist)

                    # both sides repeat: rare enough to fall back to the general dp.
                    # emd splits into the unavoidable unmatched mass abs(c1 - c2) plus the
                    # displacement of the mass that did match, and c1 + c2 - abs(c1 - c2) is
                    # exactly 2 * min(c1, c2), so this is the same form as the branches above
                    for other_word_index, other_locs in multi.get(n_gram, ()):
                        other_count = len(other_locs)
                        displacement = (emd_1d_dp(locations, other_locs)
                                        - abs(n_locations - other_count))
                        score[other_word_index] += weight * (2.0 * min(n_locations, other_count)
                                                             - position_weight * displacement)

        if normalize:
            scores = [scores[n_idx] / self._denominator(n_idx, query_totals[n_idx], denominator)
                      for n_idx in range(num_n)]
        return scores, candidates

    def _denominator(self, n_idx: int, query_total: float, denominator: str = 'dice') -> np.ndarray:
        """
        total n-gram count of (query, indexed word), for normalizing a similarity to 0..1
        weighted by idf when `idf_exponent` is set, in which case both sides are weighted sums

        `dice` is the additive `total_query + total_word` this index has always used. `geo` is
        the cosine-shaped `2 * sqrt(total_query * total_word)`, which is never larger (AM-GM),
        so it softens the penalty for a length mismatch instead of charging the full difference

        clamped away from zero: a word shorter than n produces no n-grams at all, so its
        numerator is zero anyway and the exact divisor is irrelevant -- we only need to
        avoid a 0/0 nan leaking into the scores
        """
        totals = self._weighted_total_by_word[n_idx] if self._weighted else self._num_grams_by_word[n_idx]
        if denominator == 'geo':
            return np.maximum(2.0 * np.sqrt(totals * query_total), 1.0)
        return np.maximum(totals + query_total, 1.0)

    def lookup(self,
               word: str,
               top_k: int = 5,
               dim: Union[int, float] = 1,
               invert: bool = True,
               normalize: bool = True,
               position_weight: float = 1.0,
               denominator: str = 'dice',
               ) -> List[Tuple[str, float]]:
        """
        find the closest indexed words to `word`

        unlike `ApproxWordListV6.lookup`, this returns a plain float score per word rather
        than a tuple of (index score, recomputed nmd) -- the index score already *is* the
        exact nmd similarity, so there was nothing for the second value to add

        results are always *ranked* by similarity, because that is what the index can prune
        on soundly. `invert` only changes how the score is reported, exactly as it does in
        `ngram_movers_distance`. with `normalize=True` distance is just `1 - similarity`, so
        the reported scores increase monotonically down the list; with `normalize=False` an
        un-normalized distance also grows with word length, so the reported values are not
        monotonic in rank (a longer, more similar word can carry a larger raw distance)

        :param word: the word to look up
        :param top_k: how many results to return
        :param dim: exponent of the power mean used to combine per-n scores (1 == plain mean)
        :param invert: return similarity (default) instead of distance
        :param normalize: divide by the combined n-gram total, giving a score between 0 and 1.
            defaults to True, unlike `ApproxWordListV6.lookup` and `ngram_movers_distance`, which
            both default to False. an un-normalized score is a raw similarity sum and therefore
            grows with candidate length, so leaving it off biases the ranking toward long
            documents. measured over six retrieval benchmarks at the default `idf_exponent`
            (experiments/index_param_search.py, 2026-09-06) turning it on gained +0.339 MAP on
            long documents, +0.195 on product names, and +0.050 to +0.073 on typo correction,
            helping in 108 of 120 paired comparisons.

            ⚠ it is not a free win at every setting, because `normalize` and `idf_exponent` both
            counteract length bias and applying both over-corrects. on product names it is worth
            +0.145 MAP at `idf_exponent=0` but -0.079 at `idf_exponent=3`. if you raise the
            exponent, re-check whether you still want this on
        :param position_weight: how much of the positional displacement to charge for, in
            [0, 1]. an n-gram shared at counts (c1, c2) contributes
            `2 * min(c1, c2) - position_weight * displacement`, where `displacement` is the
            part of the mover's distance left after the unavoidable unmatched mass
            `abs(c1 - c2)`. 1.0 (the default) is n-gram mover's distance exactly, bit-for-bit;
            0.0 discards position entirely and leaves Dice over n-gram multisets, which is
            what `experiments/position_penalty.py` measured as *better* on product names
            (train AP 0.464 order-blind vs 0.441 for full nmd -- the displacement removes 7.2%
            of the score on true matches but only 6.7% on non-matches, so it costs separation).

            values above 1.0 are rejected rather than clamped, because they would break
            pruning: the two-sided bound needs the contribution to stay within
            [min(c1, c2), 2 * min(c1, c2)], and `displacement <= min(c1, c2)` only bounds
            `position_weight * displacement` from above while the weight is at most 1
        :param denominator: `dice` (the default) normalizes by `total_query + total_word`;
            `geo` normalizes by `2 * sqrt(total_query * total_word)`. only has any effect when
            `normalize=True`. docs/bow-plan.md part 6 measured geo above dice on product names
            at every idf exponent it tried. both are per-word positive scalars, so either one
            preserves pruning
        :return: [(word, score), ...], most similar first
        """
        if not isinstance(word, str):
            raise TypeError(word)
        if len(word) == 0:
            raise ValueError(word)
        if _START in word or _END in word:
            raise ValueError(word)
        if not isinstance(top_k, int):
            raise TypeError(top_k)
        if top_k <= 0:
            raise ValueError(top_k)
        if isinstance(position_weight, bool) or not isinstance(position_weight, (int, float)):
            raise TypeError(position_weight)
        if not 0.0 <= position_weight <= 1.0:
            raise ValueError(position_weight)  # outside [0, 1] the pruning bound stops holding
        if denominator not in ('dice', 'geo'):
            raise ValueError(denominator)

        if self._case_insensitive:
            word = word.casefold()

        found = self._similarity_vectors(word, top_k, normalize, dim, position_weight, denominator)
        if found is None:
            return []
        scores, candidates = found

        combined = _generalized_mean(scores, dim)
        candidate_idxs = np.flatnonzero(candidates)
        candidate_scores = combined[candidate_idxs]

        # partial sort: we only need the best top_k of what survived pruning
        if candidate_idxs.size > top_k:
            best = np.argpartition(-candidate_scores, top_k - 1)[:top_k]
            candidate_idxs = candidate_idxs[best]
            candidate_scores = candidate_scores[best]

        query_num_grams = [num_n_grams(len(word), n) for n in self._n_list]

        # V6 re-ran ngram_movers_distance here to break ties, but this index already produces
        # the exact score rather than an approximation of it, so that recomputation returned
        # the same number every time -- and only differed in the last bit or two of float
        # noise, which made tie order depend on summation order. sort on the word instead
        ranked = sorted(zip(candidate_idxs.tolist(), candidate_scores.tolist()),
                        key=lambda pair: (-pair[1], self._word_list[pair[0]]))

        if invert:
            return [(self._word_list[wi], score) for wi, score in ranked]

        # distance is defined per-n as (total n-grams) - similarity, then averaged
        out = []
        for word_index, _score in ranked:
            other_len = self._word_lens[word_index]
            if normalize:
                parts = [1.0 - scores[n_idx][word_index] for n_idx in range(len(self._n_list))]
            else:
                parts = [max(0.0, query_num_grams[n_idx] + num_n_grams(other_len, n)
                             - scores[n_idx][word_index])
                         for n_idx, n in enumerate(self._n_list)]
            out.append((self._word_list[word_index],
                        float(_generalized_mean([np.array(p) for p in parts], dim))))
        return out
