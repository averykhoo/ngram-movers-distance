"""
numpy-backed rewrite of the n-gram index

split into a separate file because this needs `numpy`, so (like `nmd_bow`) it isn't
exported from the `nmd` namespace by default

see `nmd.nmd_index.ApproxWordListV6` for the pure-python original, which this is
API-compatible with apart from the return type of `lookup()` (see its docstring)

measured against V6 on a 41.5k-word english vocabulary with n=(2, 4), this is roughly
20x faster per lookup and uses about a tenth of the memory. what changed:

* scoring reads flat `n_gram -> [word_index, ...]` posting arrays and accumulates with
  `np.bincount`, instead of a python loop over `(word_index, count)` tuples. this needs
  neither the `__ngram_counts` nor the `__ngram_positions` index
* the score-combining and top-k threshold steps are array ops over the whole vocabulary
  rather than a python loop over every touched word (which was V6's real bottleneck --
  emd was only ~7% of a V6 lookup)
* n-grams that occur exactly once in a word (~98% of postings) are stored as two parallel
  arrays, so their emd reduces to a vectorised `2 - abs(x - y)` with no python loop at all.
  the general dynamic-programming emd only runs for repeated n-grams
* postings live in int32/float64 arrays (12 bytes each) rather than as tuples-of-floats
  inside tuples inside lists, which is where most of V6's ~146 MB went

V6's `filter_n` prefilter is gone: it existed to keep the python candidate loop short, and
once that loop is vectorised it costs more (in memory and build time) than it saves.

there is no candidate pruning: every posting of every query n-gram is scored, exactly, by one
weighted `np.bincount` per n, and the top k is a partial sort of the resulting vocabulary-sized
score array. until 2026-09-06 a first pass computed a count-based bound per word (a word
sharing an n-gram at counts (c1, c2) has similarity in [min(c1, c2), 2 * min(c1, c2)]) and
masked out words that could not reach the top k. the bound was sound but never saved work --
the scoring pass ran over every posting regardless, and restricting it to the survivors
measured 0.81-1.09x -- so it was a second pass over the postings for nothing, 27% of a lookup
on a 300k-phrase index. `lookup` returns what an exhaustive scan would, and
tests/test_index_v7.py pins that from the outside.
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


# smallest posting group worth handing to `_emd_1d_batch`. the batched dp costs m*n numpy
# operations regardless of how many rows it serves, while the scalar dp costs m*n*rows python
# ones, so the break-even is a row count and does NOT depend on the dp shape.
#
# tuned 2026-09-06 on ermagellan (ms per query, lower better):
#
#     threshold   never    256    128     64     48     32     16   always
#     n=(2, 4)      164    162    144    142    127    124    120      162
#     n=(2, 3)      258    204    166    146    143    137    138      186
#     n=(1,)       1625   1748   1587    938    880    865    880     1947
#
# ⚠ batching unconditionally is NOT the win -- it is a wash at n=(2, 4) and a loss at n=(1,),
# where it is worse (1947 ms) than never batching at all (1625 ms) and 2.25x worse than this
# threshold,
# because within a single query n-gram the postings fragment by occurrence count into groups
# far smaller than the workload-wide average suggests. the threshold is what makes it pay.
_EMD_BATCH_MIN_ROWS = 32


def _emd_1d_batch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    `emd_1d_dp` for a whole batch of equally-shaped point sets at once

    `x` is (batch, m) and `y` is (batch, n), each row ascending; returns (batch,) distances that
    match `emd_1d_dp(x[b], y[b])` exactly. this is the same recurrence, not an approximation of
    it: `dp[i][j] = min(|x_i - y_j| + dp[i-1][j-1], 1 + dp[i-1][j], 1 + dp[i][j-1])`.

    only the batch axis is vectorised. `j` cannot be, because `dp[i][j]` depends on `dp[i][j-1]`
    within the same row -- so this is m*n numpy operations over `batch` elements, in place of
    m*n*batch python ones. that trade only pays above ~15 rows, which is why the caller groups
    postings by length first: measured 2026-09-06 on ermagellan the groups average 455 rows at
    n=(2, 4) and 75 at n=(1,).

    the dp fallback exists at all because V7's vectorised path only covers n-grams occurring
    exactly once in a document. that is ~98% of postings for 8-character words and only 25.5%
    for 138-character ones, which is why this matters on long documents and nowhere else.
    """
    batch, len_x = x.shape
    len_y = y.shape[1]
    if len_x > len_y:  # emd is symmetric; keep the outer loop over the shorter side
        x, y = y, x
        len_x, len_y = len_y, len_x

    # dp[0][j] == j, the cost of leaving j points of y unmatched
    prev = np.tile(np.arange(len_y + 1, dtype=np.float64), (batch, 1))
    curr = np.empty_like(prev)
    for i in range(1, len_x + 1):
        curr[:, 0] = float(i)  # dp[i][0] == i
        x_i = x[:, i - 1]
        for j in range(1, len_y + 1):
            match = np.abs(x_i - y[:, j - 1]) + prev[:, j - 1]
            np.minimum(match, prev[:, j] + 1.0, out=match)
            np.minimum(match, curr[:, j - 1] + 1.0, out=curr[:, j])
        prev, curr = curr, prev
    return prev[:, len_y]


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
    if dim == 2:
        # `x ** 2.0` dispatches to numpy's general power; a plain multiply is bit-identical and
        # measured 1.6x faster on a 200k array (0.51 ms -> 0.32 ms), which matters because this
        # runs over the whole vocabulary twice per lookup. `** 0.5` is left alone: it is also
        # bit-identical to `np.sqrt` but marginally *faster*, so there is nothing to win there
        out = parts[0] * parts[0]
        for part in parts[1:]:
            out += part * part
        out /= len(parts)
        return out ** 0.5
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
            change any matching decision -- only the aggregation: a shared n-gram's contribution
            is in [w * min(c1, c2), 2 * w * min(c1, c2)] for any non-negative w.

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
        # the same multi-occurrence postings flattened: n_gram -> (locations, segment offsets,
        # word indices). the dp fallback needs them as per-word tuples, but the far hotter
        # "query n-gram occurs once, indexed word repeats it" branch only needs a minimum per
        # word, which `np.minimum.reduceat` computes over the flat array in one call instead of
        # a python loop with a generator-expression `min` per posting. built in `_freeze`
        self._multi_flat: List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = [
            dict() for _ in self._n_list]
        # and the same postings grouped by occurrence count, for the "both sides repeat" branch:
        # n_gram -> [(count, word indices, (rows, count) locations, [location tuples]), ...].
        # pass 2 used to build this grouping per lookup, which on phrases -- where " e" or "e "
        # repeats in nearly every document -- was a python loop over a few hundred thousand
        # postings per query. the tuples are the same objects `_multi` holds, kept for the
        # scalar dp fallback that small groups take
        self._multi_groups: List[Dict[str, List[Tuple[int, np.ndarray, np.ndarray, list]]]] = [
            dict() for _ in self._n_list]

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
        # sqrt of whichever per-word total is in force, precomputed for the `geo` denominator:
        # sqrt(total * query_total) == sqrt(total) * sqrt(query_total), so the per-lookup work
        # drops from a vocabulary-sized sqrt to an array-scalar multiply. ⚠ that identity is not
        # exact in floating point (~2.7e-16 relative), see `_denominator`
        self._sqrt_total_by_word: List[np.ndarray] = [np.empty(0, dtype=np.float64)
                                                      for _ in self._n_list]

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
            multi_flat = self._multi_flat[n_idx]
            multi_groups = self._multi_groups[n_idx]

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

                # flatten the same multi-occurrence postings for the reduceat path. segments are
                # never empty -- an entry is only in `multi` when it holds 2+ locations -- which
                # matters because `reduceat` returns the element itself rather than an identity
                # for a zero-length segment
                if multi_entries:
                    lengths = np.fromiter((len(locs) for _, locs in multi_entries),
                                          dtype=np.int64, count=len(multi_entries))
                    offsets = np.zeros(len(multi_entries), dtype=np.int64)
                    np.cumsum(lengths[:-1], out=offsets[1:])
                    multi_idxs = np.fromiter((wi for wi, _ in multi_entries),
                                             dtype=np.int64, count=len(multi_entries))
                    flat_locs = np.concatenate([np.asarray(locs, dtype=np.float64)
                                                for _, locs in multi_entries])
                    multi_flat[n_gram] = (flat_locs, offsets, multi_idxs, lengths)

                    # group by occurrence count, preserving posting order within each group
                    # and first-seen order between groups, so that the scatter-adds in pass 2
                    # happen in exactly the order the per-lookup grouping produced
                    by_count: Dict[int, List[Tuple[int, Tuple[float, ...]]]] = dict()
                    for entry in multi_entries:
                        by_count.setdefault(len(entry[1]), []).append(entry)
                    multi_groups[n_gram] = [
                        (count,
                         np.fromiter((wi for wi, _ in group), dtype=np.int32, count=len(group)),
                         np.array([locs for _, locs in group], dtype=np.float64),
                         [locs for _, locs in group])
                        for count, group in by_count.items()
                    ]
                else:
                    multi_flat.pop(n_gram, None)
                    multi_groups.pop(n_gram, None)

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
        totals = self._weighted_total_by_word if self._weighted else self._num_grams_by_word
        self._sqrt_total_by_word = [np.sqrt(total) for total in totals]
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
                            normalize: bool,
                            dim: Union[int, float],
                            position_weight: float = 1.0,
                            denominator: str = 'dice',
                            ) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
        """
        returns (per-n similarity arrays, their power mean over n), or None if nothing matched

        every entry is the exact similarity of that word to the query; a word sharing no n-gram
        with it scores exactly 0.0. this scores every posting of every query n-gram in one
        weighted `np.bincount` per n, with the contributions concatenated in query n-gram order
        so that each word's terms are summed in the same sequence a per-n-gram scatter-add
        would have used (bincount accumulates sequentially), which keeps it bit-identical.

        ⚠ there is no pruning pass any more, on purpose. until 2026-09-06 a first pass computed a
        count-based lower bound per word, took the k-th best, and masked out words whose upper
        bound (exactly twice the lower) could not reach it. the bound was sound, but this pass
        never used it to skip work -- restricting the scoring to the surviving words was measured
        at 0.81-1.09x, because the mask gather costs what the scatter it replaces costs -- so the
        bound's only remaining effect was to shrink the final partial sort, at the price of a
        second full pass over the postings: concatenate, bincount, astype, a vocabulary-sized
        divide, a power mean and a partition, 27% of a lookup at 300k phrases. the exact scores
        select the same top k directly (a word outside the old candidate set had upper bound
        strictly below the k-th best lower bound, so it could not even tie), which is what
        `tests/test_index_v7.py::TestScoring::test_top_k_matches_an_exhaustive_scan` pins from
        the outside
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
        denominators = ([self._denominator(n_idx, query_totals[n_idx], denominator)
                         for n_idx in range(num_n)] if normalize else None)

        scores: List[np.ndarray] = []
        matched_any = False
        for n_idx in range(num_n):
            words_by_gram = self._words[n_idx]
            locs_by_gram = self._locs[n_idx]
            multi_flat = self._multi_flat[n_idx]
            multi_groups = self._multi_groups[n_idx]

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

            # (word indices, contributions) per branch per query n-gram, summed by one bincount
            # at the end. word indices are unique within one n-gram's postings, so a word gets at
            # most one term per query n-gram, and the terms arrive in query n-gram order
            idx_parts: List[np.ndarray] = []
            val_parts: List[np.ndarray] = []
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
                    #
                    # written in place: `2.0 - w * d` is `(-(w * d)) + 2.0` exactly in ieee
                    # arithmetic, and these arrays are the longest in the whole lookup (every
                    # posting of "e " in a phrase corpus), so the temporaries are not free
                    if n_single:
                        contrib = other_locs_arr - locations[0]
                        np.abs(contrib, out=contrib)
                        contrib *= -position_weight
                        contrib += 2.0
                        if weight != 1.0:
                            contrib *= weight
                        idx_parts.append(word_idxs[:n_single])
                        val_parts.append(contrib)

                    # 1-vs-m emd is min(min_dist, 2) + (m - 1), so the contribution
                    # 1 + m - emd collapses to 2 - min(min_dist, 2), independent of m.
                    #
                    # this is the hottest branch on long documents, not the dp: most query
                    # n-grams occur once in the query and several times in each document, so it
                    # runs for far more (n-gram, document) pairs than the dp does. profiled
                    # 2026-09-06 on ermagellan n=(2,4) it was 453 716 `min` and 315 668 `abs`
                    # calls over 8 queries. `reduceat` takes the per-word minimum over the
                    # flattened postings in one call instead
                    flat_entry = multi_flat.get(n_gram)
                    if flat_entry is not None:
                        flat_locs, offsets, multi_idxs, _ = flat_entry
                        min_dist = np.minimum.reduceat(np.abs(flat_locs - locations[0]), offsets)
                        np.minimum(min_dist, 2.0, out=min_dist)
                        idx_parts.append(multi_idxs)
                        val_parts.append(weight * (2.0 - position_weight * min_dist))
                else:
                    # the query n-gram repeats: same collapse with the roles swapped. the
                    # nearest query position is found with one running minimum per position,
                    # not a (postings, m) broadcast -- same values, no 2-d temporary and no
                    # reduction over a tiny axis, which measured 8.8 ms of a 78 ms lookup
                    n_locations = len(locations)
                    if n_single:
                        min_dist = np.abs(other_locs_arr - locations[0])
                        for location in locations[1:]:
                            np.minimum(min_dist, np.abs(other_locs_arr - location), out=min_dist)
                        np.minimum(min_dist, 2.0, out=min_dist)
                        idx_parts.append(word_idxs[:n_single])
                        val_parts.append(weight * (2.0 - position_weight * min_dist))

                    # both sides repeat. emd splits into the unavoidable unmatched mass
                    # abs(c1 - c2) plus the displacement of the mass that did match, and
                    # c1 + c2 - abs(c1 - c2) is exactly 2 * min(c1, c2), so this is the same
                    # form as the branches above.
                    #
                    # postings are grouped by occurrence count (in `_freeze`) so each group
                    # shares a dp shape and can be solved as one batch. both branches below
                    # need every location list ascending, which holds by construction --
                    # `add_word` and the query loop above append in increasing index order --
                    # and is pinned by `tests/test_index_v7_knobs.py::TestPresortedLocations`,
                    # because losing it would return silently wrong distances rather than raise
                    groups = multi_groups.get(n_gram)
                    if groups:
                        query_locs = np.asarray(locations, dtype=np.float64)
                        for other_count, group_idxs, other, group_locs in groups:
                            if other_count == n_locations:
                                # equal masses: nothing goes unmatched, so in one dimension the
                                # optimal transport is the monotone (sorted) matching and the
                                # displacement is just the sum of paired differences -- no dp
                                displacement = np.abs(other - query_locs).sum(axis=1)
                            elif group_idxs.size >= _EMD_BATCH_MIN_ROWS:
                                displacement = _emd_1d_batch(
                                    np.broadcast_to(query_locs,
                                                    (group_idxs.size, n_locations)), other)
                                displacement = displacement - abs(n_locations - other_count)
                            else:
                                # too few rows to amortise numpy's per-operation overhead; the
                                # batched dp costs m*n array ops either way, so a small group
                                # pays that overhead without spreading it over anything
                                displacement = np.fromiter(
                                    (emd_1d_dp(locations, locs) for locs in group_locs),
                                    dtype=np.float64, count=group_idxs.size)
                                displacement = displacement - abs(n_locations - other_count)
                            idx_parts.append(group_idxs)
                            val_parts.append(weight * (2.0 * min(n_locations, other_count)
                                                       - position_weight * displacement))

            if idx_parts:
                matched_any = True
                score = np.bincount(np.concatenate(idx_parts), weights=np.concatenate(val_parts),
                                    minlength=vocab_size)
                if normalize:
                    score /= denominators[n_idx]
            else:
                score = np.zeros(vocab_size, dtype=np.float64)
            scores.append(score)

        if not matched_any:
            return None
        return scores, _generalized_mean(scores, dim)

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
        if denominator == 'geo':
            # sqrt(total * query_total) factored as sqrt(total) * sqrt(query_total), with the
            # per-word half precomputed in `_freeze`. at a 300k vocabulary this turns a
            # vocabulary-sized sqrt into an array-scalar multiply.
            #
            # ⚠ NOT bit-identical to the unfactored form: the two differ by ~2.7e-16 relative,
            # so scores can move in the last ulp and exact ties can reorder. measured 2026-09-06
            # the retrieval effect is nil -- see the note in HANDOFF; accepted deliberately
            return np.maximum(2.0 * self._sqrt_total_by_word[n_idx] * math.sqrt(query_total), 1.0)
        totals = self._weighted_total_by_word[n_idx] if self._weighted else self._num_grams_by_word[n_idx]
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

        results are always *ranked* by similarity, because that is what the index computes;
        ties are broken alphabetically. `invert` only changes how the score is reported, as in
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

            values above 1.0 are rejected rather than clamped: `displacement <= min(c1, c2)`
            keeps the contribution within [min(c1, c2), 2 * min(c1, c2)] only while the weight
            is at most 1. past that a shared n-gram counts for less than one shared unit, and
            past 2.0 it can go negative -- a word sharing an n-gram could then score below a
            word sharing none, which the "unshared scores exactly 0 and is never returned"
            rule of `lookup` assumes cannot happen
        :param denominator: `dice` (the default) normalizes by `total_query + total_word`;
            `geo` normalizes by `2 * sqrt(total_query * total_word)`. only has any effect when
            `normalize=True`. docs/bow-plan.md part 6 measured geo above dice on product names
            at every idf exponent it tried. both are per-word positive scalars
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
            raise ValueError(position_weight)  # outside [0, 1] a shared n-gram can score < 0
        if denominator not in ('dice', 'geo'):
            raise ValueError(denominator)

        if self._case_insensitive:
            word = word.casefold()

        found = self._similarity_vectors(word, normalize, dim, position_weight, denominator)
        if found is None:
            return []
        scores, combined = found

        # partial sort over the whole vocabulary for the k-th best score, then keep every word
        # at or above it -- *including* ties at the boundary, which a bare `argpartition` would
        # split arbitrarily; the alphabetical tie-break below then decides which of them are
        # returned. words sharing no n-gram score exactly 0 and are never returned
        if combined.size > top_k:
            kth_best = combined[np.argpartition(combined, combined.size - top_k)[-top_k]]
        else:
            kth_best = 0.0
        candidate_idxs = np.flatnonzero(combined >= kth_best if kth_best > 0 else combined > 0)
        candidate_scores = combined[candidate_idxs]

        query_num_grams = [num_n_grams(len(word), n) for n in self._n_list]

        # V6 re-ran ngram_movers_distance here to break ties, but this index already produces
        # the exact score rather than an approximation of it, so that recomputation returned
        # the same number every time -- and only differed in the last bit or two of float
        # noise, which made tie order depend on summation order. sort on the word instead
        ranked = sorted(zip(candidate_idxs.tolist(), candidate_scores.tolist()),
                        key=lambda pair: (-pair[1], self._word_list[pair[0]]))[:top_k]

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
