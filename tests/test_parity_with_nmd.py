"""
parity: does each index actually return the score `ngram_movers_distance` would?

this is HANDOFF item 10 -- `nmd/nmd_word_set.py`'s core scoring was untested, and neither V6
nor V7 had a check that their index score *is* the metric rather than an approximation of it.
every index here is scored against `nmd.nmd_core.ngram_movers_distance` recomputed from the
raw strings, over a sample of `experiments/words_en.txt`.

there are two different aggregations, and they are both exactly right -- for different formulas:

* **V5 / V6 / V7 return the plain mean over n of nmd itself** (`dim=1`; a power mean otherwise).
  with `normalize=True` that is the mean of the per-n normalized scores.
* **`WordSet.find_similar` pools instead**: one ratio, `sum_n similarity / sum_n n-gram count`,
  because it keeps every n in a single `ngrams_pos` dict. that is a weighted mean of the same
  per-n scores, weighted by each n's n-gram count, so it is *not* the V6/V7 number --
  `TestWordSetPoolsRatherThanAveraging` pins a case where the two differ by ~0.09.

measured 2026-09-05 on 2000 words / 30 queries, `n=(2, 4)`: V6 agrees bit-for-bit (max abs
error 0.0), V7 and V5 and `WordSet` agree to float summation noise (worst 3.6e-15), so
`TOLERANCE` below is 1e-12 with three orders of magnitude to spare.

three divergences are deliberate, and are pinned here rather than fixed:

1. **`ApproxWordListV3` does not compute nmd at all.** It adds `max(c1, c2) - emd` where nmd
   adds `c1 + c2 - emd`, and normalizes by the query's n-gram count alone. V3 is frozen
   (HANDOFF #6), so `TestV3DoesNotMatchNmd` asserts the disagreement.
2. **`WordSet` keeps the START/END markers at n == 1**, where `nmd_core` and
   `nmd_index.get_n_grams` both drop them. Two markers that match between any pair of words
   inflate every unigram score; `TestWordSetUnigramDivergence` pins it. Irrelevant to the
   `(2, 4)` default, which is why it is documented and not fixed.
3. **identical words too short to produce any n-gram of size n** score 0 for that n in V6/V7,
   where nmd's normalized score falls back to comparing the strings and returns 1.0. reachable
   only for `len(word) <= n - 3` (at n=4: a 1-character word) -- see
   `TestShortWordNormalizationDivergence`.

the frozen `ApproxWordListV5` scores correctly but still raises `ZeroDivisionError` for a query
producing exactly one n-gram (HANDOFF #7), so its parity tests use corpus-length words only.
"""
import os
import random

import pytest

from nmd import WordList
from nmd.nmd_core import ngram_movers_distance
from nmd.nmd_index import ApproxWordListV3
from nmd.nmd_index import ApproxWordListV5
from nmd.nmd_index import ApproxWordListV6
from nmd.nmd_index import num_n_grams

N_LIST = (2, 4)
TOP_K = 5
VOCAB_SIZE = 2000
NUM_QUERIES = 30
TOLERANCE = 1e-12

CORPUS = os.path.join(os.path.dirname(__file__), os.pardir, 'experiments', 'words_en.txt')


def _load_corpus():
    """a deterministic sample of the english word list, casefolded as every index would"""
    if not os.path.isfile(CORPUS):
        pytest.skip(f'corpus not found: {CORPUS}', allow_module_level=True)
    with open(CORPUS, encoding='utf8') as f:
        words = sorted({word.casefold() for word in f.read().split() if word})
    if len(words) < VOCAB_SIZE:
        pytest.skip(f'corpus has only {len(words)} words', allow_module_level=True)
    return random.Random(0).sample(words, VOCAB_SIZE)


def _make_queries(vocabulary):
    """half exact vocabulary words, half single-edit typos of one (which may leave the vocabulary)"""
    rng = random.Random(1)
    queries = [rng.choice(vocabulary) for _ in range(NUM_QUERIES // 2)]
    for _ in range(NUM_QUERIES - len(queries)):
        word = rng.choice(vocabulary)
        idx = rng.randrange(len(word))
        kind = rng.choice(('swap', 'drop', 'insert'))
        if kind == 'swap' and idx + 1 < len(word):
            queries.append(word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:])
        elif kind == 'drop' and len(word) > 1:
            queries.append(word[:idx] + word[idx + 1:])
        else:
            queries.append(word[:idx] + rng.choice('abcdefghijklmnopqrstuvwxyz') + word[idx:])
    return queries


# collected at import time rather than in a fixture, so that tests can parametrize over queries
VOCABULARY = _load_corpus()
QUERIES = _make_queries(VOCABULARY)


# ---------------------------------------------------------------- reference implementations


def nmd_per_n_mean(query: str, word: str, normalize: bool = False, invert: bool = True) -> float:
    """
    the score V5 / V6 / V7 aggregate to at `dim=1`: the plain mean over n of nmd itself

    deliberately calls `ngram_movers_distance` rather than reimplementing anything, so this
    tracks the metric even if the metric changes
    """
    return sum(ngram_movers_distance(query, word, n=n, invert=invert, normalize=normalize)
               for n in N_LIST) / len(N_LIST)


def nmd_pooled(query: str, word: str) -> float:
    """
    the score `WordSet.find_similar` aggregates to: one ratio over every n at once

    the denominator is the same total `ngram_movers_distance` normalizes by, summed over n --
    `TestReferenceMatchesNmdInternals` checks that claim against nmd directly
    """
    numerator = sum(ngram_movers_distance(query, word, n=n, invert=True, normalize=False)
                    for n in N_LIST)
    denominator = sum(num_n_grams(len(query), n) + num_n_grams(len(word), n) for n in N_LIST)
    assert denominator > 0
    return max(0.0, min(1.0, numerator / denominator))


# ---------------------------------------------------------------- indices under test


@pytest.fixture(scope='module')
def v6():
    index = WordList(N_LIST)
    for word in VOCABULARY:
        index.add_word(word)
    return index


@pytest.fixture(scope='module')
def v6_unfiltered():
    """same class with the lossy `filter_n` prefilter off, which makes its pruning exact"""
    index = WordList(N_LIST, filter_n=0)
    for word in VOCABULARY:
        index.add_word(word)
    return index


@pytest.fixture(scope='module')
def v5():
    index = ApproxWordListV5(N_LIST)
    for word in VOCABULARY:
        index.add_word(word)
    return index


@pytest.fixture(scope='module')
def v3():
    index = ApproxWordListV3(N_LIST)
    for word in VOCABULARY:
        index.add_word(word)
    return index


@pytest.fixture(scope='module')
def v7():
    pytest.importorskip('numpy')
    from nmd.nmd_index_v7 import ApproxWordListV7
    index = ApproxWordListV7(N_LIST)
    for word in VOCABULARY:
        index.add_word(word)
    return index


@pytest.fixture(scope='module')
def word_set():
    pytest.importorskip('pyroaring')
    pytest.importorskip('regex')
    from nmd.nmd_word_set import WordSet
    index = WordSet(ngram_sizes=N_LIST)
    for word in VOCABULARY:
        index.add(word)
    return index


@pytest.fixture(scope='module')
def exhaustive_top_k():
    """
    query -> the true top-k under each aggregation, by scoring the whole vocabulary

    computed once for the module: it is O(queries x vocabulary) calls into the metric
    """
    per_n, pooled = dict(), dict()
    for query in QUERIES:
        per_n[query] = sorted(((nmd_per_n_mean(query, word, normalize=True), word)
                               for word in VOCABULARY), reverse=True)[:TOP_K]
        pooled[query] = sorted(((nmd_pooled(query, word), word)
                                for word in VOCABULARY), reverse=True)[:TOP_K]
    return per_n, pooled


def _scores(results):
    """V6 returns `(index_score, recomputed_nmd)` per word where the others return a float"""
    return [score[0] if isinstance(score, tuple) else score for _word, score in results]


# ---------------------------------------------------------------- the reference itself


class TestReferenceMatchesNmdInternals:
    """the two reference aggregations above are only worth anything if they are really nmd"""

    @pytest.mark.parametrize('query', QUERIES[:5])
    def test_normalize_divides_by_the_ngram_counts(self, query):
        """`nmd_pooled`'s denominator is the same total nmd normalizes each n by"""
        for word in VOCABULARY[:200]:
            for n in N_LIST:
                raw = ngram_movers_distance(query, word, n=n, invert=True, normalize=False)
                denominator = num_n_grams(len(query), n) + num_n_grams(len(word), n)
                if denominator == 0:
                    continue
                normalized = ngram_movers_distance(query, word, n=n, invert=True, normalize=True)
                assert normalized == pytest.approx(raw / denominator, abs=TOLERANCE)

    @pytest.mark.parametrize('query', QUERIES[:5])
    def test_similarity_and_distance_are_complementary(self, query):
        """the identity nmd_core relies on, and what licenses testing `invert=False` via it"""
        for word in VOCABULARY[:200]:
            for n in N_LIST:
                similarity = ngram_movers_distance(query, word, n=n, invert=True, normalize=False)
                distance = ngram_movers_distance(query, word, n=n, invert=False, normalize=False)
                total = num_n_grams(len(query), n) + num_n_grams(len(word), n)
                assert similarity + distance == pytest.approx(total, abs=TOLERANCE)


# ---------------------------------------------------------------- the shipped WordList (V6)


class TestV6MatchesNmd:
    """`ApproxWordListV6` is what `nmd.WordList` is, so this is the shipped surface"""

    def test_wordlist_is_v6(self):
        assert WordList is ApproxWordListV6

    @pytest.mark.parametrize('query', QUERIES)
    def test_unnormalized_score_is_the_mean_over_n(self, query, v6):
        for word, (index_score, _) in v6.lookup(query, top_k=TOP_K, normalize=False):
            assert index_score == pytest.approx(nmd_per_n_mean(query, word, normalize=False),
                                                abs=TOLERANCE)

    @pytest.mark.parametrize('query', QUERIES)
    def test_normalized_score_is_the_mean_over_n(self, query, v6):
        for word, (index_score, _) in v6.lookup(query, top_k=TOP_K, normalize=True):
            assert index_score == pytest.approx(nmd_per_n_mean(query, word, normalize=True),
                                                abs=TOLERANCE)

    @pytest.mark.parametrize('normalize', [False, True])
    @pytest.mark.parametrize('query', QUERIES[:10])
    def test_second_element_is_the_2gram_nmd(self, query, normalize, v6):
        """V6 returns a tuple whose second element is a plain `ngram_movers_distance` call"""
        for word, (_, tiebreak_score) in v6.lookup(query, top_k=TOP_K, normalize=normalize):
            assert tiebreak_score == pytest.approx(
                ngram_movers_distance(query, word, n=2, invert=True, normalize=normalize),
                abs=TOLERANCE)

    @pytest.mark.parametrize('query', QUERIES)
    def test_unfiltered_lookup_returns_the_exhaustive_top_k(self, query, v6_unfiltered, exhaustive_top_k):
        """
        with `filter_n=0` the only pruning left is the two-sided count bound, which is exact,
        so the scores must be the top-k of scoring every word in the vocabulary directly

        compared as scores rather than words: words that tie are returned in an arbitrary order
        """
        per_n, _ = exhaustive_top_k
        got = _scores(v6_unfiltered.lookup(query, top_k=TOP_K, normalize=True))
        assert got == pytest.approx([score for score, _ in per_n[query]], abs=TOLERANCE)

    def test_default_filter_is_lossy_but_close(self, v6, exhaustive_top_k):
        """
        the `filter_n=3` prefilter is lossy by design -- a candidate has to share a 3-gram

        measured 2026-09-05 on this corpus sample: 146/150 of the true top-5 recovered, and 29
        of 30 queries exactly right. asserted loosely; it is a recall property, not correctness
        """
        per_n, _ = exhaustive_top_k
        hits = 0
        for query in QUERIES:
            got = {word for word, _ in v6.lookup(query, top_k=TOP_K, normalize=True)}
            hits += len(got & {word for _, word in per_n[query]})
        assert hits >= 0.9 * TOP_K * len(QUERIES)


# ---------------------------------------------------------------- V7


class TestV7MatchesNmd:
    """V7's docstring claims its index score already *is* the exact nmd similarity"""

    @pytest.mark.parametrize('normalize', [False, True])
    @pytest.mark.parametrize('query', QUERIES)
    def test_similarity_is_the_mean_over_n(self, query, normalize, v7):
        for word, score in v7.lookup(query, top_k=TOP_K, normalize=normalize, invert=True):
            assert score == pytest.approx(nmd_per_n_mean(query, word, normalize=normalize),
                                          abs=TOLERANCE)

    @pytest.mark.parametrize('normalize', [False, True])
    @pytest.mark.parametrize('query', QUERIES)
    def test_distance_is_the_mean_over_n(self, query, normalize, v7):
        """`invert=False` reports distance, which must be nmd's distance under the same mean"""
        for word, score in v7.lookup(query, top_k=TOP_K, normalize=normalize, invert=False):
            assert score == pytest.approx(
                nmd_per_n_mean(query, word, normalize=normalize, invert=False), abs=TOLERANCE)

    @pytest.mark.parametrize('query', QUERIES)
    def test_lookup_returns_the_exhaustive_top_k(self, query, v7, exhaustive_top_k):
        """V7 has no lossy prefilter at all, so its pruning must never cost a true top-k word"""
        per_n, _ = exhaustive_top_k
        got = [score for _, score in v7.lookup(query, top_k=TOP_K, normalize=True)]
        assert got == pytest.approx([score for score, _ in per_n[query]], abs=TOLERANCE)

    @pytest.mark.parametrize('query', QUERIES[:10])
    def test_agrees_with_v6(self, query, v6_unfiltered, v7):
        """both claim to be exact, so they must agree with each other as well as with nmd"""
        assert _scores(v7.lookup(query, top_k=TOP_K, normalize=True)) == pytest.approx(
            _scores(v6_unfiltered.lookup(query, top_k=TOP_K, normalize=True)), abs=TOLERANCE)


# ---------------------------------------------------------------- V5 (frozen, but exact here)


class TestV5MatchesNmd:
    """
    V5 is frozen with known bugs (HANDOFF #7), but none of them are in the score itself:
    it accumulates `c1 + c2 - emd` like nmd does, and its normalization agrees with nmd for
    every n > 1 on a word long enough to produce n-grams -- which is every corpus word here
    """

    @pytest.mark.parametrize('normalize', [False, True])
    @pytest.mark.parametrize('query', QUERIES)
    def test_score_is_the_mean_over_n(self, query, normalize, v5):
        for word, score in v5.lookup(query, top_k=TOP_K, normalize=normalize):
            assert score == pytest.approx(nmd_per_n_mean(query, word, normalize=normalize),
                                          abs=TOLERANCE)

    def test_still_raises_on_a_single_ngram_query(self):
        """
        the frozen half of the same bug V6 had fixed (HANDOFF #8) -- pins the freeze, not a fix

        a query of length exactly n - 2 produces one n-gram, and V5 divides its position by
        `len(n_grams) - 1`
        """
        index = ApproxWordListV5(4)
        for word in ('hello', 'help', 'he'):
            index.add_word(word)
        with pytest.raises(ZeroDivisionError):
            index.lookup('he', top_k=3)


# ---------------------------------------------------------------- V3 (frozen, and not nmd)


class TestV3DoesNotMatchNmd:
    """
    V3 sums `max(c1, c2) - emd` where nmd sums `c1 + c2 - emd`, and divides by the query's
    n-gram count alone rather than by both words'. it is frozen (HANDOFF #6), so the point of
    this class is that the disagreement is real and expected -- if V3 ever starts matching,
    something changed that was supposed to be frozen
    """

    def test_disagrees_on_a_worked_example(self):
        index = ApproxWordListV3(N_LIST)
        for word in ('gut', 'gutter', 'gum'):
            index.add_word(word)
        scores = {word: score for word, score, _ in index.lookup('gumt', top_k=3)}
        # measured 2026-09-05; nmd's own normalized mean for the same pairs is in the comment
        assert scores['gum'] == pytest.approx(0.441667, abs=1e-6)  # nmd: 0.519444
        assert scores['gut'] == pytest.approx(0.291667, abs=1e-6)  # nmd: 0.328704
        assert scores['gutter'] == pytest.approx(0.191667, abs=1e-6)  # nmd: 0.163194
        for word, score in scores.items():
            assert score != pytest.approx(nmd_per_n_mean('gumt', word, normalize=True), abs=1e-6)

    @pytest.mark.parametrize('query', QUERIES[:10])
    def test_disagrees_across_the_corpus(self, query, v3):
        """not a single-example fluke: on real words the gap is large, not float noise"""
        gaps = [abs(score - nmd_per_n_mean(query, word, normalize=True))
                for word, score, _ in v3.lookup(query, top_k=TOP_K)]
        assert max(gaps) > 1e-3

    @pytest.mark.parametrize('query', QUERIES[:10])
    def test_third_element_is_the_2gram_nmd(self, query, v3):
        """its debugging column, though, is a straight `ngram_movers_distance` call"""
        for word, _score, debug_score in v3.lookup(query, top_k=TOP_K):
            assert debug_score == pytest.approx(
                ngram_movers_distance(query, word, n=2, invert=True, normalize=True), abs=TOLERANCE)


# ---------------------------------------------------------------- WordSet


class TestWordSetMatchesNmd:
    """
    HANDOFF #10: `find_similar` returns the approximate score directly, with the exact
    rescoring pass commented out -- so this is the test that the approximate score is exact
    """

    @pytest.mark.parametrize('query', QUERIES)
    def test_score_is_the_pooled_nmd_ratio(self, query, word_set):
        for word, score in word_set.find_similar(query, k=TOP_K):
            assert score == pytest.approx(nmd_pooled(query, word), abs=TOLERANCE)

    @pytest.mark.parametrize('query', QUERIES)
    def test_lookup_returns_the_exhaustive_top_k(self, query, word_set, exhaustive_top_k):
        """
        the candidate filter is `min(n) == 2` here, and any shared n-gram of size 4 contains a
        shared 2-gram, so nothing with a non-zero score can be filtered out -- unlike V6's
        `filter_n=3`, this prefilter is lossless for `n=(2, 4)`
        """
        _, pooled = exhaustive_top_k
        got = [score for _, score in word_set.find_similar(query, k=TOP_K)]
        assert got == pytest.approx([score for score, _ in pooled[query]], abs=TOLERANCE)

    def test_exact_match_scores_one(self, word_set):
        for word in VOCABULARY[:20]:
            results = word_set.find_similar(word, k=1)
            assert results[0][0] == word
            assert results[0][1] == pytest.approx(1.0, abs=TOLERANCE)

    @pytest.mark.parametrize('query', QUERIES[:10])
    def test_min_similarity_filters_on_that_same_score(self, query, word_set):
        """the 2026-09-05 fix, restated in terms of nmd: the threshold cuts on the nmd ratio"""
        threshold = 0.5
        for word, score in word_set.find_similar(query, k=TOP_K, min_similarity=threshold):
            assert nmd_pooled(query, word) >= threshold - TOLERANCE

    def test_unicode_normalization_keeps_parity(self):
        """casefold + NFC happens before scoring, so parity holds against the normalized forms"""
        pytest.importorskip('pyroaring')
        pytest.importorskip('regex')
        import unicodedata
        from nmd.nmd_word_set import WordSet
        index = WordSet(ngram_sizes=N_LIST)
        for word in ('café', 'caféteria', 'CAFFEINE'):
            index.add(word)
        query = 'CAFÉ'
        normalized_query = unicodedata.normalize('NFC', query.casefold())
        for word, score in index.find_similar(query, k=3):
            normalized_word = unicodedata.normalize('NFC', word.casefold())
            assert score == pytest.approx(nmd_pooled(normalized_query, normalized_word), abs=TOLERANCE)


class TestWordSetPoolsRatherThanAveraging:
    """
    `WordSet` and V6/V7 are both exactly nmd, and still disagree, because they combine the
    per-n scores differently: V6/V7 take the mean of the per-n ratios, `WordSet` takes one
    ratio of the sums. that is a weighted mean weighted by each n's n-gram count, so the two
    coincide only when those counts are equal -- worth knowing before comparing their outputs
    """

    def test_the_two_aggregations_differ_on_a_worked_example(self):
        query, word = 'gumt', 'gut'
        assert nmd_pooled(query, word) == pytest.approx(0.422619, abs=1e-6)
        assert nmd_per_n_mean(query, word, normalize=True) == pytest.approx(0.328704, abs=1e-6)

    @pytest.mark.parametrize('query', QUERIES[:10])
    def test_word_set_follows_the_pooled_one_on_the_corpus(self, query, word_set):
        """where they differ, `WordSet` is the pooled ratio -- pinning which is which"""
        differences = []
        for word, score in word_set.find_similar(query, k=TOP_K):
            assert score == pytest.approx(nmd_pooled(query, word), abs=TOLERANCE)
            differences.append(abs(score - nmd_per_n_mean(query, word, normalize=True)))
        assert max(differences) > TOLERANCE


# ---------------------------------------------------------------- the documented divergences


class TestWordSetUnigramDivergence:
    """
    divergence 2: `WordSet._get_n_grams_list` pads with START/END for every n, including n == 1

    `nmd_core` and `nmd_index.get_n_grams` both drop the markers at n == 1 on purpose -- they
    would otherwise match between any two words. so a `WordSet` configured with 1 among its
    `ngram_sizes` scores higher than nmd does. it does not affect the `(2, 4)` default
    """

    def test_unigram_score_exceeds_nmd(self):
        pytest.importorskip('pyroaring')
        pytest.importorskip('regex')
        from nmd.nmd_word_set import WordSet
        index = WordSet(ngram_sizes=(1, 2))
        for word in ('cat', 'car', 'cot'):
            index.add(word)
        scores = dict(index.find_similar('cat', k=3))

        # nmd's own answer, pooled over n=(1, 2) the way WordSet pools
        def nmd_unigram_pooled(query, word):
            numerator = sum(ngram_movers_distance(query, word, n=n, invert=True, normalize=False)
                            for n in (1, 2))
            denominator = sum(num_n_grams(len(query), n) + num_n_grams(len(word), n) for n in (1, 2))
            return numerator / denominator

        assert scores['car'] == pytest.approx(2 / 3, abs=1e-6)
        assert nmd_unigram_pooled('cat', 'car') == pytest.approx(4 / 7, abs=1e-6)
        assert scores['car'] > nmd_unigram_pooled('cat', 'car')

    def test_default_ngram_sizes_are_unaffected(self):
        """the divergence needs n == 1 to be indexed, and the default does not index it"""
        pytest.importorskip('pyroaring')
        pytest.importorskip('regex')
        from nmd.nmd_word_set import WordSet
        assert 1 not in WordSet().ngram_sizes


class TestShortWordNormalizationDivergence:
    """
    divergence 3: a pair of identical words too short to produce any n-gram of size n

    nmd normalizes `0 / 0` by falling back to comparing the strings, so it returns 1.0 for that
    n. V6 and V7 instead clamp the denominator with `max(1, ...)`, giving 0.0 -- they have no
    reason to look at the strings, since the words only reached the scoring loop by sharing an
    n-gram at some *other* n. reachable only for `len(word) <= n - 3`
    """

    def test_nmd_returns_one_for_an_identical_short_word(self):
        assert ngram_movers_distance('a', 'a', n=4, invert=True, normalize=True) == 1.0
        assert num_n_grams(1, 4) == 0

    def test_v6_scores_that_n_as_zero_instead(self):
        index = WordList(N_LIST, filter_n=0)
        for word in ('a', 'ab', 'abc'):
            index.add_word(word)
        scores = {word: score[0] for word, score in index.lookup('a', top_k=3, normalize=True)}
        # n=2 contributes 1.0 (exact match) and n=4 contributes 0.0 rather than nmd's 1.0
        assert scores['a'] == pytest.approx(0.5, abs=TOLERANCE)
        assert nmd_per_n_mean('a', 'a', normalize=True) == pytest.approx(1.0, abs=TOLERANCE)

    def test_v7_agrees_with_v6_here(self):
        pytest.importorskip('numpy')
        from nmd.nmd_index_v7 import ApproxWordListV7
        index = ApproxWordListV7(N_LIST)
        for word in ('a', 'ab', 'abc'):
            index.add_word(word)
        scores = dict(index.lookup('a', top_k=3, normalize=True))
        assert scores['a'] == pytest.approx(0.5, abs=TOLERANCE)

    @pytest.mark.parametrize('query', ['ab', 'abc', 'abcd'])
    def test_longer_words_keep_full_parity(self, query):
        """one character is the whole of the exception at n=4; two characters is already fine"""
        index = WordList(N_LIST, filter_n=0)
        for word in ('ab', 'abc', 'abcd', 'abcde'):
            index.add_word(word)
        for word, (index_score, _) in index.lookup(query, top_k=4, normalize=True):
            assert index_score == pytest.approx(nmd_per_n_mean(query, word, normalize=True),
                                                abs=TOLERANCE)
