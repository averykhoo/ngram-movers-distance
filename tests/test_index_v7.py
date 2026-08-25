"""
Tests for the numpy-backed ApproxWordListV7 index.

The important property is that V7's approximate index score equals the exact n-gram
mover's distance similarity it is meant to approximate -- the pruning it does to avoid
scoring the whole vocabulary is supposed to be exact, not lossy.
"""
import random

import pytest

from nmd.emd_1d import emd_1d_dp
from nmd.nmd_index import get_n_grams
from nmd.nmd_index import num_grams

np = pytest.importorskip('numpy')

from nmd.nmd_index_v7 import ApproxWordListV7  # noqa: E402
from nmd.nmd_index_v7 import n_grams  # noqa: E402
from nmd.nmd_index_v7 import num_n_grams  # noqa: E402

WORDS = [
    'hello', 'yellow', 'mellow', 'bellow', 'fellow', 'hell', 'hollow', 'follow',
    'below', 'billow', 'willow', 'pillow', 'shallow', 'swallow', 'wallow', 'callow',
    'assalamualaikum', 'waalaikumsalam', 'banana', 'bandana', 'bandanna', 'ananas',
    'a', 'ab', 'abc', 'abcd', 'abababab', 'ababab', 'aaaa',
]


def exact_similarity(word_1, word_2, n_list, normalize):
    """brute-force nmd similarity over several n, averaged -- what V7's score should equal"""
    parts = []
    for n in n_list:
        grams_1, grams_2 = n_grams(word_1, n), n_grams(word_2, n)
        locs_1, locs_2 = {}, {}
        for grams, locs in ((grams_1, locs_1), (grams_2, locs_2)):
            denominator = max(1, len(grams) - 1)
            for idx, n_gram in enumerate(grams):
                locs.setdefault(n_gram, []).append(idx / denominator)
        similarity = 0.0
        for n_gram, positions_1 in locs_1.items():
            positions_2 = locs_2.get(n_gram)
            if positions_2 is not None:
                similarity += len(positions_1) + len(positions_2) - emd_1d_dp(positions_1, positions_2)
        if normalize:
            similarity /= max(1, len(grams_1) + len(grams_2))
        parts.append(similarity)
    return sum(parts) / len(parts)


class TestNGramHelpers:
    @pytest.mark.parametrize('word', ['a', 'ab', 'hello', 'assalamualaikum', 'x' * 40])
    @pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
    def test_n_grams_matches_v6(self, word, n):
        assert n_grams(word, n) == get_n_grams(word, n)

    @pytest.mark.parametrize('word', ['a', 'ab', 'hello', 'assalamualaikum', 'x' * 40])
    @pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
    def test_num_n_grams_is_consistent_with_n_grams(self, word, n):
        """unlike V6's num_grams, this must always equal the actual number of n-grams"""
        assert num_n_grams(len(word), n) == len(n_grams(word, n))

    def test_v6_num_grams_disagrees_at_n_1(self):
        """regression guard: V6 over-counts 1-grams by the two flag chars it never adds"""
        assert len(get_n_grams('hello', 1)) == 5
        assert num_grams(5, 1) == 7
        assert num_n_grams(5, 1) == 5

    def test_num_n_grams_never_negative(self):
        """V6's num_grams returns -1 here, which would corrupt a normalized score"""
        assert num_grams(1, 5) < 0
        assert num_n_grams(1, 5) == 0


class TestContainerProtocol:
    def test_empty_lookup_returns_empty(self):
        assert ApproxWordListV7((2, 4)).lookup('anything') == []

    def test_casefolds_by_default(self):
        word_list = ApproxWordListV7((2, 4)).add_words(['Hello', 'hello', 'HELLO'])
        assert len(word_list) == 1
        assert word_list.vocabulary == ['hello']
        assert 'hELLo' in word_list
        assert 'nope' not in word_list

    def test_case_sensitive(self):
        word_list = ApproxWordListV7((2, 4), case_sensitive=True).add_words(['Hello', 'hello'])
        assert len(word_list) == 2
        assert 'Hello' in word_list
        assert 'HELLO' not in word_list

    def test_contains_non_string(self):
        assert 123 not in ApproxWordListV7(2).add_words(['abc'])

    def test_iter_and_len(self):
        word_list = ApproxWordListV7(2).add_words(['abc', 'def'])
        assert len(word_list) == 2
        assert sorted(word_list) == ['abc', 'def']

    def test_repr(self):
        assert 'ApproxWordListV7' in repr(ApproxWordListV7(2))

    @pytest.mark.parametrize('bad, exc', [(123, TypeError), (None, TypeError),
                                          ('', ValueError), ('a\2b', ValueError), ('a\3b', ValueError)])
    def test_add_word_rejects(self, bad, exc):
        with pytest.raises(exc):
            ApproxWordListV7(2).add_word(bad)

    @pytest.mark.parametrize('bad, exc', [(123, TypeError), ('', ValueError), ('a\2b', ValueError)])
    def test_lookup_rejects(self, bad, exc):
        with pytest.raises(exc):
            ApproxWordListV7(2).add_words(['abc']).lookup(bad)

    @pytest.mark.parametrize('top_k, exc', [(0, ValueError), (-1, ValueError), (1.5, TypeError)])
    def test_lookup_rejects_bad_top_k(self, top_k, exc):
        with pytest.raises(exc):
            ApproxWordListV7(2).add_words(['abc']).lookup('abc', top_k=top_k)

    @pytest.mark.parametrize('bad', [0, -1, (), (0,), (2, -1), 'x'])
    def test_init_rejects_bad_n(self, bad):
        with pytest.raises((ValueError, TypeError)):
            ApproxWordListV7(bad)

    def test_interleaved_add_and_lookup(self):
        """the numpy view is built lazily, so it must be invalidated by a later add"""
        word_list = ApproxWordListV7(2).add_words(['aaa'])
        assert [w for w, _ in word_list.lookup('aaa')] == ['aaa']
        word_list.add_word('bbb')
        assert 'bbb' in [w for w, _ in word_list.lookup('bbb', top_k=2)]

    def test_add_word_is_idempotent(self):
        word_list = ApproxWordListV7(2).add_words(['abc'])
        word_list.add_word('abc')
        assert len(word_list) == 1

    @pytest.mark.parametrize('n_list', [(2,), (2, 4)])
    def test_incremental_build_matches_bulk_build(self, n_list):
        """
        a lookup after every single add forces a freeze after every add, so the merge path
        in _freeze runs ~30 times instead of once -- it must end up identical to one freeze
        """
        bulk = ApproxWordListV7(n_list).add_words(WORDS)
        incremental = ApproxWordListV7(n_list)
        for word in WORDS:
            incremental.add_word(word)
            incremental.lookup('helo', top_k=3)

        assert incremental.vocabulary == bulk.vocabulary
        for query in ['helo', 'banan', 'abab', 'ab', 'assalamalaikum']:
            for normalize in (False, True):
                assert (incremental.lookup(query, top_k=8, normalize=normalize)
                        == bulk.lookup(query, top_k=8, normalize=normalize)), (query, normalize)

    def test_incremental_build_keeps_repeated_ngrams_aligned(self):
        """
        the merge splices new single-occurrence postings between the old ones and the
        repeated-n-gram tail, so a word with repeats added mid-stream must stay correct
        """
        word_list = ApproxWordListV7(2)
        for word in ['abcd', 'abab', 'bcde', 'ababab', 'cdef']:
            word_list.add_word(word)
            word_list.lookup('ab', top_k=2)
        for word, score in word_list.lookup('abab', top_k=5):
            assert score == pytest.approx(exact_similarity('abab', word, (2,), False), abs=1e-12)

    def test_freeze_does_not_retain_python_posting_lists(self):
        """the numpy arrays replace the python lists rather than sitting alongside them"""
        word_list = ApproxWordListV7((2, 4)).add_words(WORDS)
        word_list.lookup('helo')
        assert all(not pending for pending in word_list._pending_words)
        assert all(not pending for pending in word_list._pending_locs)
        assert all(not pending for pending in word_list._pending_grams)


class TestScoring:
    @pytest.mark.parametrize('n_list', [(2,), (3,), (2, 4), (1, 2, 3)])
    @pytest.mark.parametrize('normalize', [False, True])
    def test_score_equals_exact_nmd(self, n_list, normalize):
        """
        the whole point of the index: its score is the real thing, not an approximation.
        the count-based pruning bound is exact, so it must never change a score.
        """
        word_list = ApproxWordListV7(n_list).add_words(WORDS)
        for query in ['helo', 'yelow', 'banan', 'abab', 'a', 'ab', 'assalamalaikum']:
            for word, score in word_list.lookup(query, top_k=8, normalize=normalize):
                expected = exact_similarity(query, word, n_list, normalize)
                assert score == pytest.approx(expected, abs=1e-12), (query, word)

    @pytest.mark.parametrize('normalize', [False, True])
    def test_pruning_never_drops_a_true_top_k(self, normalize):
        """compare against an exhaustive scan of the vocabulary"""
        n_list = (2, 4)
        random.seed(12345)
        vocab = WORDS + [''.join(random.choice('abcdefg') for _ in range(random.randint(2, 9)))
                         for _ in range(400)]
        vocab = sorted(set(vocab))
        word_list = ApproxWordListV7(n_list).add_words(vocab)
        for query in ['helo', 'abcdef', 'gfedcba', 'banan', 'ab', 'aaaa']:
            got = word_list.lookup(query, top_k=5, normalize=normalize)
            brute = sorted(((exact_similarity(query, w, n_list, normalize), w) for w in vocab),
                           key=lambda pair: (-pair[0], pair[1]))
            # scores must match rank for rank, even where ties make the words differ
            assert [s for _, s in got] == pytest.approx([s for s, _ in brute[:len(got)]], abs=1e-12)

    def test_exact_match_scores_highest(self):
        word_list = ApproxWordListV7((2, 4)).add_words(WORDS)
        for word in ['hello', 'banana', 'assalamualaikum']:
            assert word_list.lookup(word, top_k=1)[0][0] == word

    def test_normalized_score_in_unit_range(self):
        word_list = ApproxWordListV7((2, 4)).add_words(WORDS)
        for word, score in word_list.lookup('helo', top_k=10, normalize=True):
            assert 0.0 <= score <= 1.0
        assert word_list.lookup('hello', top_k=1, normalize=True)[0][1] == pytest.approx(1.0)

    def test_ties_are_ordered_alphabetically(self):
        word_list = ApproxWordListV7((2, 4)).add_words(['hello', 'yellow', 'mellow', 'bellow'])
        assert [w for w, _ in word_list.lookup('helo')] == ['hello', 'bellow', 'mellow', 'yellow']

    @pytest.mark.parametrize('dim', [1, 2, 3, 0.5])
    def test_dim_is_a_power_mean(self, dim):
        n_list = (2, 4)
        word_list = ApproxWordListV7(n_list).add_words(WORDS)
        for word, score in word_list.lookup('helo', top_k=5, dim=dim, normalize=True):
            parts = [exact_similarity('helo', word, (n,), True) for n in n_list]
            expected = (sum(p ** dim for p in parts) / len(parts)) ** (1 / dim)
            assert score == pytest.approx(expected, abs=1e-12)

    def test_repeated_ngrams_on_both_sides(self):
        """exercises the dynamic-programming fallback rather than the vectorised path"""
        word_list = ApproxWordListV7(2).add_words(['abababab', 'ababab', 'ab', 'ba', 'aaaa'])
        assert word_list.lookup('ababab', top_k=1)[0][0] == 'ababab'
        for word, score in word_list.lookup('abab', top_k=5):
            assert score == pytest.approx(exact_similarity('abab', word, (2,), False), abs=1e-12)

    def test_top_k_limits_results(self):
        word_list = ApproxWordListV7((2, 4)).add_words(WORDS)
        assert len(word_list.lookup('helo', top_k=3)) <= 3
        assert len(word_list.lookup('helo', top_k=1)) == 1


class TestInvert:
    """V6/V5 got this wrong -- V5 returned `normalize - score`, i.e. negative distances"""

    @pytest.mark.parametrize('normalize', [False, True])
    def test_invert_keeps_the_similarity_ranking(self, normalize):
        word_list = ApproxWordListV7((2, 4)).add_words(WORDS)
        similar = word_list.lookup('helo', top_k=5, invert=True, normalize=normalize)
        distant = word_list.lookup('helo', top_k=5, invert=False, normalize=normalize)
        assert [w for w, _ in similar] == [w for w, _ in distant]

    def test_normalized_distance_is_one_minus_similarity(self):
        word_list = ApproxWordListV7((2, 4)).add_words(WORDS)
        similar = word_list.lookup('helo', top_k=5, invert=True, normalize=True)
        distant = word_list.lookup('helo', top_k=5, invert=False, normalize=True)
        for (_, sim), (_, dist) in zip(similar, distant):
            assert dist == pytest.approx(1.0 - sim, abs=1e-12)
        scores = [s for _, s in distant]
        assert scores == sorted(scores)

    @pytest.mark.parametrize('normalize', [False, True])
    def test_distance_is_never_negative(self, normalize):
        word_list = ApproxWordListV7((2, 4)).add_words(WORDS)
        for _, score in word_list.lookup('helo', top_k=10, invert=False, normalize=normalize):
            assert score >= 0.0


class TestShortWords:
    """queries whose length makes V6 raise ZeroDivisionError must work here"""

    @pytest.mark.parametrize('n_list', [(2,), (4,), (2, 4), (1, 2, 3, 4, 5)])
    @pytest.mark.parametrize('query', ['a', 'ab', 'abc', 'abcd'])
    def test_short_queries_do_not_raise(self, n_list, query):
        word_list = ApproxWordListV7(n_list).add_words(['a', 'ab', 'abc', 'abcd', 'hello'])
        for word, score in word_list.lookup(query, top_k=5, normalize=True):
            assert 0.0 <= score <= 1.0

    def test_query_of_length_n_minus_two(self):
        """the exact case that divides by zero in ApproxWordListV6.lookup"""
        word_list = ApproxWordListV7((2, 4)).add_words(['ab', 'abc', 'hello'])
        assert word_list.lookup('ab', top_k=2)[0][0] == 'ab'

    def test_word_shorter_than_n(self):
        word_list = ApproxWordListV7(5).add_words(['ab', 'abcdefgh'])
        assert word_list.lookup('abcdefgh', top_k=2)[0][0] == 'abcdefgh'

    def test_no_nan_in_scores(self):
        word_list = ApproxWordListV7((4, 5)).add_words(['a', 'ab', 'abc', 'hello'])
        for query in ['a', 'ab', 'abc', 'hello']:
            for _, score in word_list.lookup(query, top_k=5, normalize=True):
                assert not np.isnan(score)
