"""
regression tests for the two bugs fixed in `ApproxWordListV6`, the class exported as `WordList`

these were items 8 and 9 in HANDOFF.md, kept as known-bugs until 2026-09-05. V3 and V5 are
still frozen and still carry them (plus their own), which is why the assertions here name V6
specifically rather than being written against `WordList` in general.

verified 2026-09-05 by re-running this file against the pre-fix tree (with `num_n_grams`
shimmed in so it would still import). exactly four tests went red, and they are the four that
pin the actual bugs:

    TestQueryLengthNMinusTwo::test_lookup_does_not_raise
    TestQueryLengthNMinusTwo::test_lookup_normalized_does_not_raise
    TestQueryLengthNMinusTwo::test_single_n_gram_query_finds_the_exact_word
    TestNormalizationAtN1::test_exact_match_scores_exactly_one

the rest pass before and after on purpose: they guard the freeze (V3 / V5 keep the old
behaviour), the untouched n=(2, 4) default path, or the new clamp. the clamp cannot be caught
by a before/after run at all, since it guards a divide-by-zero that only the *fix* makes
reachable -- it was verified instead by deleting the `max(1, ...)` and watching
TestShortWordDenominatorClamp raise ZeroDivisionError.
"""
import pytest

from nmd import WordList
from nmd.nmd_index import ApproxWordListV6
from nmd.nmd_index import get_n_grams
from nmd.nmd_index import num_grams
from nmd.nmd_index import num_n_grams


class TestNumNGrams:
    """item 9: `num_grams` disagreed with `get_n_grams`, and V6 used it for normalization"""

    @pytest.mark.parametrize('word', ['a', 'ab', 'hello', 'assalamualaikum', 'x' * 40])
    @pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6])
    def test_agrees_with_get_n_grams(self, word, n):
        assert num_n_grams(len(word), n) == len(get_n_grams(word, n))

    def test_never_negative(self):
        assert num_n_grams(1, 6) == 0
        assert num_n_grams(0, 4) == 0

    def test_frozen_num_grams_still_disagrees(self):
        """
        the old function is deliberately left wrong for V3 / V5, so pin that it stayed wrong

        this one passes before and after -- it guards the freeze, not the fix
        """
        assert num_grams(5, 1) == 7 != num_n_grams(5, 1) == 5
        assert num_grams(1, 6) == -2
        assert num_n_grams(1, 6) == 0


class TestQueryLengthNMinusTwo:
    """item 8: a query producing exactly one n-gram divided by zero on the lookup path"""

    def test_lookup_does_not_raise(self):
        word_list = WordList(4, filter_n=0)
        for word in ['hello', 'help', 'he', 'hi']:
            word_list.add_word(word)
        assert word_list.lookup('he', top_k=3)

    def test_lookup_normalized_does_not_raise(self):
        word_list = WordList(4, filter_n=0)
        for word in ['hello', 'help', 'he']:
            word_list.add_word(word)
        results = word_list.lookup('he', top_k=3, normalize=True)
        assert results
        assert all(0.0 <= index_score <= 1.0 for _, (index_score, _) in results)

    def test_single_n_gram_query_finds_the_exact_word(self):
        word_list = WordList(4, filter_n=0)
        for word in ['he', 'hello', 'xyzzy']:
            word_list.add_word(word)
        assert word_list.lookup('he', top_k=1)[0][0] == 'he'

    def test_indexing_side_always_handled_this(self):
        """`add_word` had the guard all along; this is the asymmetry that made it a lookup-only bug"""
        word_list = WordList(4, filter_n=0)
        word_list.add_word('he')  # would have raised at index time if the guard were missing
        assert 'he' in word_list.vocabulary


class TestNormalizationAtN1:
    """item 9, the observable half: the over-count landed in the normalizing denominator"""

    def test_exact_match_scores_exactly_one(self):
        word_list = WordList(1, filter_n=0)
        for word in ['hello', 'world']:
            word_list.add_word(word)
        matched_word, (index_score, _) = word_list.lookup('hello', top_k=1, normalize=True)[0]
        assert matched_word == 'hello'
        # the old denominator added two flag chars that 1-grams never get, giving 10 / 14
        assert index_score == pytest.approx(1.0)

    def test_normalized_scores_stay_in_range(self):
        word_list = WordList((1, 2), filter_n=0)
        for word in ['hello', 'yellow', 'help', 'word']:
            word_list.add_word(word)
        for _, (index_score, _) in word_list.lookup('helo', top_k=4, normalize=True):
            assert 0.0 <= index_score <= 1.0


class TestShortWordDenominatorClamp:
    """
    both words too short for the larger n, while still sharing a smaller n-gram

    the pre-fix tree reached this with a *negative* denominator, which divided a zero numerator
    and so silently produced -0.0. correcting the counts turns that same cell into 0 / 0, so the
    clamp is what keeps the fix from trading one crash for another -- removing the `max(1, ...)`
    in `__lookup_similarity` makes both of these raise ZeroDivisionError
    """

    def test_mixed_n_list_with_short_words(self):
        word_list = WordList((2, 6), filter_n=0)
        for word in ['abc', 'abd']:
            word_list.add_word(word)
        results = word_list.lookup('ab', top_k=2, normalize=True)
        assert len(results) == 2
        assert all(index_score >= 0.0 for _, (index_score, _) in results)

    def test_bound_pass_also_clamped(self):
        """the same 0 / 0 exists in the count-bound pass, which runs before the emd pass"""
        word_list = ApproxWordListV6((2, 7), filter_n=0)
        for word in ['abc', 'abd', 'abe']:
            word_list.add_word(word)
        assert word_list.lookup('ab', top_k=3, normalize=True)


class TestFixesDidNotMoveTheDefaults:
    """
    n=(2, 4) is the documented default, and for it `num_grams` and `num_n_grams` agree on every
    non-empty word -- so none of the above should have changed a single default-path score
    """

    @pytest.mark.parametrize('n', [2, 4])
    @pytest.mark.parametrize('length', range(1, 20))
    def test_counts_agree_on_the_default_n_list(self, n, length):
        assert num_grams(length, n) == num_n_grams(length, n)

    def test_default_lookup_is_unchanged(self):
        word_list = WordList((2, 4))
        for word in ['assalamualaikum', 'waalaikumsalam', 'hello', 'yellow']:
            word_list.add_word(word)
        assert word_list.lookup('asalamalaikum', top_k=1)[0][0] == 'assalamualaikum'
        assert word_list.lookup('walaikumalasam', top_k=1)[0][0] == 'waalaikumsalam'
