"""
tests for `WordSet` behaviour outside the idf path (which `test_word_set_idf.py` covers)

written 2026-09-05 alongside two fixes:

* `find_similar(min_similarity=...)` was validated and then silently ignored -- the only code
  applying it was inside the commented-out exact-rescoring block, so a caller asking for
  `>= 0.99` got whatever the top k happened to be. `TestMinSimilarity` fails without the fix.
* the default `ngram_sizes` was `(2, 3, 4)` while the docstring said `(2, 4)`.

this file is not a correctness proof of `find_similar` -- that is still open (HANDOFF #10).
"""
import pytest

from nmd.nmd_word_set import WordSet
from nmd.nmd_word_set import normalize_nfd_strip_marks
from nmd.nmd_word_set import normalize_nfkc

WORDS = ['hello', 'hallo', 'hullo', 'yellow', 'help', 'world', 'banana', 'pineapple']


def build(words=WORDS, **kwargs):
    word_set = WordSet(**kwargs)
    for word in words:
        word_set.add(word)
    return word_set


class TestDefaults:
    def test_default_ngram_sizes_matches_the_docstring(self):
        assert WordSet().ngram_sizes == (2, 4)

    def test_default_is_case_insensitive(self):
        word_set = build(['Hello'])
        assert 'hello' in word_set
        assert 'HELLO' in word_set

    def test_original_casing_is_what_comes_back(self):
        word_set = build(['Hello'])
        assert word_set.find_similar('hello', k=1)[0][0] == 'Hello'


class TestMinSimilarity:
    """the parameter used to be accepted and ignored entirely"""

    def test_threshold_actually_filters(self):
        word_set = build()
        unfiltered = word_set.find_similar('helo', k=5)
        filtered = word_set.find_similar('helo', k=5, min_similarity=0.4)
        assert len(filtered) < len(unfiltered), (filtered, unfiltered)

    @pytest.mark.parametrize('threshold', [0.0, 0.1, 0.25, 0.4, 0.6, 0.9, 1.0])
    def test_every_result_meets_the_threshold(self, threshold):
        word_set = build()
        for query in ['helo', 'hell', 'wrld', 'pineaple', 'xyzzy']:
            for _, score in word_set.find_similar(query, k=10, min_similarity=threshold):
                assert score >= threshold

    def test_impossible_threshold_returns_nothing(self):
        word_set = build()
        assert word_set.find_similar('helo', k=5, min_similarity=0.99) == []

    def test_zero_threshold_matches_no_threshold(self):
        word_set = build()
        assert word_set.find_similar('helo', k=5, min_similarity=0.0) == word_set.find_similar('helo', k=5)

    def test_none_threshold_matches_no_threshold(self):
        word_set = build()
        assert word_set.find_similar('helo', k=5, min_similarity=None) == word_set.find_similar('helo', k=5)

    def test_k_still_caps_the_result_count(self):
        word_set = build()
        assert len(word_set.find_similar('helo', k=1, min_similarity=0.0)) == 1

    def test_exact_match_survives_a_high_threshold(self):
        word_set = build()
        assert word_set.find_similar('hello', k=1, min_similarity=0.99)[0][0] == 'hello'

    @pytest.mark.parametrize('bad', [-0.1, 1.1, 2, 'a lot'])
    def test_out_of_range_threshold_is_rejected(self, bad):
        with pytest.raises(ValueError):
            build().find_similar('helo', k=5, min_similarity=bad)

    def test_suggest_is_unaffected(self):
        """suggest() passes min_similarity=None, so it must still return up to k names"""
        word_set = build()
        assert word_set.suggest('helo', k=3) == [w for w, _ in word_set.find_similar('helo', k=3)]


class TestResultOrdering:
    def test_results_are_sorted_best_first(self):
        word_set = build()
        scores = [score for _, score in word_set.find_similar('helo', k=10)]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_normalized(self):
        word_set = build()
        for query in ['helo', 'hello', 'xyzzy', 'pineaple']:
            for _, score in word_set.find_similar(query, k=10):
                assert 0.0 <= score <= 1.0

    def test_exact_match_ranks_first(self):
        word_set = build()
        for word in WORDS:
            assert word_set.find_similar(word, k=1)[0][0] == word


class TestSetProtocol:
    def test_add_discard_contains_len(self):
        word_set = WordSet()
        word_set.add('hello')
        assert 'hello' in word_set and len(word_set) == 1
        word_set.discard('hello')
        assert 'hello' not in word_set and len(word_set) == 0

    def test_discard_is_silent_but_remove_raises(self):
        word_set = build(['hello'])
        word_set.discard('nope')  # no error
        with pytest.raises(KeyError):
            word_set.remove('nope')

    def test_discarded_words_stop_being_returned(self):
        word_set = build()
        assert word_set.find_similar('helo', k=1)[0][0] == 'hello'
        word_set.remove('hello')
        assert 'hello' not in [w for w, _ in word_set.find_similar('helo', k=10)]

    def test_clear(self):
        word_set = build()
        word_set.clear()
        assert len(word_set) == 0
        assert word_set.find_similar('helo', k=5) == []

    def test_iteration_yields_every_word(self):
        assert sorted(build()) == sorted(WORDS)

    def test_adding_twice_is_a_no_op(self):
        word_set = build(['hello', 'hello'])
        assert len(word_set) == 1


class TestEdgeCases:
    def test_empty_set_returns_nothing(self):
        assert WordSet().find_similar('hello', k=5) == []

    def test_empty_query_returns_nothing(self):
        assert build().find_similar('', k=5) == []

    @pytest.mark.parametrize('bad_k', [0, -1])
    def test_non_positive_k_is_rejected(self, bad_k):
        with pytest.raises(ValueError):
            build().find_similar('hello', k=bad_k)

    def test_non_string_query_is_rejected(self):
        with pytest.raises(TypeError):
            build().find_similar(None, k=5)

    def test_short_query_still_works_on_the_default(self):
        """
        the START / END padding means even a 1-char query produces filter n-grams at n=2, so the
        default `(2, 4)` has no short-query dead zone
        """
        assert build().find_similar('h', k=5)

    def test_query_too_short_for_the_filter_returns_nothing(self):
        """
        with a single large n the filter is that same n, and a query too short to produce one
        n-gram of that size is dropped before scoring -- it returns [] rather than falling back
        to a full scan. deliberate (the alternative is scanning the whole vocabulary), but it is
        a silent empty result, so it is pinned here
        """
        word_set = build(ngram_sizes=4)
        assert word_set.find_similar('he', k=5) == []
        assert word_set.find_similar('hell', k=5)  # 4 chars is long enough

    def test_filter_is_disabled_above_n_4(self):
        """
        `_effective_filter_n` is None once the smallest n exceeds 4, so every word is scored and
        even completely dissimilar ones come back with a score of 0.0
        """
        word_set = build(ngram_sizes=5)
        assert word_set._effective_filter_n is None
        results = word_set.find_similar('h', k=10)
        assert len(results) == len(WORDS)
        assert all(score == 0.0 for _, score in results)


class TestUnicodeNormalization:
    def test_nfc_default_folds_composed_and_decomposed(self):
        word_set = build(['café'])
        assert 'café' in word_set  # decomposed form of the same word

    def test_strip_marks_ignores_accents(self):
        word_set = build(['café'], unicode_normalizer=normalize_nfd_strip_marks)
        assert 'cafe' in word_set

    def test_nfkc_is_accepted(self):
        assert build(['hello'], unicode_normalizer=normalize_nfkc).ngram_sizes == (2, 4)

    def test_normalizer_can_be_disabled(self):
        assert build(['hello'], unicode_normalizer=None).find_similar('hello', k=1)[0][0] == 'hello'
