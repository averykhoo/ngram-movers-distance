import itertools

import pytest

from nmd.nmd_core import ngram_movers_distance


def test_n_must_be_at_least_1():
    with pytest.raises(ValueError):
        ngram_movers_distance('abc', 'abc', n=0)
    with pytest.raises(ValueError):
        ngram_movers_distance('abc', 'abc', n=-1)


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_identical_words_are_identical(n):
    for word in ['a', 'ab', 'hello', 'assalamualaikum']:
        assert ngram_movers_distance(word, word, n=n) == pytest.approx(0.0)
        assert ngram_movers_distance(word, word, n=n, normalize=True) == pytest.approx(0.0)
        assert ngram_movers_distance(word, word, n=n, invert=True, normalize=True) == pytest.approx(1.0)


def test_unigrams_drop_the_markers():
    # with markers, '\2a\3' vs '\2b\3' would share two of three unigrams and look 67% similar.
    # without them, 'a' and 'b' share nothing at all
    assert ngram_movers_distance('a', 'b', n=1, invert=True) == pytest.approx(0.0)
    assert ngram_movers_distance('a', 'b', n=1, normalize=True) == pytest.approx(1.0)


def test_unigrams_ignore_adjacency_but_not_position():
    # an anagram has an identical character multiset, so n=1 sees only the positional displacement,
    # while n=2 also sees that no bigram survives
    assert ngram_movers_distance('abc', 'cba', n=1, invert=True, normalize=True) > 0.5
    assert ngram_movers_distance('abc', 'cba', n=2, invert=True, normalize=True) < 0.3
    # a pure reordering of identical characters cannot be free, or n=1 would be a bag-of-characters
    assert ngram_movers_distance('abcd', 'dcba', n=1, normalize=True) > 0.0


def test_no_ngrams_at_all_does_not_raise():
    # both words empty at n == 1, and both words too short at larger n: previously ZeroDivisionError
    assert ngram_movers_distance('', '', n=1, normalize=True) == pytest.approx(0.0)
    assert ngram_movers_distance('', '', n=1, invert=True, normalize=True) == pytest.approx(1.0)
    assert ngram_movers_distance('a', 'a', n=4, normalize=True) == pytest.approx(0.0)


def test_no_ngrams_at_all_still_distinguishes_different_words():
    """
    with no n-grams on either side there is no evidence from n-grams, but the words are still
    different and must not come back as identical. this reproduces the fallback the callers used to
    implement themselves in the ZeroDivisionError handler, `int(word_1 != word_2)`.
    """
    assert ngram_movers_distance('a', 'b', n=4, normalize=True) == pytest.approx(1.0)
    assert ngram_movers_distance('a', 'b', n=4, invert=True, normalize=True) == pytest.approx(0.0)
    assert ngram_movers_distance('', 'x', n=2, normalize=True) > 0.0
    # and equal words at the same length still agree
    assert ngram_movers_distance('ab', 'ab', n=4, normalize=True) == pytest.approx(0.0)


@pytest.mark.parametrize('n', [1, 2, 3])
def test_symmetry(n):
    for a, b in [('hello', 'yellow'), ('pineapple', 'apple'), ('abc', ''), ('kitten', 'sitting')]:
        assert ngram_movers_distance(a, b, n=n) == pytest.approx(ngram_movers_distance(b, a, n=n))


@pytest.mark.parametrize('n', [1, 2, 3])
def test_normalized_output_is_in_range(n):
    for a, b in [('hello', 'yellow'), ('a', 'zzzzzzzz'), ('abc', 'abc'), ('', 'x')]:
        distance = ngram_movers_distance(a, b, n=n, normalize=True)
        similarity = ngram_movers_distance(a, b, n=n, invert=True, normalize=True)
        assert 0.0 <= distance <= 1.0, (a, b, n)
        assert 0.0 <= similarity <= 1.0, (a, b, n)
        assert distance + similarity == pytest.approx(1.0)


@pytest.mark.parametrize('n', range(1, 10))
@pytest.mark.parametrize('word', ['', 'a', 'ab', 'abc', 'hello'])
def test_a_word_is_identical_to_itself_at_every_n(word, n):
    """
    a word too short to contain one n-gram has no n-grams, not a negative number of them.

    the counts used to be `len(word) - n + 1` with no clamp, so a word of fewer than n - 3
    characters produced a negative count on both sides. that skipped the
    `num_grams_1 + num_grams_2 == 0` fallback, and the normalized distance of a word against
    *itself* came back as 1.0 -- maximally far apart. the existing parametrization stopped at
    n=4, which is exactly one short of where the shortest word breaks. fixed 2026-09-18.
    """
    assert ngram_movers_distance(word, word, n=n) == pytest.approx(0.0)
    assert ngram_movers_distance(word, word, n=n, normalize=True) == pytest.approx(0.0)
    assert ngram_movers_distance(word, word, n=n, invert=True, normalize=True) == pytest.approx(1.0)


@pytest.mark.parametrize('n', range(1, 10))
def test_distance_stays_non_negative_and_normalized_stays_in_range(n):
    """the metric's headline guarantees, swept over the short words that used to break them"""
    words = ['', 'a', 'ab', 'abc', 'abcd', 'hello', 'yellow']
    for word_1, word_2 in itertools.product(words, repeat=2):
        distance = ngram_movers_distance(word_1, word_2, n=n)
        similarity = ngram_movers_distance(word_1, word_2, n=n, invert=True)
        assert distance >= 0.0, (word_1, word_2, n)
        assert similarity >= 0.0, (word_1, word_2, n)

        normalized_distance = ngram_movers_distance(word_1, word_2, n=n, normalize=True)
        normalized_similarity = ngram_movers_distance(word_1, word_2, n=n, invert=True, normalize=True)
        assert 0.0 <= normalized_distance <= 1.0, (word_1, word_2, n)
        assert 0.0 <= normalized_similarity <= 1.0, (word_1, word_2, n)
        assert normalized_distance + normalized_similarity == pytest.approx(1.0), (word_1, word_2, n)
