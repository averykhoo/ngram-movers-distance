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
