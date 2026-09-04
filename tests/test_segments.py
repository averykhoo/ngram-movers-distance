import random
from itertools import product

import pytest

from nmd.nmd_bow import bow_ngram_movers_distance
from nmd.nmd_core import ngram_movers_distance
from nmd.nmd_segments import _pair_gain
from nmd.nmd_segments import _segments
from nmd.nmd_segments import align_segments
from nmd.nmd_segments import segment_movers_distance


def brute_force_similarity(tokens_a, tokens_b, n=(2, 4), max_len=3, lam=1.0, mu=0.0, mass='ngrams'):
    """enumerate every valid matching of segments, return the max total gain"""
    n_list = (n,) if isinstance(n, int) else tuple(n)
    segs_a = _segments(tokens_a, max_len, n_list, mass)
    segs_b = _segments(tokens_b, max_len, n_list, mass)
    pairs = [(s, t, _pair_gain(s, t, n_list, mass, lam, mu)[1]) for s, t in product(segs_a, segs_b)]

    def cover(seg):
        return set(range(seg.start, seg.start + seg.length))

    best = 0.0

    def rec(idx, used_a, used_b, total):
        nonlocal best
        best = max(best, total)
        for j in range(idx, len(pairs)):
            s, t, gain = pairs[j]
            if cover(s) & used_a or cover(t) & used_b:
                continue
            rec(j + 1, used_a | cover(s), used_b | cover(t), total + gain)

    rec(0, set(), set(), 0.0)
    return best


WORDS = ['pine', 'apple', 'pineapple', 'sports', 'hub', 'sportshub', 'new', 'york', 'newyork', 'city', 'the', 'a']


@pytest.mark.parametrize('mass', ['tokens', 'ngrams'])
@pytest.mark.parametrize('seed', range(20))
def test_ilp_matches_brute_force(seed, mass):
    rng = random.Random(seed)
    tokens_a = [rng.choice(WORDS) for _ in range(rng.randint(1, 4))]
    tokens_b = [rng.choice(WORDS) for _ in range(rng.randint(1, 4))]
    lam = rng.choice([0.0, 0.5, 1.0])
    mu = rng.choice([0.0, 0.25])
    expected = brute_force_similarity(tokens_a, tokens_b, lam=lam, mu=mu, mass=mass)
    actual = align_segments(tokens_a, tokens_b, lam=lam, mu=mu, mass=mass).similarity
    assert actual == pytest.approx(expected, abs=1e-9), (tokens_a, tokens_b, lam, mu)


@pytest.mark.parametrize('mass', ['tokens', 'ngrams'])
def test_split_merge_is_perfect(mass):
    for a, b in [('pineapple', 'pine apple'), ('sports hub clementi', 'sports hubclementi'), ('a b c', 'abc')]:
        ta, tb = a.split(), b.split()
        assert segment_movers_distance(ta, tb, mass=mass, invert=True, normalize=True) == pytest.approx(1.0), (a, b)
        assert segment_movers_distance(ta, tb, mass=mass) == pytest.approx(0.0), (a, b)


@pytest.mark.parametrize('mass', [
    'ngrams',
    # known defect of token mass: short junk tokens add a full unit of mass but few n-grams, so absorbing them
    # into a segment raises (p + q) * sim even though nothing new matched. this is why 'ngrams' is the default.
    pytest.param('tokens', marks=pytest.mark.xfail(strict=True, reason='token mass absorbs junk tokens')),
])
def test_junk_is_not_absorbed(mass):
    # merging unrelated short tokens into a segment must not increase the gain
    alignment = align_segments('theater in a'.split(), ['theater'], mass=mass, lam=0.0)
    assert [(c.segment_a.length, c.segment_b.length) for c in alignment.matches] == [(1, 1)], alignment


@pytest.mark.parametrize('mass', ['tokens', 'ngrams'])
def test_single_tokens_reduce_to_nmd(mass):
    for a, b in [('hello', 'yellow'), ('abc', 'abc'), ('abc', 'xyz')]:
        expected = ngram_movers_distance(a, b, n=2, invert=True, normalize=True)
        assert segment_movers_distance([a], [b], n=2, mass=mass, invert=True, normalize=True) == pytest.approx(expected)


def test_max_len_1_lam_0_reduces_to_bow():
    # same assignment, only the normalization differs: bow divides by max(k, m), this divides by k + m
    for a, b in [('clementi sports hub', 'sport hubs clemmeti'), ('a b c', 'c b a x'), ('hello', 'hello world')]:
        ta, tb = a.split(), b.split()
        expected = bow_ngram_movers_distance(ta, tb, n=2, invert=True) * 2 / (len(ta) + len(tb))
        actual = segment_movers_distance(ta, tb, n=2, max_len=1, lam=0.0, mass='tokens', invert=True, normalize=True)
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize('mass', ['tokens', 'ngrams'])
def test_symmetry(mass):
    for a, b in [('pineapple', 'pine apple'), ('new york city', 'newyork'), ('a b c d', 'd c b a')]:
        ta, tb = a.split(), b.split()
        assert segment_movers_distance(ta, tb, mass=mass) == pytest.approx(segment_movers_distance(tb, ta, mass=mass))


def test_empty():
    assert segment_movers_distance([], []) == 0.0
    assert segment_movers_distance(['a'], [], mass='tokens') == 1.0
    assert segment_movers_distance(['a'], [], invert=True, normalize=True) == 0.0
