"""
Tests for bow_ngram_movers_distance and the emd_1d_fast shortcut it relies on.

bow_ngram_movers_distance is checked against a reference that builds the cost matrix by
calling ngram_movers_distance per cell -- which is what the function used to do internally,
before the n-gram splitting was hoisted out of the loop.
"""
import itertools
import random

import pytest

from nmd.emd_1d import emd_1d_dp
from nmd.emd_1d import emd_1d_fast
from nmd.nmd_core import ngram_movers_distance

scipy = pytest.importorskip('scipy.optimize')

from nmd.nmd_bow import bow_ngram_movers_distance  # noqa: E402

WORDS = ['hello', 'yellow', 'sports', 'sport', 'hub', 'hubs', 'clementi', 'clemmeti',
         'banana', 'bananana', 'abababab', 'a', 'ab', 'abc']


def reference_bow(bag_of_words_1, bag_of_words_2, n=2, invert=False, normalize=False):
    """the original implementation: one ngram_movers_distance call per matrix cell"""
    bag_1 = list(bag_of_words_1)
    bag_2 = list(bag_of_words_2)
    costs = []
    for word_1 in bag_1:
        row = []
        for word_2 in bag_2:
            try:
                row.append(ngram_movers_distance(word_1, word_2, n=n, normalize=True))
            except ZeroDivisionError:
                row.append(int(word_1 != word_2))
        costs.append(row)
    if costs:
        row_idxs, col_idxs = scipy.linear_sum_assignment(costs)
    else:
        row_idxs, col_idxs = [], []
    if invert:
        out = min(len(bag_1), len(bag_2))
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            out -= costs[row_idx][col_idx]
    else:
        out = abs(len(bag_1) - len(bag_2))
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            out += costs[row_idx][col_idx]
    if normalize:
        out /= max(len(bag_1), len(bag_2))
    return out


class TestEmd1dFast:
    """the shortcut must be a drop-in for the dp, not an approximation of it"""

    @pytest.mark.parametrize('len_x', range(0, 5))
    @pytest.mark.parametrize('len_y', range(0, 5))
    def test_matches_dp_on_random_input(self, len_x, len_y):
        random.seed(len_x * 10 + len_y)
        for _ in range(300):
            x = [round(random.random(), 3) for _ in range(len_x)]
            y = [round(random.random(), 3) for _ in range(len_y)]
            assert emd_1d_fast(x, y) == pytest.approx(emd_1d_dp(x, y), abs=1e-12), (x, y)

    def test_matches_dp_on_a_grid(self):
        """exhaustive over a small quantized grid, where ties and duplicates are common"""
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        for len_x in range(0, 3):
            for len_y in range(0, 3):
                for x in itertools.product(grid, repeat=len_x):
                    for y in itertools.product(grid, repeat=len_y):
                        assert emd_1d_fast(list(x), list(y)) == pytest.approx(
                            emd_1d_dp(list(x), list(y)), abs=1e-12), (x, y)

    def test_matches_dp_outside_the_unit_interval(self):
        """the min(dist, 2) clamp matters only when points are further apart than 2"""
        for x, y in [([0.0], [5.0]), ([5.0], [0.0]), ([0.0], [1.9]), ([0.0], [2.1]),
                     ([-3.0], [3.0]), ([0.0], [3.0, 7.0]), ([10.0, 20.0], [0.0])]:
            assert emd_1d_fast(x, y) == pytest.approx(emd_1d_dp(x, y), abs=1e-12), (x, y)

    def test_empty_inputs(self):
        assert emd_1d_fast([], []) == emd_1d_dp([], [])
        assert emd_1d_fast([0.5], []) == emd_1d_dp([0.5], [])
        assert emd_1d_fast([], [0.5]) == emd_1d_dp([], [0.5])

    def test_does_not_mutate_its_inputs(self):
        x, y = [0.9, 0.1, 0.5], [0.4, 0.2]
        emd_1d_fast(x, y)
        assert x == [0.9, 0.1, 0.5] and y == [0.4, 0.2]


class TestBowMatchesReference:
    @pytest.mark.parametrize('invert', [False, True])
    @pytest.mark.parametrize('normalize', [False, True])
    @pytest.mark.parametrize('n', [2, 3, 4])
    def test_random_bags(self, invert, normalize, n):
        random.seed(hash((invert, normalize, n)) % 10000)
        for _ in range(40):
            bag_1 = [random.choice(WORDS) for _ in range(random.randint(1, 6))]
            bag_2 = [random.choice(WORDS) for _ in range(random.randint(1, 6))]
            assert (bow_ngram_movers_distance(bag_1, bag_2, n=n, invert=invert, normalize=normalize)
                    == pytest.approx(reference_bow(bag_1, bag_2, n=n, invert=invert,
                                                   normalize=normalize), abs=1e-12))

    @pytest.mark.parametrize('invert', [False, True])
    @pytest.mark.parametrize('normalize', [False, True])
    def test_repeated_words(self, invert, normalize):
        """duplicate words share one n-gram entry, so duplicate rows must still be right"""
        for bag_1, bag_2 in [(['hub', 'hub', 'hub'], ['hubs', 'hub']),
                             (['a', 'a'], ['a', 'a', 'a']),
                             (['banana'] * 4, ['bananana'] * 2)]:
            assert (bow_ngram_movers_distance(bag_1, bag_2, invert=invert, normalize=normalize)
                    == pytest.approx(reference_bow(bag_1, bag_2, invert=invert,
                                                   normalize=normalize), abs=1e-12))

    @pytest.mark.parametrize('invert', [False, True])
    @pytest.mark.parametrize('normalize', [False, True])
    def test_uneven_bag_sizes(self, invert, normalize):
        for bag_1, bag_2 in [(['sports'], ['sport', 'hub', 'clementi']),
                             (['a', 'b', 'c', 'd', 'e'], ['a'])]:
            assert (bow_ngram_movers_distance(bag_1, bag_2, invert=invert, normalize=normalize)
                    == pytest.approx(reference_bow(bag_1, bag_2, invert=invert,
                                                   normalize=normalize), abs=1e-12))

    @pytest.mark.parametrize('invert', [False, True])
    def test_one_empty_bag(self, invert):
        for normalize in (False, True):
            assert (bow_ngram_movers_distance(['hello'], [], invert=invert, normalize=normalize)
                    == pytest.approx(reference_bow(['hello'], [], invert=invert,
                                                   normalize=normalize), abs=1e-12))

    def test_words_shorter_than_n(self):
        """num_grams goes non-positive here, which is the total_grams == 0 branch"""
        for n in (4, 5, 6):
            for bag_1, bag_2 in [(['a'], ['a']), (['a'], ['b']), (['a', 'ab'], ['b', 'abc'])]:
                assert (bow_ngram_movers_distance(bag_1, bag_2, n=n, invert=True)
                        == pytest.approx(reference_bow(bag_1, bag_2, n=n, invert=True), abs=1e-12))


class TestBowSemantics:
    def test_readme_example(self):
        text_1 = 'Clementi Sports Hub'
        text_2 = 'sport hubs clemmeti'
        score = bow_ngram_movers_distance(text_1.casefold().split(), text_2.casefold().split(),
                                          invert=True, normalize=True)
        assert 0.0 <= score <= 1.0
        assert score > 0.5, score  # these are meant to be recognisably similar

    def test_identical_bags_score_one(self):
        bag = ['clementi', 'sports', 'hub']
        assert bow_ngram_movers_distance(bag, bag, invert=True, normalize=True) == pytest.approx(1.0)
        assert bow_ngram_movers_distance(bag, bag, invert=False, normalize=True) == pytest.approx(0.0)

    def test_word_order_does_not_matter(self):
        bag = ['clementi', 'sports', 'hub']
        shuffled = ['hub', 'clementi', 'sports']
        assert (bow_ngram_movers_distance(bag, shuffled, invert=True, normalize=True)
                == pytest.approx(1.0))

    def test_symmetric(self):
        bag_1 = ['clementi', 'sports', 'hub']
        bag_2 = ['sport', 'hubs', 'clemmeti']
        for invert in (False, True):
            assert (bow_ngram_movers_distance(bag_1, bag_2, invert=invert, normalize=True)
                    == pytest.approx(bow_ngram_movers_distance(bag_2, bag_1, invert=invert,
                                                               normalize=True), abs=1e-12))

    def test_normalized_score_in_unit_range(self):
        random.seed(3)
        for _ in range(50):
            bag_1 = [random.choice(WORDS) for _ in range(random.randint(1, 5))]
            bag_2 = [random.choice(WORDS) for _ in range(random.randint(1, 5))]
            for invert in (False, True):
                score = bow_ngram_movers_distance(bag_1, bag_2, invert=invert, normalize=True)
                assert 0.0 <= score <= 1.0, (bag_1, bag_2, invert, score)

    def test_similar_beats_dissimilar(self):
        query = ['clementi', 'sports', 'hub']
        close = ['sport', 'hubs', 'clemmeti']
        far = ['banana', 'yellow', 'abababab']
        assert (bow_ngram_movers_distance(query, close, invert=True, normalize=True)
                > bow_ngram_movers_distance(query, far, invert=True, normalize=True))

    def test_split_words_score_low_as_documented(self):
        """the README warns this does not merge or split words"""
        assert bow_ngram_movers_distance(['pineapple'], ['pine', 'apple'],
                                         invert=True, normalize=True) < 0.6
