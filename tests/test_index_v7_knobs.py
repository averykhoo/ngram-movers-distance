"""
tests for `ApproxWordListV7`'s `position_weight` and `denominator` lookup knobs

both were added after the idf work and both are inert at their defaults, which is the property
most worth pinning: `position_weight=1.0, denominator='dice'` has to reproduce the index as it
scored before they existed, bit-for-bit, not approximately.

sabotage-checked 2026-09-05 -- each of the following makes at least one test here fail:
  * dropping the `position_weight *` factor from any of the four scoring branches
  * using `c1 + c2 - emd` instead of `2 * min(c1, c2) - position_weight * displacement` in the
    both-sides-repeat branch (passes at the default, fails every position_weight < 1 test)
  * forgetting `abs(n_locations - other_count)` in that branch's displacement
  * accepting position_weight > 1 instead of raising
  * swapping the geo denominator for `sqrt(total_query * total_word)` without the factor 2
"""
import math
from collections import Counter

import pytest

from nmd.nmd_index_v7 import ApproxWordListV7
from nmd.nmd_index_v7 import n_grams

# deliberately repeat-heavy: 'abababab' and 'mississippi' put the same n-gram at several
# positions in one word, which is the only way to reach the dynamic-programming branch
REPEAT_WORDS = ['abababab', 'ababab', 'abab', 'banana', 'bananana', 'mississippi', 'missisippi',
                'aaaaaa', 'aaaa', 'papaya', 'pappaya', 'cocoa', 'cacao', 'rococo', 'bookkeeper',
                'hello', 'yellow', 'mellow', 'bellow', 'tomato', 'potato', 'barbarian']


def dice_multiset(a: str, b: str, n: int) -> float:
    """2 * |A n B| / (|A| + |B|) over n-gram multisets -- nmd with position discarded"""
    count_a, count_b = Counter(n_grams(a, n)), Counter(n_grams(b, n))
    overlap = sum((count_a & count_b).values())
    return 2 * overlap / (sum(count_a.values()) + sum(count_b.values()))


@pytest.fixture(scope='module')
def index():
    return ApproxWordListV7(n=(2,)).add_words(REPEAT_WORDS)


class TestPositionWeightZeroIsDiceMultiset:
    """position_weight=0 must discard position entirely, leaving Dice over n-gram multisets"""

    @pytest.mark.parametrize('query', ['abab', 'banana', 'mississippi', 'cocoa', 'aaaa'])
    def test_matches_dice_multiset_exactly(self, index, query):
        results = index.lookup(query, top_k=len(REPEAT_WORDS), normalize=True, position_weight=0.0)
        assert results
        for word, score in results:
            assert score == pytest.approx(dice_multiset(query, word, 2), abs=1e-12)

    def test_forgives_reordering(self):
        """
        dropping the position term stops charging for moved text

        note it does not make the score *equal* to an exact match: reordering the words changes
        which character n-grams span the boundaries (`\\2a` becomes `\\2g`, and so on), so the
        multisets genuinely differ and Dice is right to score below 1.0. what the knob removes
        is the extra penalty for the shared n-grams having moved
        """
        word_list = ApproxWordListV7(n=(2,)).add_words(['alpha beta gamma', 'gamma beta alpha'])
        reordered_off = dict(word_list.lookup('alpha beta gamma', top_k=2, normalize=True,
                                              position_weight=0.0))['gamma beta alpha']
        reordered_on = dict(word_list.lookup('alpha beta gamma', top_k=2, normalize=True,
                                             position_weight=1.0))['gamma beta alpha']
        # position_weight=1 charges for the displacement; position_weight=0 does not.
        # measured 2026-09-05: 0.8824 off vs 0.7206 on, a gap of 0.162
        assert reordered_off > reordered_on + 0.1
        # and the exact match is unaffected either way, since nothing moved
        for position_weight in (0.0, 1.0):
            exact = dict(word_list.lookup('alpha beta gamma', top_k=2, normalize=True,
                                          position_weight=position_weight))['alpha beta gamma']
            assert exact == pytest.approx(1.0, abs=1e-12)


class TestPositionWeightMonotone:
    """displacement is non-negative, so charging more of it can only lower a score"""

    @pytest.mark.parametrize('query', ['abab', 'banana', 'mississippi', 'papaya'])
    @pytest.mark.parametrize('normalize', [False, True])
    def test_score_is_non_increasing_in_position_weight(self, index, query, normalize):
        previous = None
        for position_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            scores = dict(index.lookup(query, top_k=len(REPEAT_WORDS), normalize=normalize,
                                       position_weight=position_weight))
            total = sum(scores.values())
            if previous is not None:
                assert total <= previous + 1e-9
            previous = total

    def test_strictly_lower_somewhere(self, index):
        """non-increasing would also hold for a knob that did nothing at all"""
        low = dict(index.lookup('abab', top_k=len(REPEAT_WORDS), normalize=True,
                                position_weight=0.0))
        high = dict(index.lookup('abab', top_k=len(REPEAT_WORDS), normalize=True,
                                 position_weight=1.0))
        assert any(high[word] < low[word] - 1e-9 for word in set(low) & set(high))


class TestDefaultsAreInert:
    """the knobs at their defaults must reproduce the index as it scored before they existed"""

    @pytest.mark.parametrize('n', [(1,), (2,), (3,), (2, 4), (1, 2, 3)])
    @pytest.mark.parametrize('idf_exponent', [0.0, 1.0, 2.5])
    def test_default_matches_recomputed_nmd(self, n, idf_exponent):
        """
        `2 * min(c1, c2) - displacement` is an algebraic rewrite of `c1 + c2 - emd`, so at
        position_weight=1.0 every score must still be exactly the value the docstring promises
        """
        from nmd.emd_1d import emd_1d_dp

        word_list = ApproxWordListV7(n=n, idf_exponent=idf_exponent).add_words(REPEAT_WORDS)

        def locate(text, size):
            grams = n_grams(text, size)
            denominator = len(grams) - 1
            out = {}
            for idx, gram in enumerate(grams):
                out.setdefault(gram, []).append(idx / denominator if denominator > 0 else 0.0)
            return out

        for query in ('abab', 'banana', 'mississippi', 'cocoa'):
            for word, score in word_list.lookup(query, top_k=8, normalize=True):
                parts = []
                for size in n:
                    query_locs, word_locs = locate(query, size), locate(word, size)
                    weight = lambda gram: word_list._gram_weight(n.index(size), gram) \
                        if idf_exponent else 1.0
                    numerator = sum(weight(gram) * (len(locs) + len(word_locs[gram])
                                                    - emd_1d_dp(locs, word_locs[gram]))
                                    for gram, locs in query_locs.items() if gram in word_locs)
                    total_q = sum(weight(gram) * len(locs) for gram, locs in query_locs.items())
                    total_w = sum(weight(gram) * len(locs) for gram, locs in word_locs.items())
                    parts.append(numerator / max(1.0, total_q + total_w))
                assert score == pytest.approx(sum(parts) / len(parts), abs=1e-9)


class TestGeoDenominator:
    """geo is the cosine-shaped denominator; AM-GM fixes its relationship to dice"""

    @pytest.mark.parametrize('query', ['abab', 'banana', 'mississippi', 'hello'])
    def test_geo_never_scores_below_dice(self, index, query):
        """2*sqrt(a*b) <= a+b for non-negative a, b, so the geo score is never smaller"""
        dice = dict(index.lookup(query, top_k=len(REPEAT_WORDS), normalize=True,
                                 denominator='dice'))
        geo = dict(index.lookup(query, top_k=len(REPEAT_WORDS), normalize=True,
                                denominator='geo'))
        assert set(dice) & set(geo)
        for word in set(dice) & set(geo):
            assert geo[word] >= dice[word] - 1e-12

    def test_equal_lengths_make_geo_and_dice_agree(self):
        """AM == GM exactly when the two totals are equal, i.e. equal-length words"""
        words = ['abcdef', 'abcdeg', 'bcdefg', 'zbcdef']
        word_list = ApproxWordListV7(n=(2,)).add_words(words)
        dice = dict(word_list.lookup('abcdef', top_k=4, normalize=True, denominator='dice'))
        geo = dict(word_list.lookup('abcdef', top_k=4, normalize=True, denominator='geo'))
        for word in dice:
            assert dice[word] == pytest.approx(geo[word], abs=1e-12)

    def test_geo_strictly_higher_when_lengths_differ(self):
        """otherwise the knob would be doing nothing"""
        word_list = ApproxWordListV7(n=(2,)).add_words(['ab', 'abcdefghijklmnop'])
        dice = dict(word_list.lookup('ab', top_k=2, normalize=True, denominator='dice'))
        geo = dict(word_list.lookup('ab', top_k=2, normalize=True, denominator='geo'))
        assert geo['abcdefghijklmnop'] > dice['abcdefghijklmnop'] + 1e-6

    def test_geo_is_inert_without_normalize(self, index):
        """the denominator only exists on the normalizing path"""
        without = index.lookup('banana', top_k=5, normalize=False, denominator='dice')
        with_geo = index.lookup('banana', top_k=5, normalize=False, denominator='geo')
        assert without == with_geo

    def test_matches_closed_form(self):
        """pin the actual formula, not just its ordering relative to dice"""
        word_list = ApproxWordListV7(n=(2,)).add_words(['abcde'])
        [(word, score)] = word_list.lookup('abcde', top_k=1, normalize=True, denominator='geo')
        # identical strings: numerator is 2 * num_grams, denominator 2 * sqrt(g * g) == 2g
        grams = len(n_grams('abcde', 2))
        assert word == 'abcde'
        assert score == pytest.approx(2 * grams / (2 * math.sqrt(grams * grams)), abs=1e-12)


class TestValidation:
    """out-of-range values are refused rather than clamped -- see the lookup docstring"""

    @pytest.mark.parametrize('bad', [1.5, -0.1, 2.0, -1.0])
    def test_position_weight_outside_unit_interval_raises(self, index, bad):
        with pytest.raises(ValueError):
            index.lookup('banana', position_weight=bad)

    @pytest.mark.parametrize('bad', ['1.0', None, True, [1.0]])
    def test_position_weight_wrong_type_raises(self, index, bad):
        with pytest.raises(TypeError):
            index.lookup('banana', position_weight=bad)

    @pytest.mark.parametrize('bad', ['cosine', 'DICE', '', None, 1])
    def test_unknown_denominator_raises(self, index, bad):
        with pytest.raises(ValueError):
            index.lookup('banana', denominator=bad)


class TestPruningStillExact:
    """
    the two-sided count bound is what lets lookup prune; both knobs have to preserve it

    position_weight <= 1 keeps the contribution inside [min(c1,c2), 2*min(c1,c2)], and any
    positive per-word denominator scales the bound and the score by the same factor
    """

    @pytest.mark.parametrize('position_weight', [0.0, 0.5, 1.0])
    @pytest.mark.parametrize('denominator', ['dice', 'geo'])
    def test_lookup_matches_exhaustive_scan(self, position_weight, denominator):
        words = sorted(set(REPEAT_WORDS))
        word_list = ApproxWordListV7(n=(2, 4)).add_words(words)
        for query in ('abab', 'banana', 'mississippi', 'hello', 'cocoa'):
            top_k = 3
            pruned = word_list.lookup(query, top_k=top_k, normalize=True,
                                      position_weight=position_weight, denominator=denominator)
            # an exhaustive scan is just the same lookup with no room to prune
            exhaustive = word_list.lookup(query, top_k=len(words), normalize=True,
                                          position_weight=position_weight,
                                          denominator=denominator)[:top_k]
            assert [w for w, _ in pruned] == [w for w, _ in exhaustive]
            for (_, a), (_, b) in zip(pruned, exhaustive):
                assert a == pytest.approx(b, abs=1e-12)
