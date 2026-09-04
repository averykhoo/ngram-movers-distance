"""
idf weighting in ApproxWordListV7

the property that matters most is that weighting does not break the pruning. V7's whole claim
is that `lookup` returns the scores an exhaustive scan would, because the count bound brackets
the true score two-sidedly. weighting each n-gram scales both ends of that bracket by the same
non-negative constant, so the bound should survive -- these tests check that it actually does,
against a brute-force scorer written independently of the index internals.
"""
import math
import random
from collections import Counter

import pytest

np = pytest.importorskip('numpy')

from nmd.emd_1d import emd_1d_dp  # noqa: E402
from nmd.nmd_index_v7 import ApproxWordListV7  # noqa: E402
from nmd.nmd_index_v7 import n_grams  # noqa: E402
from nmd.nmd_index_v7 import num_n_grams  # noqa: E402

WORDS = ['hello', 'yellow', 'mellow', 'bellow', 'help', 'hell', 'hello world', 'banana',
         'bananana', 'abababab', 'sports', 'sport', 'hub', 'hubs', 'clementi', 'clemmeti',
         'a', 'ab', 'abc', 'pineapple', 'pine', 'apple', 'camera', 'cameras']


def _locations(grams):
    """same convention the index uses: positions normalized by len(grams) - 1"""
    locations = {}
    if len(grams) > 1:
        denominator = len(grams) - 1
        for idx, gram in enumerate(grams):
            locations.setdefault(gram, []).append(idx / denominator)
    elif grams:
        locations[grams[0]] = [0.0]
    return locations


def brute_force_scores(words, query, n_list, exponent, dim=1):
    """normalized weighted nmd similarity of the query against every word, no index involved"""
    vocab_size = len(words)
    document_frequency = []
    for n in n_list:
        counter = Counter()
        for word in words:
            counter.update(set(n_grams(word, n)))
        document_frequency.append(counter)

    def weight(n_idx, gram):
        if exponent == 0:
            return 1.0
        df = document_frequency[n_idx].get(gram, 0)
        return math.log((vocab_size + 1) / df) ** exponent if df else math.log(vocab_size + 1) ** exponent

    scores = {}
    for word in words:
        parts = []
        for n_idx, n in enumerate(n_list):
            query_grams, word_grams = n_grams(query, n), n_grams(word, n)
            query_locations, word_locations = _locations(query_grams), _locations(word_grams)
            similarity = 0.0
            for gram, positions in query_locations.items():
                other = word_locations.get(gram)
                if other is not None:
                    similarity += weight(n_idx, gram) * (len(positions) + len(other)
                                                         - emd_1d_dp(positions, other))
            denominator = (sum(weight(n_idx, gram) for gram in query_grams)
                           + sum(weight(n_idx, gram) for gram in word_grams))
            parts.append(similarity / max(denominator, 1.0))
        scores[word] = sum(p ** dim for p in parts) / len(parts) if dim != 1 else sum(parts) / len(parts)
    return scores


@pytest.mark.parametrize('exponent', [0.0, 0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize('n_list', [(2,), (1,), (2, 4), (1, 2, 3)])
@pytest.mark.parametrize('query', ['helo', 'hello', 'banana', 'clemetni', 'zzz', 'pineaple'])
def test_pruning_stays_exact_when_weighted(exponent, n_list, query):
    """the index must return exactly what an exhaustive weighted scan returns"""
    index = ApproxWordListV7(n_list, idf_exponent=exponent)
    index.add_words(WORDS)
    expected = brute_force_scores(WORDS, query, n_list, exponent)
    nonzero = {word: score for word, score in expected.items() if score > 0}

    actual = dict(index.lookup(query, top_k=len(WORDS), normalize=True))
    assert set(actual) == set(nonzero), (sorted(set(actual) ^ set(nonzero)), exponent, n_list, query)
    for word, score in nonzero.items():
        assert actual[word] == pytest.approx(score, abs=1e-12), (word, exponent, n_list, query)


@pytest.mark.parametrize('exponent', [0.0, 1.0, 2.0, 4.0])
def test_small_top_k_over_a_large_vocabulary_still_exact(exponent):
    """
    the test above passes top_k=len(WORDS), so `bounds >= kth_best / 2` keeps everything and the
    pruning branch never actually runs. this one asks for a handful of results out of hundreds of
    words over a small alphabet, so the bound genuinely discards candidates -- which is the only
    way to catch a weighted numerator paired with an unweighted bound.
    """
    rng = random.Random(11)
    alphabet = 'abcdefg'
    vocabulary = sorted({''.join(rng.choice(alphabet) for _ in range(rng.randint(4, 8)))
                         for _ in range(600)})
    index = ApproxWordListV7((2, 4), idf_exponent=exponent)
    index.add_words(vocabulary)

    for _ in range(8):
        query = ''.join(rng.choice(alphabet) for _ in range(rng.randint(4, 8)))
        expected = brute_force_scores(vocabulary, query, (2, 4), exponent)
        best = sorted(expected.values(), reverse=True)
        for top_k in (1, 3, 10):
            actual = index.lookup(query, top_k=top_k, normalize=True)
            assert len(actual) == min(top_k, sum(score > 0 for score in expected.values()))
            assert [score for _, score in actual] == pytest.approx(best[:len(actual)], abs=1e-12), \
                (query, top_k, exponent)
            for word, score in actual:
                assert score == pytest.approx(expected[word], abs=1e-12), (word, query, exponent)


@pytest.mark.parametrize('exponent', [0.0, 1.0, 2.0])
def test_top_k_agrees_with_a_full_scan(exponent):
    """
    pruning to top_k must not change the scores that come back, only how many

    compares the score sequence rather than the (word, score) pairs: V7 documents that equal
    scores come back in alphabetical order, but that only holds within one call -- which of two
    tied words survives depends on top_k, e.g. 'hubz' at k=3 returns 'help' where the full scan
    returns the alphabetically earlier 'hell'. that predates idf and is unrelated to it.
    """
    index = ApproxWordListV7((2, 4), idf_exponent=exponent)
    index.add_words(WORDS)
    for query in ['helo', 'camra', 'hubz', 'pineaple']:
        full = index.lookup(query, top_k=len(WORDS), normalize=True)
        for k in (1, 3, 5):
            actual = index.lookup(query, top_k=k, normalize=True)
            assert [score for _, score in actual] == pytest.approx([score for _, score in full[:k]]), \
                (query, k, exponent)


def test_default_is_unweighted():
    assert ApproxWordListV7((2, 4))._weighted is False
    assert ApproxWordListV7((2, 4), idf_exponent=0.0)._weighted is False
    assert ApproxWordListV7((2, 4), idf_exponent=1.0)._weighted is True


def test_exponent_zero_matches_no_exponent_exactly():
    plain, zero = ApproxWordListV7((2, 4)), ApproxWordListV7((2, 4), idf_exponent=0.0)
    plain.add_words(WORDS)
    zero.add_words(WORDS)
    for query in ['helo', 'camra', 'zzz', 'a']:
        assert plain.lookup(query, top_k=10, normalize=True) == zero.lookup(query, top_k=10, normalize=True)


def test_exponent_changes_the_ranking():
    plain, weighted = ApproxWordListV7((2,)), ApproxWordListV7((2,), idf_exponent=3.0)
    plain.add_words(WORDS)
    weighted.add_words(WORDS)
    assert any(plain.lookup(q, top_k=8) != weighted.lookup(q, top_k=8)
               for q in ['helo', 'camra', 'pineaple', 'clemetni'])


@pytest.mark.parametrize('exponent', [0.0, 1.0, 2.0])
def test_exact_match_scores_one(exponent):
    """
    only for words long enough to produce n-grams at every indexed n. a word shorter than that
    contributes 0 to the mean for the n it cannot fill, so it caps below 1 -- 'a' at n=(2, 4)
    scores 0.5 even unweighted, because it has no 4-grams at all. that predates idf.
    """
    index = ApproxWordListV7((2, 4), idf_exponent=exponent)
    index.add_words(WORDS)
    for word in WORDS:
        if num_n_grams(len(word), 4) == 0:
            continue
        top_word, top_score = index.lookup(word, top_k=1, normalize=True)[0]
        assert top_word == word, (word, exponent)
        assert top_score == pytest.approx(1.0), (word, exponent)


@pytest.mark.parametrize('exponent', [0.0, 2.0])
def test_word_too_short_for_one_n_caps_below_one(exponent):
    """pinning the behaviour above, so it is a decision rather than a surprise"""
    index = ApproxWordListV7((2, 4), idf_exponent=exponent)
    index.add_words(WORDS)
    assert index.lookup('a', top_k=1, normalize=True)[0] == ('a', pytest.approx(0.5))
    single = ApproxWordListV7((2,), idf_exponent=exponent)
    single.add_words(WORDS)
    assert single.lookup('a', top_k=1, normalize=True)[0] == ('a', pytest.approx(1.0))


def test_weights_track_the_vocabulary():
    """idf has the vocabulary size in it, so a later add must change earlier words' scores"""
    index = ApproxWordListV7((2,), idf_exponent=2.0)
    index.add_words(['abcd', 'abce', 'abcf'])
    before = index.lookup('abcd', top_k=3, normalize=True)
    index.add_words(['zzzz', 'zzzy', 'zzzx', 'wxyz', 'wxyy'])
    after = index.lookup('abcd', top_k=3, normalize=True)
    assert before != after
    # and the new state must equal a freshly built index with the same words
    fresh = ApproxWordListV7((2,), idf_exponent=2.0)
    fresh.add_words(['abcd', 'abce', 'abcf', 'zzzz', 'zzzy', 'zzzx', 'wxyz', 'wxyy'])
    assert after == fresh.lookup('abcd', top_k=3, normalize=True)


def test_single_word_vocabulary_still_returns_it():
    """log(V / df) would make every n-gram weigh 0 here, zeroing the denominator"""
    index = ApproxWordListV7((2,), idf_exponent=2.0)
    index.add_words(['alpha'])
    results = index.lookup('alpha', top_k=1, normalize=True)
    assert results and results[0][0] == 'alpha'
    assert results[0][1] == pytest.approx(1.0)


def test_repeated_ngrams_are_weighted_too():
    """the multi-occurrence path is a separate branch and must carry the same weight"""
    index = ApproxWordListV7((2,), idf_exponent=2.0)
    index.add_words(['banana', 'bananana', 'abababab', 'band'])
    expected = brute_force_scores(['banana', 'bananana', 'abababab', 'band'], 'banana', (2,), 2.0)
    actual = dict(index.lookup('banana', top_k=4, normalize=True))
    for word, score in expected.items():
        if score > 0:
            assert actual[word] == pytest.approx(score, abs=1e-12), word


def test_invalid_exponent_is_rejected():
    with pytest.raises(ValueError):
        ApproxWordListV7((2,), idf_exponent=-1.0)
    with pytest.raises(TypeError):
        ApproxWordListV7((2,), idf_exponent='2')
