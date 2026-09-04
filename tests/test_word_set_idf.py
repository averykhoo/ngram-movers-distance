import math
import random
from collections import Counter

import pytest

from nmd.nmd_word_set import WordSet

WORDS = ['hello', 'yellow', 'help', 'hell', 'world', 'pineapple', 'pine', 'apple',
         'sports', 'hub', 'clementi', 'camera', 'cameras', 'digital']
QUERIES = ['helo', 'pineaple', 'wrld', 'hell', 'xyzzy', 'camra', 'clementy']


def build(words=WORDS, **kwargs):
    word_set = WordSet(**kwargs)
    for word in words:
        word_set.add(word)
    return word_set


def test_default_is_unweighted():
    assert WordSet().idf_exponent == 0.0


def test_exponent_zero_is_bit_identical_to_unweighted():
    # not approx: exponent 0 must take the original code path, so the floats must match exactly
    plain = build()
    zero = build(idf_exponent=0.0)
    for query in QUERIES:
        assert plain.find_similar(query, k=10) == zero.find_similar(query, k=10), query


def test_exponent_changes_the_ranking():
    weighted = build(idf_exponent=2.0)
    plain = build()
    assert any(plain.find_similar(q, k=10) != weighted.find_similar(q, k=10) for q in QUERIES)


@pytest.mark.parametrize('exponent', [0.0, 0.5, 1.0, 2.0])
def test_scores_stay_in_range(exponent):
    word_set = build(idf_exponent=exponent)
    for query in QUERIES:
        for _, score in word_set.find_similar(query, k=10):
            assert 0.0 <= score <= 1.0, (query, score, exponent)


@pytest.mark.parametrize('exponent', [0.0, 1.0, 2.0])
def test_exact_match_still_wins(exponent):
    word_set = build(idf_exponent=exponent)
    for word in WORDS:
        results = word_set.find_similar(word, k=3)
        assert results[0][0] == word, (word, results)
        assert results[0][1] == pytest.approx(1.0)


# --------------------------------------------------------------- incremental document frequency

def brute_force_df(words, n_list):
    """document frequency computed from scratch, the way the incremental counter must agree with"""
    df = Counter()
    for word in words:
        grams = set()
        for n in n_list:
            padded = f'\2{word}\3'
            grams.update(padded[i:i + n] for i in range(len(padded) - n + 1))
        df.update(grams)
    return df


def test_df_matches_a_from_scratch_count():
    word_set = build()
    expected = brute_force_df(WORDS, word_set.ngram_sizes)
    for gram, count in expected.items():
        assert word_set.document_frequency(gram) == count, gram


def test_df_survives_add_and_discard():
    rng = random.Random(0)
    word_set = WordSet()
    present = []
    for _ in range(60):
        if present and rng.random() < 0.4:
            word = rng.choice(present)
            present.remove(word)
            word_set.discard(word)
        else:
            word = rng.choice(WORDS)
            if word in present:
                continue
            present.append(word)
            word_set.add(word)
        expected = brute_force_df(present, word_set.ngram_sizes)
        for gram, count in expected.items():
            assert word_set.document_frequency(gram) == count, (gram, sorted(present))
        # no stale keys left behind by discard
        assert all(word_set.document_frequency(g) == expected.get(g, 0) for g in list(expected))


def test_df_is_cleared():
    word_set = build()
    gram = 'ell'
    assert word_set.document_frequency(gram) >= 0
    word_set.clear()
    assert word_set.document_frequency(gram) == 0
    assert len(word_set) == 0


def test_weights_follow_mutations():
    """the whole point of maintaining df in place: idf must reflect the set as it is now"""
    word_set = WordSet(ngram_sizes=2, idf_exponent=1.0)
    for word in ['camera', 'cameras', 'cameraman']:
        word_set.add(word)
    before = word_set.document_frequency('am')
    word_set.add('ambulance')
    assert word_set.document_frequency('am') == before + 1
    word_set.discard('ambulance')
    assert word_set.document_frequency('am') == before


def test_weight_cache_is_invalidated_by_add():
    """
    a lookup populates the weight cache; adding words changes every idf, so the next lookup must not
    reuse it. checking document_frequency() is not enough -- that reads the counter directly and
    would pass even with a permanently stale cache.
    """
    word_set = WordSet(ngram_sizes=2, idf_exponent=2.0)
    for word in ['abcd', 'abce', 'abcf']:
        word_set.add(word)
    before = word_set.find_similar('abcd', k=3)
    for word in ['zzzz', 'zzzy', 'zzzx', 'wxyz', 'wxyy']:
        word_set.add(word)
    after = word_set.find_similar('abcd', k=3)
    assert before != after, (before, after)


def test_weight_cache_is_invalidated_by_discard():
    word_set = WordSet(ngram_sizes=2, idf_exponent=2.0)
    for word in ['abcd', 'abce', 'abcf', 'zzzz', 'zzzy', 'wxyz']:
        word_set.add(word)
    before = word_set.find_similar('abcd', k=3)
    for word in ['zzzz', 'zzzy', 'wxyz']:
        word_set.discard(word)
    after = word_set.find_similar('abcd', k=3)
    assert before != after, (before, after)


def test_results_do_not_depend_on_mutation_history():
    """
    the strongest invariant for anything cached per word: a set assembled through adds, lookups and
    discards must score exactly like one built fresh with the same final contents. a stale per-word
    weighted total survives the before/after tests above (the weight table still updates, so the
    numerator moves) but breaks this one.
    """
    final_words = ['abcd', 'abce', 'abcf', 'zzzz', 'wxyz']
    fresh = WordSet(ngram_sizes=2, idf_exponent=2.0)
    for word in final_words:
        fresh.add(word)

    incremental = WordSet(ngram_sizes=2, idf_exponent=2.0)
    for word in ['abcd', 'abce']:
        incremental.add(word)
    incremental.find_similar('abcd', k=3)  # populate the caches at an early generation
    for word in ['qqqq', 'qqqr']:
        incremental.add(word)
    incremental.find_similar('abcd', k=3)  # and again, at another
    for word in ['abcf', 'zzzz', 'wxyz']:
        incremental.add(word)
    for word in ['qqqq', 'qqqr']:
        incremental.discard(word)

    assert set(fresh) == set(incremental)
    for query in ['abcd', 'zzzz', 'abcx', 'wxy']:
        expected = dict(fresh.find_similar(query, k=5))
        actual = dict(incremental.find_similar(query, k=5))
        assert expected.keys() == actual.keys(), query
        for word in expected:
            assert expected[word] == pytest.approx(actual[word]), (query, word)


def test_exponent_zero_uses_the_unweighted_code_path():
    """
    the numeric equality test above cannot distinguish the two paths, because weight 1.0 makes them
    agree. pin the branch itself, so the fast path is not silently lost.
    """
    assert WordSet(idf_exponent=0.0)._weighted is False
    assert WordSet(idf_exponent=1.0)._weighted is True


def test_universal_ngrams_do_not_zero_out_the_score():
    """
    with idf = log(N / df) an n-gram in every word weighs exactly 0. a one-word set has every gram
    universal, so every denominator would be 0 and find_similar would return nothing at all.
    """
    for size in (1, 2, 3):
        word_set = WordSet(ngram_sizes=2, idf_exponent=2.0)
        for word in ['alpha', 'alphb', 'alphc'][:size]:
            word_set.add(word)
        results = word_set.find_similar('alpha', k=3)
        assert results, f'no results from a {size}-word set'
        assert results[0][0] == 'alpha'
        assert results[0][1] == pytest.approx(1.0)


def test_negative_exponent_is_rejected():
    with pytest.raises(ValueError):
        WordSet(idf_exponent=-1.0)
    with pytest.raises(TypeError):
        WordSet(idf_exponent='2')


def test_idf_upweights_the_rare_gram():
    # 'q' appears in one word only; 'e' appears in all of them. with a large exponent the word
    # sharing the rare character must outrank the word sharing the common one
    word_set = WordSet(ngram_sizes=1, idf_exponent=3.0)
    for word in ['qe', 'ea', 'eb', 'ec', 'ed']:
        word_set.add(word)
    ranked = [word for word, _ in word_set.find_similar('qz', k=5)]
    assert ranked[0] == 'qe', ranked
    plain = WordSet(ngram_sizes=1)
    for word in ['qe', 'ea', 'eb', 'ec', 'ed']:
        plain.add(word)
    assert plain.find_similar('qz', k=5)[0][0] == 'qe'  # still first, but for a weaker reason
    assert math.isclose(word_set.find_similar('qz', k=1)[0][1], word_set.find_similar('qz', k=1)[0][1])
