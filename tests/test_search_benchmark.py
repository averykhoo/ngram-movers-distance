"""
the baseline scorers in experiments/search_benchmark.py

these produce the numbers V7 is compared against in docs/bow-plan.md, so a wrong BM25 would
silently flatter or bury the library. each scorer is checked against a hand computation on a
corpus small enough to do by hand, and the shared ranking / rerank / fusion plumbing is
checked for the tie and cutoff rules the protocol promises
"""
import math

import pytest

np = pytest.importorskip('numpy')

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiments'))

import search_benchmark as sb  # noqa: E402

DOCS = sorted(['apple pie', 'apple tart', 'banana bread', 'cherry pie pie'])


class TestRankFromScores:
    def test_ties_break_alphabetically_and_zeros_are_dropped(self):
        scores = np.array([0.5, 0.0, 0.5, 0.9])
        assert sb.rank_from_scores(scores, DOCS) == [DOCS[3], DOCS[0], DOCS[2]]

    def test_cutoff_keeps_boundary_ties_then_truncates_alphabetically(self):
        scores = np.array([0.5, 0.5, 0.5, 0.9])
        assert sb.rank_from_scores(scores, DOCS, top_k=2) == [DOCS[3], DOCS[0]]
        assert sb.rank_from_scores(scores, DOCS, top_k=3) == [DOCS[3], DOCS[0], DOCS[1]]

    def test_all_zero_returns_nothing(self):
        assert sb.rank_from_scores(np.zeros(4), DOCS) == []


class TestBM25:
    def test_matches_hand_computation(self):
        postings = sb.Postings(DOCS, sb.word_tokenizer)
        k1, b = 1.2, 0.75
        system = sb.BM25System(postings, k1, b, 'bm25', {})
        scores = system.scores('pie')
        n = len(DOCS)
        idf = math.log(1.0 + (n - 2 + 0.5) / (2 + 0.5))  # 'pie' is in two documents
        lengths = [len(d.split()) for d in DOCS]
        avg = sum(lengths) / n
        for i, doc in enumerate(DOCS):
            tf = doc.split().count('pie')
            expected = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[i] / avg)) if tf else 0.0
            assert scores[i] == pytest.approx(expected), doc

    def test_repeated_query_term_counts_twice(self):
        postings = sb.Postings(DOCS, sb.word_tokenizer)
        system = sb.BM25System(postings, 1.2, 0.75, 'bm25', {})
        assert system.scores('pie pie') == pytest.approx(2 * system.scores('pie'))

    def test_idf_exponent_raises_the_idf_and_nothing_else(self):
        postings = sb.Postings(DOCS, sb.word_tokenizer)
        plain = sb.BM25System(postings, 1.2, 0.75, 'bm25', {}).scores('pie')
        cubed = sb.BM25System(postings, 1.2, 0.75, 'bm25', {}, idf_exp=3.0).scores('pie')
        assert cubed == pytest.approx(plain * postings.idf['pie'] ** 2)

    def test_unknown_terms_score_nothing(self):
        postings = sb.Postings(DOCS, sb.word_tokenizer)
        assert not sb.BM25System(postings, 1.2, 0.75, 'bm25', {}).scores('durian').any()

    def test_idf_is_positive_even_for_a_term_in_every_document(self):
        postings = sb.Postings(['a b', 'a c', 'a d'], sb.word_tokenizer)
        assert postings.idf['a'] > 0


class TestTfidfCosine:
    def test_identical_text_scores_one_and_disjoint_scores_zero(self):
        postings = sb.Postings(DOCS, sb.char_tokenizer((3,)))
        system = sb.TfidfCosineSystem(postings, 'tfidf', {})
        scores = system.scores('banana bread')
        assert scores[DOCS.index('banana bread')] == pytest.approx(1.0)
        assert scores[DOCS.index('cherry pie pie')] == pytest.approx(0.0, abs=1e-12)

    def test_is_bounded_by_one(self):
        postings = sb.Postings(DOCS, sb.char_tokenizer((2, 3)))
        system = sb.TfidfCosineSystem(postings, 'tfidf', {})
        for query in ['apple', 'pie', 'apple pie tart', 'zzz']:
            assert system.scores(query).max() <= 1.0 + 1e-12


class TestCharTokenizer:
    def test_mirrors_v7_padded_ngrams_tagged_by_n(self):
        tokens = sb.char_tokenizer((2, 3))('ab')
        assert tokens == ['2:\2a', '2:ab', '2:b\3', '3:\2ab', '3:ab\3']


class TestDamerauLevenshtein:
    def test_exact_match_scores_one_and_the_band_excludes(self):
        docs = sorted(['hello', 'help', 'hello world', 'h'])
        system = sb.DamerauLevenshteinSystem(docs, band=2)
        scores = system.scores('hello')
        assert scores[docs.index('hello')] == 1.0
        assert scores[docs.index('help')] == pytest.approx(1 / 3)  # two edits
        assert scores[docs.index('hello world')] == 0.0  # six characters longer: outside the band
        assert scores[docs.index('h')] == 0.0


class _Fixed:
    """a system with a preset score vector"""

    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=float)

    def scores(self, query):
        return self._scores.copy()


class TestRerank:
    def test_keeps_the_first_stage_set_and_orders_by_the_second(self):
        first = _Fixed([0.9, 0.8, 0.7, 0.0])
        second = _Fixed([0.1, 0.5, 0.3, 0.9])
        ranked = sb.rank_from_scores(sb.RerankSystem(first, second, depth=2, label='r').scores('q'), DOCS)
        # depth 2 keeps documents 0 and 1; the second stage puts 1 above 0; document 3 is
        # never admitted however well the second stage likes it
        assert ranked == [DOCS[1], DOCS[0]]

    def test_second_stage_ties_fall_back_to_first_stage_order(self):
        first = _Fixed([0.7, 0.9, 0.8, 0.0])
        second = _Fixed([0.5, 0.5, 0.5, 0.5])
        ranked = sb.rank_from_scores(sb.RerankSystem(first, second, depth=3, label='r').scores('q'), DOCS)
        assert ranked == [DOCS[1], DOCS[2], DOCS[0]]


class TestRRF:
    def test_matches_hand_computation(self):
        a = _Fixed([0.9, 0.8, 0.0, 0.0])  # ranks: doc0 = 1, doc1 = 2
        b = _Fixed([0.0, 0.9, 0.8, 0.0])  # ranks: doc1 = 1, doc2 = 2
        scores = sb.RRFSystem([a, b], depth=10, label='rrf', k=60).scores('q')
        assert scores[0] == pytest.approx(1 / 61)
        assert scores[1] == pytest.approx(1 / 62 + 1 / 61)
        assert scores[2] == pytest.approx(1 / 62)
        assert scores[3] == 0.0


class TestV7System:
    def test_scores_agree_with_lookup(self):
        system = sb.V7System(DOCS, n=(2, 4), idf=1.0, dim=2.0, normalize=True,
                             position_weight=0.5, denominator='geo')
        ranked = sb.rank_from_scores(system.scores('aple pie'), DOCS)
        expected = system.index.lookup('aple pie', top_k=len(DOCS), dim=2.0, normalize=True,
                                       position_weight=0.5, denominator='geo')
        assert ranked == [word for word, _ in expected]


class TestSplit:
    def test_tune_and_test_halves_are_disjoint_and_seeded(self):
        sb.TASKS['_toy'] = (lambda: (DOCS, [(f'q{i}', {DOCS[i % 4]}) for i in range(10)]),
                            Path(__file__))
        try:
            docs, tune, test = sb.load_task('_toy', seed=0)
            docs2, tune2, test2 = sb.load_task('_toy', seed=0)
            assert (tune, test) == (tune2, test2)
            assert len(tune) == 5 and len(test) == 5
            assert not {q for q, _ in tune} & {q for q, _ in test}
        finally:
            del sb.TASKS['_toy']
