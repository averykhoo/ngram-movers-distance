"""
is nmd-as-an-index useful for text search, or should it stay out of that business?

`index_param_search.py` tunes `ApproxWordListV7` against itself. this script puts the tuned
index next to the systems anyone would reach for instead, on the same tasks, under the same
protocol, and asks three questions the parameter search cannot:

    1. does V7 beat BM25 / tf-idf over the *same character n-grams*? that is what elasticsearch
       with an `ngram` tokenizer is, so it is the fair comparison -- token BM25 is included as
       the thing people actually deploy for documents, and brute-force damerau-levenshtein as
       the thing they deploy for typos
    2. does the positional (mover's distance) part of the score help at all? `position_weight=0`
       is plain dice over n-gram multisets, so the best-over-everything-else score at pw=0 vs
       pw>0 is the value of position, per task
    3. if position does not help as a first-stage ranker, does it help as a *reranker* of the
       top of a BM25 list, fused with it (RRF), or interpolated into its score with a tuned
       weight (mix)? word order is extra information; if it cannot be made to pay even there,
       the heuristic is not informative on that task

protocol. every task's queries are split in half by seed: each system picks its configuration
on the *tune* half (best MAP) and is reported on the *test* half. per-task tuning is allowed --
typo correction and entity search are different jobs -- but it is symmetric: BM25 gets its k1/b
and n-gram size tuned the same way V7 gets its knobs, and the untuned library default is
reported alongside so the cost of not tuning is visible. ranked lists are cut at TOP_K=50 and
ties are broken alphabetically in every system, exactly as V7 does. a document scoring 0 is never
returned, so all metrics are cutoff metrics at 50.

tasks. the six from index_param_search, plus:

    norvig        real misspellings: Roger Mitton's Birkbeck-derived sets as distributed with
                  Norvig's spelling corrector (spell-testset1/2.txt). the dictionary is
                  words_en.txt plus every correct answer, so it is the `typo` protocol with
                  human errors instead of `corrupt`'s uniform random edits
    dblp_acm, dblp_scholar, amazon_google, walmart_amazon, fodors_zagats, itunes_amazon,
    dirty_dblp_acm, dirty_walmart_amazon
                  the rest of the Magellan entity-matching collection in ditto's format,
                  reframed into retrieval exactly like `ermagellan` (which is Abt-Buy from the
                  same source). structured records of ~100-300 characters
    scifact, nfcorpus
                  BEIR document retrieval: title + abstract, ~1.5k characters, natural-language
                  queries with graded qrels. this is where BM25 is the published baseline
                  (SciFact nDCG@10 0.665 for token BM25 in the BEIR paper, on all 300 test
                  queries) and where a character n-gram matcher is expected to lose. it is here
                  as the negative boundary, so the answer to "should this be used for search"
                  has a number attached

everything beyond the six lives in gitignored .scratch/data/ (see `DATASETS` for the sources)
and skips itself when absent. results go to experiments/results/search_benchmark_<task>.csv;
`--report` turns those into experiments/results/search_benchmark_summary.md.

run:
    python experiments/search_benchmark.py typo                 # one task
    python experiments/search_benchmark.py typo,norvig          # several, comma-separated
    python experiments/search_benchmark.py all                  # every task with data present
    python experiments/search_benchmark.py all --quick          # small grids, smoke test
    python experiments/search_benchmark.py all --backfill       # add rerank/fusion/mix rows a CSV lacks
    python experiments/search_benchmark.py all --baselines      # keep V7 rows, re-run the rest
    python experiments/search_benchmark.py --report
"""
import argparse
import csv
import itertools
import json
import math
import random
import re
import sys
import time
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import index_param_search as ips
from edit_distance import damerau_levenshtein_distance
from ranking_metrics import average_precision
from ranking_metrics import hit_at_k
from ranking_metrics import ndcg_at_k
from ranking_metrics import recall_at_k
from ranking_metrics import reciprocal_rank

from nmd.nmd_index_v7 import ApproxWordListV7
from nmd.nmd_index_v7 import n_grams

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / '.scratch' / 'data'
RESULTS = ROOT / 'experiments' / 'results'

TOP_K = 50
RERANK_DEPTH = 50
RRF_K = 60
MAX_QUERIES = 300

# where the extra data comes from. none of it is tracked
DATASETS = {
    'ditto': 'https://github.com/megagonlabs/ditto/tree/master/data/er_magellan -> '
             '.scratch/data/ditto/<Type>_<Name>/{train,valid,test}.txt',
    'typos': 'https://norvig.com/spell-testset1.txt and spell-testset2.txt -> .scratch/data/typos/',
    'beir': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<name>.zip '
            'unzipped -> .scratch/data/beir/<name>/',
}

Query = Tuple[str, Set[str]]


# --- tasks --------------------------------------------------------------------------------

def load_norvig() -> Tuple[List[str], List[Query]]:
    """`correct: misspelling misspelling ...` lines; dictionary is words_en.txt plus the answers"""
    with open(ips.WORDS, encoding='utf8') as f:
        vocab = {w.strip().casefold() for w in f
                 if w.strip().isalpha() and 3 <= len(w.strip()) <= 15}
    queries: List[Query] = []
    for name in ('spell-testset1.txt', 'spell-testset2.txt'):
        for line in (DATA / 'typos' / name).read_text(encoding='utf8').splitlines():
            if ':' not in line:
                continue
            answer, wrong = line.split(':', 1)
            answer = answer.strip().casefold()
            if not answer.isalpha():
                continue
            vocab.add(answer)
            for misspelling in wrong.split():
                misspelling = misspelling.strip().casefold()
                if misspelling.isalpha() and misspelling != answer:
                    queries.append((misspelling, {answer}))
    return sorted(vocab), queries


def load_ditto(name: str) -> Tuple[List[str], List[Query]]:
    """
    any ditto-format Magellan set, reframed like ips.load_ermagellan: the right entities of all
    splits are the corpus, each left entity with a positive in valid or test is a query.
    (ermagellan draws queries from valid only, which is plenty for Abt-Buy; the smaller sets
    here have 22-27 positives per split, so both are used)
    """
    rights, truth = set(), defaultdict(set)
    for split in ('train', 'valid', 'test'):
        path = DATA / 'ditto' / name / f'{split}.txt'
        for line in path.read_text(encoding='utf8').splitlines():
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            left, right, label = (ips.clean(parts[0]), ips.clean(parts[1]), parts[2])
            rights.add(right)
            if split != 'train' and label == '1':
                truth[left].add(right)
    documents = sorted(r for r in rights if r)
    queries = [(q, rel) for q, rel in sorted(truth.items()) if q and rel]
    return documents, queries


def load_beir(name: str) -> Tuple[List[str], List[Query]]:
    """title + abstract as the document; relevance is any qrel score > 0 on the test split"""
    base = DATA / 'beir' / name
    text_of: Dict[str, str] = {}
    for line in (base / 'corpus.jsonl').read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        text_of[row['_id']] = ips.clean(f"{row.get('title', '')} {row.get('text', '')}")
    docs_by_text: Dict[str, Set[str]] = defaultdict(set)
    for doc_id, text in text_of.items():
        if text:
            docs_by_text[text].add(doc_id)
    query_text = {}
    for line in (base / 'queries.jsonl').read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        query_text[row['_id']] = ips.clean(row['text'])
    relevant_ids: Dict[str, Set[str]] = defaultdict(set)
    with open(base / 'qrels' / 'test.tsv', encoding='utf8') as f:
        next(f)
        for line in f:
            qid, did, score = line.rstrip('\n').split('\t')
            if float(score) > 0:
                relevant_ids[qid].add(did)
    queries = []
    for qid in sorted(relevant_ids):
        relevant = {text_of[d] for d in relevant_ids[qid] if d in text_of and text_of[d]}
        if query_text.get(qid) and relevant:
            queries.append((query_text[qid], relevant))
    return sorted(docs_by_text), queries


TASKS: Dict[str, Tuple[Callable, Path]] = {
    **{task: (lambda t=task: ips.load_task(t), required) for task, (_, required) in ips.TASKS.items()},
    'norvig': (load_norvig, DATA / 'typos' / 'spell-testset1.txt'),
    'dblp_acm': (lambda: load_ditto('Structured_DBLP-ACM'), DATA / 'ditto' / 'Structured_DBLP-ACM' / 'valid.txt'),
    'dblp_scholar': (lambda: load_ditto('Structured_DBLP-GoogleScholar'),
                     DATA / 'ditto' / 'Structured_DBLP-GoogleScholar' / 'valid.txt'),
    'amazon_google': (lambda: load_ditto('Structured_Amazon-Google'),
                      DATA / 'ditto' / 'Structured_Amazon-Google' / 'valid.txt'),
    'walmart_amazon': (lambda: load_ditto('Structured_Walmart-Amazon'),
                       DATA / 'ditto' / 'Structured_Walmart-Amazon' / 'valid.txt'),
    'fodors_zagats': (lambda: load_ditto('Structured_Fodors-Zagats'),
                      DATA / 'ditto' / 'Structured_Fodors-Zagats' / 'valid.txt'),
    'itunes_amazon': (lambda: load_ditto('Structured_iTunes-Amazon'),
                      DATA / 'ditto' / 'Structured_iTunes-Amazon' / 'valid.txt'),
    'dirty_dblp_acm': (lambda: load_ditto('Dirty_DBLP-ACM'), DATA / 'ditto' / 'Dirty_DBLP-ACM' / 'valid.txt'),
    'dirty_walmart_amazon': (lambda: load_ditto('Dirty_Walmart-Amazon'),
                             DATA / 'ditto' / 'Dirty_Walmart-Amazon' / 'valid.txt'),
    'scifact': (lambda: load_beir('scifact'), DATA / 'beir' / 'scifact' / 'corpus.jsonl'),
    'nfcorpus': (lambda: load_beir('nfcorpus'), DATA / 'beir' / 'nfcorpus' / 'corpus.jsonl'),
}
TYPO_TASKS = ('typo', 'typo_ms', 'typo_hard', 'typo_brutal', 'norvig')


def load_task(task: str, seed: int = 0):
    """(documents, tune queries, test queries) or None when the data is absent"""
    loader, required = TASKS[task]
    if not required.exists():
        print(f'\n=== {task}: skipped, {required} not found ===')
        return None
    loaded = loader()
    if loaded is None:
        return None
    documents, queries = loaded
    rng = random.Random(seed)
    queries = list(queries)
    rng.shuffle(queries)
    queries = queries[:MAX_QUERIES]
    half = len(queries) // 2
    return documents, queries[:half], queries[half:]


# --- systems ------------------------------------------------------------------------------
# every system scores the whole corpus: `scores(query)` returns a float array over document
# indices (0 == no match). ranking, cutoff and tie-breaking are then identical for all of them

def rank_from_scores(scores: np.ndarray, documents: Sequence[str], top_k: int = TOP_K) -> List[str]:
    """best first, alphabetical within a tie (documents are sorted, so a stable sort does it)"""
    if scores.size > top_k:
        kth = scores[np.argpartition(scores, scores.size - top_k)[-top_k]]
        candidates = np.flatnonzero(scores >= kth if kth > 0 else scores > 0)
    else:
        candidates = np.flatnonzero(scores > 0)
    order = candidates[np.argsort(-scores[candidates], kind='stable')][:top_k]
    return [documents[i] for i in order]


class V7System:
    """ApproxWordListV7 at one constructor + lookup configuration"""

    def __init__(self, documents: Sequence[str], n: tuple, idf: float, dim: float,
                 normalize: bool, position_weight: float, denominator: str,
                 index: Optional[ApproxWordListV7] = None):
        self.documents = documents
        self.doc_index = {doc: i for i, doc in enumerate(documents)}
        self.index = index if index is not None else ApproxWordListV7(n, idf_exponent=idf).add_words(documents)
        self.index._freeze()
        self.kwargs = dict(dim=dim, normalize=normalize, position_weight=position_weight,
                           denominator=denominator)
        self.config = dict(n=repr(n), idf=idf, dim=dim, normalize=normalize,
                           position_weight=position_weight, denominator=denominator)

    def scores(self, query: str) -> np.ndarray:
        found = self.index._similarity_vectors(query, self.kwargs['normalize'], self.kwargs['dim'],
                                               self.kwargs['position_weight'], self.kwargs['denominator'])
        if found is None:
            return np.zeros(len(self.documents))
        return found[1]


class Postings:
    """
    sparse term -> (doc indices, term frequencies) over one tokenization, shared by the BM25
    and tf-idf scorers so that a k1/b sweep does not rebuild the index
    """

    def __init__(self, documents: Sequence[str], tokenize: Callable[[str], List[str]]):
        self.tokenize = tokenize
        self.num_docs = len(documents)
        lists: Dict[str, List[int]] = defaultdict(list)
        tfs: Dict[str, List[int]] = defaultdict(list)
        lengths = np.zeros(self.num_docs)
        for i, doc in enumerate(documents):
            counts = Counter(tokenize(doc))
            lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                lists[term].append(i)
                tfs[term].append(tf)
        self.postings = {term: (np.array(idxs, dtype=np.int32), np.array(tfs[term], dtype=np.float64))
                         for term, idxs in lists.items()}
        self.lengths = lengths
        self.avg_length = lengths.mean() if self.num_docs else 1.0
        # lucene's form: always positive, so a matched term never pushes a document below 0
        self.idf = {term: math.log(1.0 + (self.num_docs - idx.size + 0.5) / (idx.size + 0.5))
                    for term, (idx, _) in self.postings.items()}


def char_tokenizer(n_list: tuple) -> Callable[[str], List[str]]:
    """the same padded n-grams V7 indexes, tagged by n so a 2-gram and a 4-gram never collide"""
    def tokenize(text: str) -> List[str]:
        out = []
        for n in n_list:
            out.extend(f'{n}:{g}' for g in n_grams(text, n))
        return out
    return tokenize


_WORD = re.compile(r'\w+')


def word_tokenizer(text: str) -> List[str]:
    return _WORD.findall(text)


class BM25System:
    """
    BM25 with `idf ** idf_exp` in place of idf. textbook BM25 is `idf_exp=1`; the exponent is
    swept because V7 gets one (`idf_exponent`), and without it the comparison flatters V7:
    measured 2026-09-06 on ermagellan, n=(2,3), test MAP 0.862 at idf^1 against 0.909 at idf^3
    (the probe is transcribed in docs/bow-plan.md Part 10)
    """

    def __init__(self, postings: Postings, k1: float, b: float, label: str, config: dict,
                 idf_exp: float = 1.0):
        self.postings = postings
        self.k1, self.b, self.idf_exp = k1, b, idf_exp
        self.norm = k1 * (1.0 - b + b * postings.lengths / postings.avg_length)
        self.label = label
        self.config = dict(config, k1=k1, b=b, idf_exp=idf_exp)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.postings.num_docs)
        # query term frequency multiplies in, as in robertson's original with k3 -> infinity;
        # a repeated query n-gram counts twice, which is also what V7's mass does
        for term, qtf in Counter(self.postings.tokenize(query)).items():
            entry = self.postings.postings.get(term)
            if entry is None:
                continue
            idx, tf = entry
            idf = self.postings.idf[term] ** self.idf_exp
            out[idx] += qtf * idf * tf * (self.k1 + 1.0) / (tf + self.norm[idx])
        return out


class TfidfCosineSystem:
    """log-tf * idf, cosine -- the baseline_eval.py winner on product names, over any n-list"""

    def __init__(self, postings: Postings, label: str, config: dict):
        self.postings = postings
        self.label = label
        self.config = dict(config)
        norms_sq = np.zeros(postings.num_docs)
        self.weights: Dict[str, np.ndarray] = {}
        for term, (idx, tf) in postings.postings.items():
            w = (1.0 + np.log(tf)) * postings.idf[term]
            self.weights[term] = w
            norms_sq[idx] += w * w
        self.norms = np.sqrt(norms_sq)
        self.norms[self.norms == 0] = 1.0

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.postings.num_docs)
        q_norm_sq = 0.0
        for term, qtf in Counter(self.postings.tokenize(query)).items():
            idf = self.postings.idf.get(term)
            if idf is None:
                continue
            qw = (1.0 + math.log(qtf)) * idf
            q_norm_sq += qw * qw
            idx, _ = self.postings.postings[term]
            out[idx] += qw * self.weights[term]
        if q_norm_sq > 0:
            out /= self.norms * math.sqrt(q_norm_sq)
        return out


class DamerauLevenshteinSystem:
    """
    brute force over the dictionary, scored 1 / (1 + distance), with a length band so it is
    affordable -- a word more than `band` characters longer or shorter than the query cannot
    be within `band` edits. this is what a SymSpell / levenshtein-automaton lookup returns,
    minus their speed
    """
    label = 'damerau_levenshtein'
    config: dict = dict(band=3)

    def __init__(self, documents: Sequence[str], band: int = 3):
        self.documents = documents
        self.band = band
        self.by_length: Dict[int, List[int]] = defaultdict(list)
        for i, doc in enumerate(documents):
            self.by_length[len(doc)].append(i)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.documents))
        for length in range(len(query) - self.band, len(query) + self.band + 1):
            for i in self.by_length.get(length, ()):
                d = damerau_levenshtein_distance(query, self.documents[i])
                if d <= self.band:
                    out[i] = 1.0 / (1.0 + d)
        return out


class RerankSystem:
    """first stage's top `depth`, reordered by the second system's score (ties by first stage)"""

    def __init__(self, first, second, depth: int, label: str):
        self.first, self.second, self.depth, self.label = first, second, depth, label
        self.config = dict(depth=depth)

    def scores(self, query: str) -> np.ndarray:
        first = self.first.scores(query)
        out = np.zeros_like(first)
        if first.size > self.depth:
            kth = first[np.argpartition(first, first.size - self.depth)[-self.depth]]
            keep = np.flatnonzero(first >= kth if kth > 0 else first > 0)
        else:
            keep = np.flatnonzero(first > 0)
        if keep.size == 0:
            return out
        second = self.second.scores(query)[keep]
        # strictly positive so a reranked document is never confused with an unretrieved one,
        # and the first-stage score breaks second-stage ties (scaled below its resolution)
        out[keep] = 1.0 + second + 1e-9 * first[keep] / max(first[keep].max(), 1e-300)
        return out


class MixSystem:
    """
    (1 - lam) * first + lam * second, each score vector divided by its per-query maximum so the
    two are on the same scale. at small `lam` the second system only breaks the first's near-ties,
    which is the weakest form of "does position help": RerankSystem replaces the first stage's
    order entirely, and where the first stage's top-50 recall is ~1 that is just the second system
    """

    def __init__(self, first, second, lam: float, label: str):
        self.first, self.second, self.lam, self.label = first, second, lam, label
        self.config = dict(lam=lam)

    def scores(self, query: str) -> np.ndarray:
        a = self.first.scores(query)
        b = self.second.scores(query)
        a = a / a.max() if a.max() > 0 else a
        b = b / b.max() if b.max() > 0 else b
        return (1.0 - self.lam) * a + self.lam * b


class RRFSystem:
    """reciprocal rank fusion of two systems' top-`depth` lists"""

    def __init__(self, systems: Sequence, depth: int, label: str, k: int = RRF_K):
        self.systems, self.depth, self.label, self.k = systems, depth, label, k
        self.config = dict(depth=depth, k=k)

    def scores(self, query: str) -> np.ndarray:
        out = None
        for system in self.systems:
            s = system.scores(query)
            if out is None:
                out = np.zeros_like(s)
            if s.size > self.depth:
                kth = s[np.argpartition(s, s.size - self.depth)[-self.depth]]
                keep = np.flatnonzero(s >= kth if kth > 0 else s > 0)
            else:
                keep = np.flatnonzero(s > 0)
            order = keep[np.argsort(-s[keep], kind='stable')][:self.depth]
            out[order] += 1.0 / (self.k + np.arange(1, order.size + 1))
        return out


# --- evaluation ---------------------------------------------------------------------------

METRICS = ('map', 'mrr', 'ndcg@10', 'hit@1', 'recall@10', 'recall@50')


def evaluate(system, documents: Sequence[str], queries: Sequence[Query]) -> Dict[str, float]:
    totals = defaultdict(float)
    start = time.perf_counter()
    for query, relevant in queries:
        ranked = rank_from_scores(system.scores(query), documents)
        totals['map'] += average_precision(ranked, relevant)
        totals['mrr'] += reciprocal_rank(ranked, relevant)
        totals['ndcg@10'] += ndcg_at_k(ranked, relevant, k=10)
        totals['hit@1'] += hit_at_k(ranked, relevant, k=1)
        totals['recall@10'] += recall_at_k(ranked, relevant, k=10)
        totals['recall@50'] += recall_at_k(ranked, relevant, k=50)
    out = {key: totals[key] / max(1, len(queries)) for key in METRICS}
    out['ms_per_query'] = 1000.0 * (time.perf_counter() - start) / max(1, len(queries))
    return out


# --- grids --------------------------------------------------------------------------------
# V7's grid is index_param_search's unified grid minus what never won there, with unigrams
# dropped on long documents (n=(1,) on 138-char documents costs 341 s per configuration and
# scores MAP 0.008 -- every document contains every letter)
V7_N_SHORT = [(1, 2), (2,), (2, 3), (2, 4), (1, 2, 3)]
V7_N_LONG = [(2,), (3,), (2, 3), (2, 4), (3, 4)]
# documents drop n=2 for the same reason records drop n=1: on scifact (1 499-char documents)
# n=(2,3) scored nDCG@10 0.296 at 550 ms per query against 0.523 at 90 ms for n=(3,4), measured
# 2026-09-06 on the test half -- every abstract contains nearly every bigram
V7_N_DOC = [(3,), (4,), (3, 4)]
V7_IDF = [0.0, 1.0, 2.0, 3.0]
V7_DIM = [1.0, 2.0]
V7_NORM = [(False, 'dice'), (True, 'dice'), (True, 'geo')]
V7_POSITION_WEIGHT = [0.0, 0.5, 1.0]
V7_DEFAULT = dict(n=(2, 4), idf=0.0, dim=1.0, normalize=True, position_weight=1.0, denominator='dice')

BM25_CHAR_N = [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4)]
BM25_K1 = [0.5, 1.2, 2.0]
BM25_B = [0.3, 0.75, 1.0]
BM25_IDF = [1.0, 2.0, 3.0]  # 1.0 is textbook BM25; see BM25System
LONG_DOC_CHARS = 60  # above this mean length V7_N_LONG applies ...
DOC_CHARS = 600  # ... and above this V7_N_DOC


def v7_configs(mean_chars: float, quick: bool) -> List[dict]:
    n_grid = V7_N_DOC if mean_chars > DOC_CHARS else V7_N_LONG if mean_chars > LONG_DOC_CHARS else V7_N_SHORT
    if quick:
        n_grid, idf_grid, pw_grid = n_grid[:2], V7_IDF[:2], [0.0, 1.0]
    else:
        idf_grid, pw_grid = V7_IDF, V7_POSITION_WEIGHT
    out = []
    for n, idf in itertools.product(n_grid, idf_grid):
        for dim in (V7_DIM if len(n) > 1 else [1.0]):
            for normalize, denominator in V7_NORM:
                for pw in pw_grid:
                    out.append(dict(n=n, idf=idf, dim=dim, normalize=normalize,
                                    position_weight=pw, denominator=denominator))
    return out


# --- driver -------------------------------------------------------------------------------

def write_rows(task: str, rows: List[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f'search_benchmark_{task}.csv'
    fields = ['task', 'system', 'family', 'config', 'split', 'selected', 'n_queries',
              *METRICS, 'ms_per_query']
    with open(path, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')


def row(task: str, system: str, family: str, config: dict, split: str, selected: bool,
        queries: Sequence[Query], metrics: Dict[str, float]) -> dict:
    return dict(task=task, system=system, family=family, config=json.dumps(config, sort_keys=True),
                split=split, selected=selected, n_queries=len(queries),
                **{k: round(v, 5) for k, v in metrics.items()})


def report_line(label: str, metrics: Dict[str, float]) -> str:
    return (f'  {label:<48s} MAP {metrics["map"]:.4f}  MRR {metrics["mrr"]:.4f}  '
            f'nDCG@10 {metrics["ndcg@10"]:.4f}  R@50 {metrics["recall@50"]:.4f}  '
            f'{metrics["ms_per_query"]:7.2f} ms')


def run_task(task: str, quick: bool, seed: int = 0, baselines_only: bool = False) -> None:
    """
    the whole protocol for one task. `baselines_only` keeps the V7 rows already in the task's
    CSV (the V7 grid is the part that takes hours) and re-runs everything else against them,
    for when a baseline's grid changes
    """
    loaded = load_task(task, seed)
    if loaded is None:
        return
    documents, tune, test = loaded
    mean_chars = sum(map(len, documents)) / max(1, len(documents))
    print(f'\n=== {task}: {len(documents)} documents ({mean_chars:.0f} chars mean), '
          f'{len(tune)} tune + {len(test)} test queries, top_k={TOP_K} ===', flush=True)
    rows: List[dict] = []
    selected: Dict[str, object] = {}

    def sweep(family: str, candidates: List[Tuple[str, Callable[[], object]]]) -> None:
        select_on_tune(task, family, candidates, documents, tune, test, rows, selected)

    if baselines_only:
        kept = ips.read_csv(RESULTS / f'search_benchmark_{task}.csv')
        rows.extend(r for r in kept if r['family'].startswith('v7'))
        selected['v7'] = rebuild(documents, next(r for r in rows if r['family'] == 'v7' and r['split'] == 'test'))
        print(f'  keeping {len(rows)} V7 rows; selected {selected["v7"].config}', flush=True)
        run_baselines(task, quick, documents, tune, test, rows, selected, sweep)
        write_rows(task, rows)
        return

    # V7: one index per (n, idf), the read-path knobs swept on it. the grid is ordered by
    # (n, idf) and the cache holds one index at a time, so a long-document task never has
    # twenty indexes alive at once
    cache: Dict[tuple, ApproxWordListV7] = {}

    def make_index(n: tuple, idf: float) -> ApproxWordListV7:
        if (n, idf) not in cache:
            cache.clear()
            cache[(n, idf)] = ApproxWordListV7(n, idf_exponent=idf).add_words(documents)
        return cache[(n, idf)]

    v7_candidates = []
    for config in v7_configs(mean_chars, quick):
        label = ('v7 n={n} idf={idf} dim={dim} norm={normalize}/{denominator} pw={position_weight}'
                 .format(**config))
        v7_candidates.append((label, lambda c=config: V7System(documents, index=make_index(c['n'], c['idf']), **c)))
    sweep('v7', v7_candidates)
    cache.clear()

    # the library default, untuned, so the cost of not tuning is visible
    default = V7System(documents, **V7_DEFAULT)
    metrics = evaluate(default, documents, test)
    rows.append(row(task, 'v7 default', 'v7_default', default.config, 'test', True, test, metrics))
    selected['v7_default'] = default
    print(report_line('v7_default n=(2, 4) idf=0 dim=1 dice pw=1', metrics), flush=True)

    # the position ablation: best over every other knob, per position_weight, on test. reported
    # from the tune-selected config *per pw*, so it is three fair selections, not one
    for pw in (V7_POSITION_WEIGHT if not quick else [0.0, 1.0]):
        best_score, best = -1.0, None
        for r in rows:
            if r['family'] == 'v7' and r['split'] == 'tune' and json.loads(r['config'])['position_weight'] == pw:
                if r['map'] > best_score:
                    best_score, best = r['map'], r
        config = json.loads(best['config'])
        config['n'] = tuple(int(x) for x in re.findall(r'\d+', config['n']))
        system = V7System(documents, **config)
        metrics = evaluate(system, documents, test)
        rows.append(row(task, best['system'], f'v7_pw{pw}', system.config, 'test', True, test, metrics))
        print(report_line(f'v7_pw{pw} <- {best["system"]}', metrics), flush=True)

    run_baselines(task, quick, documents, tune, test, rows, selected, sweep)
    write_rows(task, rows)


def select_on_tune(task: str, family: str, candidates: List[Tuple[str, Callable[[], object]]],
                   documents: Sequence[str], tune: Sequence[Query], test: Sequence[Query],
                   rows: List[dict], selected: Dict[str, object]) -> None:
    """build each candidate, score it on tune, keep the best, report that on test"""
    best_score, best = -1.0, None
    t0 = time.perf_counter()
    for label, make in candidates:
        system = make()
        metrics = evaluate(system, documents, tune)
        rows.append(row(task, label, family, system.config, 'tune', False, tune, metrics))
        if metrics['map'] > best_score:
            best_score, best = metrics['map'], (label, system)
    label, system = best
    metrics = evaluate(system, documents, test)
    rows.append(row(task, label, family, system.config, 'test', True, test, metrics))
    selected[family] = system
    print(report_line(f'{family} <- {label}', metrics) +
          f'   [{len(candidates)} configs, {time.perf_counter() - t0:.0f}s]', flush=True)


def run_baselines(task: str, quick: bool, documents: Sequence[str], tune: Sequence[Query],
                  test: Sequence[Query], rows: List[dict], selected: Dict[str, object],
                  sweep: Callable) -> None:
    """every non-V7 system, then the rerank / fusion combinations with the selected V7"""
    # character n-gram BM25 and tf-idf cosine over the same features
    bm25_candidates, tfidf_candidates = [], []
    char_n = BM25_CHAR_N[:2] if quick else BM25_CHAR_N
    for n in char_n:
        postings = Postings(documents, char_tokenizer(n))
        for k1, b, idf_exp in itertools.product(BM25_K1, BM25_B, BM25_IDF):
            label = f'bm25_char n={n} k1={k1} b={b} idf^{idf_exp}'
            bm25_candidates.append((label, lambda p=postings, k1=k1, b=b, e=idf_exp, n=n, label=label:
                                    BM25System(p, k1, b, label, dict(n=repr(n)), idf_exp=e)))
        label = f'tfidf_char n={n}'
        tfidf_candidates.append((label, lambda p=postings, n=n, label=label:
                                 TfidfCosineSystem(p, label, dict(n=repr(n)))))
    sweep('bm25_char', bm25_candidates)
    sweep('tfidf_char', tfidf_candidates)

    # token BM25: what people deploy for documents. on single-word tasks it can only match
    # exact words, which is the point of including it there
    postings = Postings(documents, word_tokenizer)
    sweep('bm25_word', [(f'bm25_word k1={k1} b={b}',
                         lambda p=postings, k1=k1, b=b: BM25System(p, k1, b, f'bm25_word k1={k1} b={b}', {}))
                        for k1, b in itertools.product(BM25_K1, BM25_B)])

    # brute-force edit distance, only where the strings are short enough for it to mean anything
    if task in TYPO_TASKS:
        sweep('damerau_levenshtein', [('damerau_levenshtein', lambda: DamerauLevenshteinSystem(documents))])

    # reranking and fusion of the selected systems, on test only (nothing to tune) ...
    for family, system in combinations(selected):
        metrics = evaluate(system, documents, test)
        rows.append(row(task, system.label, family, system.config, 'test', True, test, metrics))
        print(report_line(family, metrics), flush=True)
    # ... and the interpolations, whose weight is tuned
    for family, candidates in mixes(selected):
        sweep(family, candidates)


MIX_LAMBDAS = [0.1, 0.25, 0.5, 0.75, 0.9]


def mixes(selected: Dict[str, object]) -> List[Tuple[str, List[Tuple[str, Callable[[], object]]]]]:
    """score interpolations of a baseline with the selected V7, one candidate per weight"""
    v7 = selected['v7']
    out = []
    for name in ('bm25_char', 'tfidf_char'):
        base = selected[name]
        out.append((f'mix({name},v7)',
                    [(f'{name} + v7 at lam={lam}',
                      lambda b=base, lam=lam, name=name: MixSystem(b, v7, lam, f'{name} + v7 at lam={lam}'))
                     for lam in MIX_LAMBDAS]))
    return out


def combinations(selected: Dict[str, object]) -> List[Tuple[str, object]]:
    """the rerank / fusion pairs built from the tune-selected first-stage systems"""
    v7, bm25, tfidf = selected['v7'], selected['bm25_char'], selected['tfidf_char']
    return [
        ('bm25_char>v7', RerankSystem(bm25, v7, RERANK_DEPTH, 'bm25_char top-50 reranked by v7')),
        ('v7>bm25_char', RerankSystem(v7, bm25, RERANK_DEPTH, 'v7 top-50 reranked by bm25_char')),
        ('rrf(v7,bm25_char)', RRFSystem([v7, bm25], RERANK_DEPTH, 'rrf of v7 and bm25_char')),
        ('tfidf_char>v7', RerankSystem(tfidf, v7, RERANK_DEPTH, 'tfidf_char top-50 reranked by v7')),
        ('rrf(v7,tfidf_char)', RRFSystem([v7, tfidf], RERANK_DEPTH, 'rrf of v7 and tfidf_char')),
    ]


def rebuild(documents: Sequence[str], r: dict):
    """the system a results row describes, rebuilt from its `family` and `config`"""
    config = json.loads(r['config'])
    if r['family'] == 'v7':
        config['n'] = tuple(int(x) for x in re.findall(r'\d+', config['n']))
        return V7System(documents, **config)
    n = tuple(int(x) for x in re.findall(r'\d+', config['n']))
    postings = Postings(documents, char_tokenizer(n))
    if r['family'] == 'bm25_char':
        return BM25System(postings, config['k1'], config['b'], r['system'], dict(n=repr(n)),
                          idf_exp=config.get('idf_exp', 1.0))
    if r['family'] == 'tfidf_char':
        return TfidfCosineSystem(postings, r['system'], dict(n=repr(n)))
    raise ValueError(r['family'])


def backfill_combinations(task: str, seed: int = 0) -> None:
    """
    add any combination rows missing from a task's CSV, rebuilding the selected first-stage
    systems from their recorded configurations. for CSVs written before a combination existed;
    a fresh `run_task` writes them all
    """
    path = RESULTS / f'search_benchmark_{task}.csv'
    rows = ips.read_csv(path)
    have = {r['family'] for r in rows if r['split'] == 'test'}
    loaded = load_task(task, seed)
    if loaded is None:
        return
    documents, tune, test = loaded
    selected = {r['family']: rebuild(documents, r) for r in rows
                if r['split'] == 'test' and r['family'] in ('v7', 'bm25_char', 'tfidf_char')}
    added = 0
    for family, system in combinations(selected):
        if family in have:
            continue
        metrics = evaluate(system, documents, test)
        rows.append(row(task, system.label, family, system.config, 'test', True, test, metrics))
        print(report_line(f'{task} {family}', metrics), flush=True)
        added += 1
    for family, candidates in mixes(selected):
        if family in have:
            continue
        select_on_tune(task, family, candidates, documents, tune, test, rows, selected)
        added += 1
    if added:
        write_rows(task, rows)


# --- report -------------------------------------------------------------------------------

REPORT_FAMILIES = ['v7', 'v7_default', 'v7_pw0.0', 'v7_pw0.5', 'v7_pw1.0', 'bm25_char', 'tfidf_char',
                   'bm25_word', 'damerau_levenshtein', 'bm25_char>v7', 'v7>bm25_char', 'rrf(v7,bm25_char)',
                   'tfidf_char>v7', 'rrf(v7,tfidf_char)', 'mix(bm25_char,v7)', 'mix(tfidf_char,v7)']


def report() -> None:
    rows = []
    for path in sorted(RESULTS.glob('search_benchmark_*.csv')):
        if path.stem.endswith('summary'):
            continue
        rows.extend(ips.read_csv(path))
    test_rows = [r for r in rows if r['split'] == 'test']
    present = {r['task'] for r in test_rows}
    tasks = [t for t in TASKS if t in present] + sorted(present - set(TASKS))  # in regime order
    lines = ['# search benchmark summary', '',
             'generated by `python experiments/search_benchmark.py --report` from '
             '`experiments/results/search_benchmark_<task>.csv`. every number is on the *test* half '
             'of the queries for the configuration selected on the *tune* half, ranked lists cut at '
             f'top_k={TOP_K}. see the module docstring for the protocol.', '']
    for metric in ('map', 'ndcg@10', 'recall@50', 'ms_per_query'):
        lines += [f'## {metric}', '', '| task | queries | ' + ' | '.join(REPORT_FAMILIES) + ' |',
                  '|---|---:|' + '---:|' * len(REPORT_FAMILIES)]
        for task in tasks:
            by_family = {r['family']: r for r in test_rows if r['task'] == task}
            n_queries = next(iter(by_family.values()))['n_queries']
            cells = []
            for family in REPORT_FAMILIES:
                r = by_family.get(family)
                if r is None:
                    cells.append('')
                elif metric == 'ms_per_query':
                    cells.append(f'{float(r[metric]):.2f}')
                else:
                    cells.append(f'{float(r[metric]):.3f}')
            lines.append(f'| {task} | {n_queries} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## selected configurations', '', '| task | family | configuration |', '|---|---|---|']
    for task in tasks:
        for r in test_rows:
            if r['task'] == task and r['family'] in ('v7', 'bm25_char', 'tfidf_char', 'bm25_word',
                                                     'mix(bm25_char,v7)', 'mix(tfidf_char,v7)'):
                lines.append(f'| {task} | {r["family"]} | `{r["system"]}` |')
    path = RESULTS / 'search_benchmark_summary.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(f'wrote {path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('tasks', nargs='?', default='all')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--backfill', action='store_true',
                        help='add missing rerank/fusion rows to existing CSVs instead of re-running')
    parser.add_argument('--baselines', action='store_true',
                        help='keep the V7 rows of existing CSVs and re-run every other system')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.report:
        report()
        return
    tasks = list(TASKS) if args.tasks == 'all' else args.tasks.split(',')
    for task in tasks:
        if task not in TASKS:
            raise SystemExit(f'unknown task {task!r}; choose from {list(TASKS)}')
        exists = (RESULTS / f'search_benchmark_{task}.csv').exists()
        if args.backfill:
            if exists:
                backfill_combinations(task, args.seed)
        elif args.baselines:
            if exists:
                run_task(task, args.quick, args.seed, baselines_only=True)
        else:
            run_task(task, args.quick, args.seed)


if __name__ == '__main__':
    main()
