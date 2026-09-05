"""
compare nmd / segment mover's distance against classical string matchers on Abt-Buy product names

the published Abt-Buy numbers (DeepMatcher, Ditto, ...) are F1 on pair classification with a train /
test split and both the `name` and `description` fields, so they are not comparable to anything here.
this measures a retrieval task instead: given an Abt product name, rank the Buy names.

data: https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip  ->  unzip into .scratch/data/
run:  python experiments/baseline_eval.py [num_queries] [num_candidates]

part A: rank the whole Buy catalogue with every cheap metric (no prefilter, so nothing favours nmd)
part B: rerank the top-k from a neutral tf-idf char-3gram prefilter with every metric, including the
        expensive ones. same candidates for everybody.

all baselines are implemented here rather than pulled from a library, so the env stays as it is.
"""
import csv
import functools
import math
import random
import sys
import time
from collections import Counter
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import nmd.nmd_segments
from nmd.nmd_bow import bow_ngram_movers_distance
from nmd.nmd_core import ngram_movers_distance
from nmd.nmd_segments import segment_movers_distance
from experiments.v7_ranking import build_ranker

DATA = Path(__file__).resolve().parent.parent / '.scratch' / 'data'

nmd.nmd_segments._pair_raw = functools.lru_cache(maxsize=None)(nmd.nmd_segments._pair_raw)


def read_csv(path):
    for encoding in ('utf-8', 'latin-1'):
        try:
            with open(path, encoding=encoding, newline='') as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(path)


def tokenize(name):
    return [tok for tok in name.casefold().split() if any(c.isalnum() for c in tok)]


def char_ngrams(text, n=3):
    text = f' {text} '
    return [text[i:i + n] for i in range(max(0, len(text) - n + 1))]


# --------------------------------------------------------------------------------------- jaro-winkler

@functools.lru_cache(maxsize=1 << 20)
def jaro(s, t):
    if s == t:
        return 1.0
    ls, lt = len(s), len(t)
    if not ls or not lt:
        return 0.0
    window = max(0, max(ls, lt) // 2 - 1)
    s_hit, t_hit = [False] * ls, [False] * lt
    matches = 0
    for i, ch in enumerate(s):
        for j in range(max(0, i - window), min(i + window + 1, lt)):
            if not t_hit[j] and t[j] == ch:
                s_hit[i] = t_hit[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    transpositions, k = 0, 0
    for i in range(ls):
        if s_hit[i]:
            while not t_hit[k]:
                k += 1
            transpositions += s[i] != t[k]
            k += 1
    transpositions //= 2
    return (matches / ls + matches / lt + (matches - transpositions) / matches) / 3


def jaro_winkler(s, t, p=0.1, max_prefix=4):
    j = jaro(s, t)
    if j < 0.7:  # standard boost threshold
        return j
    prefix = 0
    for a, b in zip(s[:max_prefix], t[:max_prefix]):
        if a != b:
            break
        prefix += 1
    return j + prefix * p * (1 - j)


# --------------------------------------------------------------------------------------- set / vector similarity

def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def tfidf_vector(terms, idf):
    """l2-normalized (1 + log tf) * idf"""
    vec = {term: (1 + math.log(tf)) * idf.get(term, 0.0) for term, tf in Counter(terms).items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {term: v / norm for term, v in vec.items()} if norm else {}


def cosine(vec_a, vec_b):
    if len(vec_b) < len(vec_a):
        vec_a, vec_b = vec_b, vec_a
    return sum(v * vec_b.get(term, 0.0) for term, v in vec_a.items())


def monge_elkan(tokens_a, tokens_b):
    """symmetric monge-elkan with jaro-winkler (Monge & Elkan 1996)"""
    if not tokens_a or not tokens_b:
        return 0.0
    fwd = sum(max(jaro_winkler(a, b) for b in tokens_b) for a in tokens_a) / len(tokens_a)
    rev = sum(max(jaro_winkler(b, a) for a in tokens_a) for b in tokens_b) / len(tokens_b)
    return (fwd + rev) / 2


def soft_tfidf(vec_a, vec_b, theta=0.9):
    """SoftTF-IDF (Cohen, Ravikumar & Fienberg 2003) with jaro-winkler as the secondary measure"""
    if not vec_a or not vec_b:
        return 0.0
    total = 0.0
    for term_a, weight_a in vec_a.items():
        best_sim, best_weight = 0.0, 0.0
        for term_b, weight_b in vec_b.items():
            sim = 1.0 if term_a == term_b else jaro_winkler(term_a, term_b)
            if sim > best_sim:
                best_sim, best_weight = sim, weight_b
        if best_sim >= theta:
            total += weight_a * best_weight * best_sim
    return total


# --------------------------------------------------------------------------------------- protocol

def main(num_queries=100, num_candidates=20, seed=0):
    abt = {row['id']: row['name'] for row in read_csv(DATA / 'Abt.csv')}
    buy = {row['id']: row['name'] for row in read_csv(DATA / 'Buy.csv')}
    truth = defaultdict(set)
    for row in read_csv(DATA / 'abt_buy_perfectMapping.csv'):
        truth[row['idAbt']].add(row['idBuy'])

    # idf over both catalogues, documents = product names
    corpus = list(abt.values()) + list(buy.values())
    tok_df, chr_df = Counter(), Counter()
    for name in corpus:
        tok_df.update(set(tokenize(name)))
        chr_df.update(set(char_ngrams(name.casefold())))
    n_docs = len(corpus)
    tok_idf = {t: math.log(n_docs / df) for t, df in tok_df.items()}
    chr_idf = {g: math.log(n_docs / df) for g, df in chr_df.items()}

    def features(name):
        toks = tokenize(name)
        grams = char_ngrams(name.casefold())
        return {'name': name, 'toks': toks, 'joined': ' '.join(toks), 'tokset': set(toks),
                'gramset': set(grams), 'tokvec': tfidf_vector(toks, tok_idf), 'gramvec': tfidf_vector(grams, chr_idf)}

    buy_f = {bid: features(name) for bid, name in buy.items()}

    # every metric takes two feature dicts and returns a similarity (higher = better match)
    CHEAP = {
        'jaccard token': lambda a, b: jaccard(a['tokset'], b['tokset']),
        'jaccard char-3gram': lambda a, b: jaccard(a['gramset'], b['gramset']),
        'tf-idf cos token': lambda a, b: cosine(a['tokvec'], b['tokvec']),
        'tf-idf cos char-3gram': lambda a, b: cosine(a['gramvec'], b['gramvec']),
        'difflib ratio': lambda a, b: SequenceMatcher(None, a['joined'], b['joined']).ratio(),
        'monge-elkan (JW)': lambda a, b: monge_elkan(a['toks'], b['toks']),
        'soft tf-idf (JW .9)': lambda a, b: soft_tfidf(a['tokvec'], b['tokvec']),
        'char-nmd(2)': lambda a, b: ngram_movers_distance(a['joined'], b['joined'], n=2, invert=True, normalize=True),
    }
    EXPENSIVE = {
        'bow-nmd(2)': lambda a, b: bow_ngram_movers_distance(a['toks'], b['toks'], n=2, invert=True, normalize=True),
        'segment L3 lam0': lambda a, b: segment_movers_distance(a['toks'], b['toks'], max_len=3, lam=0.0,
                                                                invert=True, normalize=True),
        'segment L3 lam0 ms.2': lambda a, b: segment_movers_distance(a['toks'], b['toks'], max_len=3, lam=0.0,
                                                                     min_sim=0.2, invert=True, normalize=True),
    }

    rng = random.Random(seed)
    query_ids = rng.sample(sorted(truth), num_queries)
    queries = [(aid, features(abt[aid])) for aid in query_ids]
    buy_ids = sorted(buy)

    # ---------------- part A: rank the whole catalogue ----------------
    print('=' * 92)
    print(f'PART A: full-catalogue ranking, {num_queries} queries x {len(buy_ids)} Buy names, no prefilter')
    print('=' * 92)
    print(f'{"metric":<24} {"hit@1":>8} {"MRR":>8} {"recall@10":>10} {"recall@20":>10} {"sec":>7}')
    for name, metric in CHEAP.items():
        if name == 'char-nmd(2)':
            continue  # served by the index below instead of scoring 1092 names pairwise
        t0 = time.perf_counter()
        hits = rr = r10 = r20 = 0
        for aid, qf in queries:
            scored = sorted(((metric(qf, buy_f[bid]), bid) for bid in buy_ids), key=lambda x: -x[0])
            rank = next(i for i, (_, bid) in enumerate(scored, 1) if bid in truth[aid])
            hits += rank == 1
            rr += 1 / rank
            r10 += rank <= 10
            r20 += rank <= 20
        elapsed = time.perf_counter() - t0
        print(f'{name:<24} {hits / num_queries:>8.3f} {rr / num_queries:>8.3f} '
              f'{r10 / num_queries:>10.3f} {r20 / num_queries:>10.3f} {elapsed:>7.1f}')

    # char-nmd(2) through an ApproxWordListV7 index rather than 1092 pairwise calls per query.
    # exact, not approximate: V7's pruning bound is two-sided and lookup(dim=1, normalize=True)
    # is the same quantity ngram_movers_distance(n=2, invert=True, normalize=True) returns.
    # the index is keyed by string, so Buy rows sharing a normalized name collapse into one
    # document and a returned name counts as correct if any row carrying it is a true match
    text_to_ids = defaultdict(set)
    for bid in buy_ids:
        if buy_f[bid]['joined']:
            text_to_ids[buy_f[bid]['joined']].add(bid)
    t0 = time.perf_counter()
    rank_fn = build_ranker(sorted(text_to_ids), n=(2,))
    hits = rr = r10 = r20 = 0
    for aid, qf in queries:
        rank = None
        for i, (text, _score) in enumerate(rank_fn(qf['joined']), 1):
            if text_to_ids[text] & truth[aid]:
                rank = i
                break
        hits += rank == 1
        rr += 1 / rank if rank else 0.0
        r10 += bool(rank and rank <= 10)
        r20 += bool(rank and rank <= 20)
    elapsed = time.perf_counter() - t0
    print(f'{"char-nmd(2)":<24} {hits / num_queries:>8.3f} {rr / num_queries:>8.3f} '
          f'{r10 / num_queries:>10.3f} {r20 / num_queries:>10.3f} {elapsed:>7.1f}')

    # ---------------- part B: rerank a neutral prefilter ----------------
    prefilter = CHEAP['tf-idf cos char-3gram']
    candidates = {}
    for aid, qf in queries:
        top = sorted(buy_ids, key=lambda bid: -prefilter(qf, buy_f[bid]))[:num_candidates]
        if truth[aid] & set(top):
            candidates[aid] = top
    print()
    print('=' * 92)
    print(f'PART B: rerank top-{num_candidates} from a tf-idf char-3gram prefilter '
          f'(true match present for {len(candidates)}/{num_queries} queries)')
    print('=' * 92)
    print(f'{"metric":<24} {"hit@1":>8} {"MRR":>8} {"sec":>7}')
    for name, metric in list(CHEAP.items()) + list(EXPENSIVE.items()):
        t0 = time.perf_counter()
        hits = rr = 0
        for aid, qf in queries:
            if aid not in candidates:
                continue
            scored = sorted(((metric(qf, buy_f[bid]), bid) for bid in candidates[aid]), key=lambda x: -x[0])
            rank = next(i for i, (_, bid) in enumerate(scored, 1) if bid in truth[aid])
            hits += rank == 1
            rr += 1 / rank
        elapsed = time.perf_counter() - t0
        print(f'{name:<24} {hits / len(candidates):>8.3f} {rr / len(candidates):>8.3f} {elapsed:>7.1f}')


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:]))
