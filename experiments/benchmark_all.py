"""
every metric, every dataset, one ranking

three tasks that stress different things:

  A. product-name retrieval   -- rank 1092 Buy names given an Abt name (hit@1 / MRR)
  B. product-name matching    -- the published ER-Magellan Abt-Buy pair split (test F1)
  C. typo correction          -- rank an 8000-word english dictionary given a corrupted word

A and B are the same domain in two protocols, which is worth keeping separate because they
disagree about the idf exponent. C is the task nmd was designed for and reverses most of what
A and B conclude.

data: .scratch/data/{Abt,Buy,abt_buy_perfectMapping}.csv, .scratch/data/er_magellan/*.txt,
      experiments/words_en.txt
run:  python experiments/benchmark_all.py
"""
import math
import random
import sys
import time
from collections import Counter
from collections import defaultdict
from difflib import SequenceMatcher

from experiments.baseline_eval import cosine
from experiments.baseline_eval import jaccard
from experiments.baseline_eval import monge_elkan
from experiments.baseline_eval import read_csv
from experiments.baseline_eval import soft_tfidf
from experiments.baseline_eval import tfidf_vector
from experiments.baseline_eval import tokenize
from experiments.er_magellan_eval import DATA as ER_DATA
from experiments.er_magellan_eval import best_threshold
from experiments.er_magellan_eval import f1_at
from experiments.er_magellan_eval import read_split
from experiments.idf_variants import grams
from experiments.idf_variants import weighted_nmd
from nmd.nmd_bow import bow_ngram_movers_distance
from nmd.nmd_core import ngram_movers_distance
from nmd.nmd_segments import segment_movers_distance

ABT_DATA = ER_DATA.parent
WORDS_FILE = ER_DATA.parent.parent.parent / 'experiments' / 'words_en.txt'


def build_idf(corpus, n):
    df = Counter()
    for text in corpus:
        df.update(set(grams(text, n)))
    size = len(corpus)
    table = {g: math.log((size + 1) / d) for g, d in df.items()}
    ceiling = math.log(size + 1)
    return lambda g, p: table.get(g, ceiling) ** p


def nmd_multi(a, b, n_list):
    return sum(ngram_movers_distance(a, b, n=n, invert=True, normalize=True) for n in n_list) / len(n_list)


def make_metrics(corpus, include_token_metrics=True):
    """
    every metric as a similarity over two raw strings, higher is better

    idf is fitted on whichever corpus is being searched, which is what an index would have.
    'pair' metrics take strings; the token-sequence ones split on whitespace, so on a
    single-word task they degenerate to plain nmd and are skipped.
    """
    idf2 = build_idf(corpus, 2)
    idf3 = build_idf(corpus, 3)
    tok_df = Counter()
    for text in corpus:
        tok_df.update(set(text.split()))
    tok_idf = {t: math.log((len(corpus) + 1) / d) for t, d in tok_df.items()}

    def tokvec(text):
        return tfidf_vector(text.split(), tok_idf)

    def gramvec(text, n, idf, p=1.0):
        counts = Counter(grams(text, n))
        vec = {g: (1 + math.log(c)) * idf(g, p) for g, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {g: v / norm for g, v in vec.items()} if norm else {}

    metrics = {
        # --- classical, order-blind ---
        'jaccard char-2gram': lambda a, b: jaccard(set(grams(a, 2)), set(grams(b, 2))),
        'jaccard char-3gram': lambda a, b: jaccard(set(grams(a, 3)), set(grams(b, 3))),
        'tf-idf cos token': lambda a, b: cosine(tokvec(a), tokvec(b)),
        'tf-idf cos char-2gram': lambda a, b: cosine(gramvec(a, 2, idf2), gramvec(b, 2, idf2)),
        'tf-idf cos char-3gram': lambda a, b: cosine(gramvec(a, 3, idf3), gramvec(b, 3, idf3)),
        'tf-idf cos char-3gram idf^2': lambda a, b: cosine(gramvec(a, 3, idf3, 2.0), gramvec(b, 3, idf3, 2.0)),
        'soft tf-idf (JW .9)': lambda a, b: soft_tfidf(tokvec(a), tokvec(b)),
        'monge-elkan (JW)': lambda a, b: monge_elkan(a.split(), b.split()),
        'difflib ratio': lambda a, b: SequenceMatcher(None, a, b).ratio(),
        # --- nmd family, unweighted ---
        'nmd n=1': lambda a, b: nmd_multi(a, b, (1,)),
        'nmd n=2': lambda a, b: nmd_multi(a, b, (2,)),
        'nmd n=3': lambda a, b: nmd_multi(a, b, (3,)),
        'nmd n=(1,2)': lambda a, b: nmd_multi(a, b, (1, 2)),
        'nmd n=(2,4)': lambda a, b: nmd_multi(a, b, (2, 4)),
        # --- nmd family, idf-weighted ---
        'nmd n=2 idf^1': lambda a, b: weighted_nmd(a, b, 2, lambda g: idf2(g, 1.0), 'geo'),
        'nmd n=2 idf^2': lambda a, b: weighted_nmd(a, b, 2, lambda g: idf2(g, 2.0), 'geo'),
        'nmd n=3 idf^2': lambda a, b: weighted_nmd(a, b, 3, lambda g: idf3(g, 2.0), 'geo'),
    }
    if include_token_metrics:
        metrics['bow-nmd n=2'] = lambda a, b: bow_ngram_movers_distance(
            a.split(), b.split(), n=2, invert=True, normalize=True)
        metrics['segment L3 lam0'] = lambda a, b: segment_movers_distance(
            a.split(), b.split(), max_len=3, lam=0.0, min_sim=0.2, invert=True, normalize=True)
    return metrics


CHEAP = {'jaccard char-2gram', 'jaccard char-3gram', 'tf-idf cos token', 'tf-idf cos char-2gram',
         'tf-idf cos char-3gram', 'tf-idf cos char-3gram idf^2', 'soft tf-idf (JW .9)',
         'monge-elkan (JW)', 'difflib ratio', 'nmd n=1', 'nmd n=2', 'nmd n=3', 'nmd n=(1,2)',
         'nmd n=(2,4)', 'nmd n=2 idf^1', 'nmd n=2 idf^2', 'nmd n=3 idf^2'}


def rank_table(title, rows, headers):
    print()
    print('=' * 96)
    print(title)
    print('=' * 96)
    print(f'{"#":>3} {"metric":<30} ' + ' '.join(f'{h:>9}' for h in headers))
    for i, (name, values) in enumerate(rows, 1):
        print(f'{i:>3} {name:<30} ' + ' '.join(f'{v:>9.3f}' for v in values))


# ------------------------------------------------------------------ task A: product retrieval

def task_a(num_queries=100):
    abt = {r['id']: r['name'] for r in read_csv(ABT_DATA / 'Abt.csv')}
    buy = {r['id']: r['name'] for r in read_csv(ABT_DATA / 'Buy.csv')}
    truth = defaultdict(set)
    for r in read_csv(ABT_DATA / 'abt_buy_perfectMapping.csv'):
        truth[r['idAbt']].add(r['idBuy'])
    norm = lambda name: ' '.join(tokenize(name))
    buy_norm = {bid: norm(name) for bid, name in buy.items()}
    corpus = [norm(name) for name in abt.values()] + list(buy_norm.values())
    metrics = make_metrics(corpus)
    queries = random.Random(0).sample(sorted(truth), num_queries)
    buy_ids = sorted(buy)

    results = []
    for name, metric in metrics.items():
        if name not in CHEAP:
            continue
        t0 = time.perf_counter()
        hits = rr = 0
        for aid in queries:
            query = norm(abt[aid])
            scored = sorted(((metric(query, buy_norm[bid]), bid) for bid in buy_ids), key=lambda x: -x[0])
            rank = next(i for i, (_, bid) in enumerate(scored, 1) if bid in truth[aid])
            hits += rank == 1
            rr += 1 / rank
        results.append((name, (hits / num_queries, rr / num_queries, time.perf_counter() - t0)))
        print(f'  A {name:<30} {hits / num_queries:.3f}', file=sys.stderr)
    results.sort(key=lambda kv: -kv[1][0])
    rank_table(f'TASK A -- product-name retrieval, {num_queries} Abt queries x {len(buy_ids)} Buy names',
               results, ['hit@1', 'MRR', 'sec'])

    # expensive metrics rerank the top 20 of a neutral prefilter
    prefilter = metrics['tf-idf cos char-3gram']
    candidates = {}
    for aid in queries:
        query = norm(abt[aid])
        top = sorted(buy_ids, key=lambda bid: -prefilter(query, buy_norm[bid]))[:20]
        if truth[aid] & set(top):
            candidates[aid] = top
    rerank = []
    for name, metric in metrics.items():
        t0 = time.perf_counter()
        hits = rr = 0
        for aid, top in candidates.items():
            query = norm(abt[aid])
            scored = sorted(((metric(query, buy_norm[bid]), bid) for bid in top), key=lambda x: -x[0])
            rank = next(i for i, (_, bid) in enumerate(scored, 1) if bid in truth[aid])
            hits += rank == 1
            rr += 1 / rank
        rerank.append((name, (hits / len(candidates), rr / len(candidates), time.perf_counter() - t0)))
    rerank.sort(key=lambda kv: -kv[1][0])
    rank_table(f'TASK A2 -- rerank top-20 of a tf-idf char-3gram prefilter '
               f'(true match present for {len(candidates)}/{num_queries})', rerank, ['hit@1', 'MRR', 'sec'])
    return dict(results), dict(rerank)


# ------------------------------------------------------------------ task B: ER-Magellan F1

def task_b():
    train = read_split(ER_DATA / 'train.txt', use_desc=False)
    test = read_split(ER_DATA / 'test.txt', use_desc=False)
    norm = lambda t: ' '.join(t.casefold().split())
    corpus = [norm(r[i]) for r in train + test for i in (0, 1)]
    metrics = make_metrics(corpus)

    results = []
    for name, metric in metrics.items():
        t0 = time.perf_counter()
        tr = [metric(norm(a), norm(b)) for a, b, _ in train]
        te = [metric(norm(a), norm(b)) for a, b, _ in test]
        thresh, _ = best_threshold(tr, [y for _, _, y in train])
        f1, precision, recall = f1_at(te, [y for _, _, y in test], thresh)
        results.append((name, (100 * f1, 100 * precision, 100 * recall, time.perf_counter() - t0)))
        print(f'  B {name:<30} {100 * f1:.2f}', file=sys.stderr)
    results.sort(key=lambda kv: -kv[1][0])
    rank_table(f'TASK B -- ER-Magellan Abt-Buy, {len(train)} train / {len(test)} test pairs, '
               f'one tuned threshold, name only', results, ['test F1', 'prec', 'rec', 'sec'])
    print('  published on this split (supervised, all attributes): '
          'Magellan 43.6   DeepMatcher+ 62.8   Ditto 89.33')
    return dict(results)


# ------------------------------------------------------------------ task C: typo correction

def task_c(dict_size=8000, num_queries=250, seed=0):
    rng = random.Random(seed)
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    with open(WORDS_FILE, encoding='utf8') as f:
        vocab = sorted({w.strip().casefold() for w in f if w.strip().isalpha() and 3 <= len(w.strip()) <= 15})
    targets = rng.sample(vocab, num_queries)
    rest = [w for w in vocab if w not in set(targets)]
    dictionary = sorted(set(targets) | set(rng.sample(rest, max(0, dict_size - num_queries))))

    def corrupt(word, edits):
        for _ in range(edits):
            if len(word) < 2:
                break
            op = rng.choice(['insert', 'delete', 'substitute', 'transpose'])
            i = rng.randrange(len(word))
            if op == 'insert':
                word = word[:i] + rng.choice(alphabet) + word[i:]
            elif op == 'delete':
                word = word[:i] + word[i + 1:]
            elif op == 'substitute':
                word = word[:i] + rng.choice(alphabet) + word[i + 1:]
            else:
                j = min(i + 1, len(word) - 1)
                chars = list(word)
                chars[i], chars[j] = chars[j], chars[i]
                word = ''.join(chars)
        return word

    queries = [(corrupt(w, rng.choice([1, 1, 2])), w) for w in targets]
    queries = [(q, a) for q, a in queries if q != a]
    metrics = make_metrics(dictionary, include_token_metrics=False)

    results = []
    for name, metric in metrics.items():
        t0 = time.perf_counter()
        hits = rr = 0
        for query, answer in queries:
            scored = sorted(((metric(query, word), word) for word in dictionary), key=lambda x: -x[0])
            rank = next(i for i, (_, word) in enumerate(scored, 1) if word == answer)
            hits += rank == 1
            rr += 1 / rank
        results.append((name, (hits / len(queries), rr / len(queries), time.perf_counter() - t0)))
        print(f'  C {name:<30} {hits / len(queries):.3f}', file=sys.stderr)
    results.sort(key=lambda kv: -kv[1][0])
    rank_table(f'TASK C -- typo correction, {len(queries)} corrupted queries x {len(dictionary)} words',
               results, ['hit@1', 'MRR', 'sec'])
    return dict(results)


def main(*args):
    a_full, a_rerank = task_a()
    b = task_b()
    c = task_c()

    # consolidated: where each metric places on each task
    print()
    print('=' * 96)
    print('CONSOLIDATED -- rank of each metric within each task (1 = best)')
    print('=' * 96)
    def ranks(table, key=0, reverse=True):
        order = sorted(table.items(), key=lambda kv: -kv[1][key] if reverse else kv[1][key])
        return {name: i for i, (name, _) in enumerate(order, 1)}
    ra, rb, rc = ranks(a_full), ranks(b), ranks(c)
    every = [n for n in a_full if n in rb and n in rc]
    print(f'{"metric":<30} {"A retrieval":>12} {"B match F1":>12} {"C typos":>10} {"worst":>7}')
    for name in sorted(every, key=lambda n: max(ra[n], rb[n], rc[n])):
        print(f'{name:<30} {ra[name]:>12} {rb[name]:>12} {rc[name]:>10} {max(ra[name], rb[name], rc[name]):>7}')


if __name__ == '__main__':
    main(*sys.argv[1:])
