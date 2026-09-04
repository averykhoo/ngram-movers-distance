"""
introspect and evaluate nmd.nmd_segments on the Abt-Buy product name matching benchmark

data: https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip  ->  unzip into .scratch/data/
run:  python experiments/segment_eval.py [num_queries] [num_candidates]

two parts:
  1. print alignments for a sample of true matches, so the matching decisions can be read
  2. ranking: for each sampled Abt name, prefilter Buy names by char-level nmd, then rerank the
     candidates with each metric and report hit@1 / MRR of the true match
"""
import csv
import functools
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import nmd.nmd_segments
from nmd.nmd_bow import bow_ngram_movers_distance
from nmd.nmd_core import ngram_movers_distance
from nmd.nmd_segments import align_segments
from nmd.nmd_segments import explain_alignment
from nmd.nmd_segments import segment_movers_distance

DATA = Path(__file__).resolve().parent.parent / '.scratch' / 'data'

# the per-pair similarity does not depend on lam / mu / max_len, so share it across configs
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


METRICS = {
    'char-nmd(2)': lambda a, b: ngram_movers_distance(' '.join(a), ' '.join(b), n=2, invert=True, normalize=True),
    'bow(2)': lambda a, b: bow_ngram_movers_distance(a, b, n=2, invert=True, normalize=True),
    'tok L1 lam0': lambda a, b: seg(a, b, max_len=1, lam=0.0, mass='tokens'),
    'tok L3 lam0.5': lambda a, b: seg(a, b, max_len=3, lam=0.5, mass='tokens'),
    'ngr L1 lam0': lambda a, b: seg(a, b, max_len=1, lam=0.0),
    'ngr L1 lam0.5': lambda a, b: seg(a, b, max_len=1, lam=0.5),
    'ngr L3 lam0': lambda a, b: seg(a, b, max_len=3, lam=0.0),
    'ngr L3 lam0.5': lambda a, b: seg(a, b, max_len=3, lam=0.5),
    'ngr L3 lam1': lambda a, b: seg(a, b, max_len=3, lam=1.0),
    'ngr L3 lam0.5 mu0.5': lambda a, b: seg(a, b, max_len=3, lam=0.5, mu=0.5),
    'ngr L3 lam0.5 n2': lambda a, b: seg(a, b, n=2, max_len=3, lam=0.5),
    'ngr L3 lam0 minsim0.2': lambda a, b: seg(a, b, max_len=3, lam=0.0, min_sim=0.2),
    'ngr L3 lam0.5 minsim0.2': lambda a, b: seg(a, b, max_len=3, lam=0.5, min_sim=0.2),
}
INSPECT = 'ngr L3 lam0 minsim0.2'


def seg(a, b, **kwargs):
    return segment_movers_distance(a, b, invert=True, normalize=True, **kwargs)


def main(num_queries=60, num_candidates=20, seed=0):
    abt = {row['id']: row['name'] for row in read_csv(DATA / 'Abt.csv')}
    buy = {row['id']: row['name'] for row in read_csv(DATA / 'Buy.csv')}
    truth = defaultdict(set)
    for row in read_csv(DATA / 'abt_buy_perfectMapping.csv'):
        truth[row['idAbt']].add(row['idBuy'])
    buy_tokens = {bid: tokenize(name) for bid, name in buy.items()}

    rng = random.Random(seed)
    query_ids = rng.sample(sorted(truth), num_queries)

    # --- part 1: read the alignments ---
    print('=' * 100)
    print(f'ALIGNMENTS OF TRUE MATCHES  ({INSPECT})')
    print('=' * 100)
    for aid in query_ids[:15]:
        for bid in truth[aid]:
            ta, tb = tokenize(abt[aid]), buy_tokens[bid]
            print(f'\nABT: {abt[aid]}\nBUY: {buy[bid]}')
            print(explain_alignment(ta, tb, align_segments(ta, tb, max_len=3, lam=0.0, min_sim=0.2)))

    # --- part 2: ranking ---
    print('\n' + '=' * 100)
    print(f'RANKING  ({num_queries} queries, prefilter top-{num_candidates} by char-nmd over {len(buy)} Buy names)')
    print('=' * 100)
    hits = {name: 0 for name in METRICS}
    rr = {name: 0.0 for name in METRICS}
    timings = {name: 0.0 for name in METRICS}
    in_candidates = 0
    misses = []
    for qi, aid in enumerate(query_ids):
        ta = tokenize(abt[aid])
        prefilter = sorted(buy, key=lambda bid: -METRICS['char-nmd(2)'](ta, buy_tokens[bid]))[:num_candidates]
        if not truth[aid] & set(prefilter):
            continue
        in_candidates += 1
        for name, metric in METRICS.items():
            t0 = time.perf_counter()
            scored = sorted(((metric(ta, buy_tokens[bid]), bid) for bid in prefilter), reverse=True)
            timings[name] += time.perf_counter() - t0
            rank = next(i for i, (_, bid) in enumerate(scored, 1) if bid in truth[aid])
            hits[name] += rank == 1
            rr[name] += 1 / rank
            if name == INSPECT and rank != 1:
                misses.append((abt[aid], buy[scored[0][1]], scored[0][0], buy[next(b for b in prefilter if b in truth[aid])],
                               next(s for s, b in scored if b in truth[aid]), rank))
        print(f'  {qi + 1}/{num_queries}', end='\r', file=sys.stderr)

    print(f'true match inside prefilter candidates: {in_candidates}/{num_queries}')
    print(f'{"metric":<24} {"hit@1":>8} {"MRR":>8} {"sec":>8}')
    for name in METRICS:
        print(f'{name:<24} {hits[name] / in_candidates:>8.3f} {rr[name] / in_candidates:>8.3f} {timings[name]:>8.1f}')

    print('\n' + '=' * 100)
    print(f'MISSES  ({INSPECT} ranked something else first)')
    print('=' * 100)
    for query, top1, top1_score, true, true_score, rank in misses:
        print(f'\nQUERY: {query}\n  top1 ({top1_score:.3f}): {top1}\n  true ({true_score:.3f}, rank {rank}): {true}')
        ta, tb = tokenize(query), tokenize(top1)
        print('  ' + explain_alignment(ta, tb, align_segments(ta, tb, max_len=3, lam=0.0, min_sim=0.2)).replace('\n', '\n  '))


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:]))
