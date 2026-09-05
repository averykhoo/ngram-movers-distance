"""
grid search over `ApproxWordListV7`'s ranking parameters, scored by MAP and NDCG

this is the first eval script in this directory that drives an *index* rather than a pairwise
metric callable. every other one (baseline_eval, word_lookup_eval, segment_eval, benchmark_all,
hyperparam_search) scores `metric(a, b)` over every candidate, which measures the scoring
formula but never the index's own parameters.

the searchable surface is exactly four knobs, since `top_k` only sets the pruning threshold and
`invert` only changes how the score is reported:

    n              n-gram sizes to index          (constructor)
    idf_exponent   idf ** p as a per-type weight  (constructor)
    dim            power-mean exponent over n     (lookup)
    normalize      length normalization on/off    (lookup)

`case_sensitive` is held at the default False: the typo dictionary is already casefolded, and on
product names True would only fragment postings.

two retrieval tasks, because an index cannot be run on ER-Magellan's labelled pairs without
reframing them:

    typo     8000-word dictionary from experiments/words_en.txt, queries are corrupted words.
             exactly one right answer, so MAP degenerates to MRR and NDCG to a log-discounted
             reciprocal rank. still strictly more sensitive than the hit@1 Part 7 used.
    abtbuy   1092 Buy product names indexed, queries are Abt names. truth is a *set* of Buy ids
             per query, so MAP is genuinely multi-answer here -- though the mapping is mostly
             1:1 in practice, so do not expect it to diverge far from MRR either.

both samplers reproduce the existing scripts exactly (word_lookup_eval.main and
baseline_eval.main, both seed 0) so the numbers line up with what is already recorded.

run:  python experiments/index_param_search.py [typo|abtbuy|both] [--quick]

the abtbuy data lives in gitignored .scratch/data/ and is absent on a fresh clone; that task
skips itself with a message rather than failing.
"""
import ast
import csv
import itertools
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

# running this as `python experiments/index_param_search.py` puts experiments/ on sys.path but
# not the repo root, so `nmd` would not import
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranking_metrics import average_precision
from ranking_metrics import hit_at_k
from ranking_metrics import ndcg_at_k
from ranking_metrics import recall_at_k
from ranking_metrics import reciprocal_rank

from nmd.nmd_index_v7 import ApproxWordListV7

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / '.scratch' / 'data'
WORDS = ROOT / 'experiments' / 'words_en.txt'
RESULTS = ROOT / 'experiments' / 'results'

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

# how deep a ranked list to score. every metric below is a cutoff metric at this depth, so
# TOP_K is part of the result and changing it makes runs incomparable
TOP_K = 50

# n-gram size combinations. (1, 2) is here because docs/bow-plan.md part 7 measured it 11 points
# of hit@1 above the (2, 4) default on typo correction -- as a *scorer*. as an index key unigrams
# have almost no selectivity, so this sweep is also the test of whether that finding survives
# being run through real candidate generation
N_GRIDS = [(1,), (2,), (3,), (4,),
           (1, 2), (1, 3), (2, 3), (2, 4), (3, 4),
           (1, 2, 3), (1, 2, 4), (2, 3, 4), (1, 2, 3, 4)]

# 0.0 keeps the unweighted path bit-for-bit. part 6 found train AP still climbing at p=15 on
# product names, but part 7 found p=2 *losing* 8 points of hit@1 on typos, so the useful range
# is task-dependent and the grid has to cover both sides
IDF_GRID = [0.0, 0.5, 1.0, 2.0, 3.0]

# dim == 0 raises ZeroDivisionError (_generalized_mean computes 1.0 / dim), and dim < 0 changes
# the semantics rather than the weighting -- a word matching on only one n gets bound 0 and is
# dropped from the candidate set entirely, turning the index conjunctive. left out of the grid
# on purpose; it is a different retrieval model, not a tuning point
DIM_GRID = [0.5, 1.0, 2.0, 3.0]

NORMALIZE_GRID = [False, True]

# --- stage 2 -------------------------------------------------------------------------------
# the two knobs added after stage 1, both pure read-path so they need no index rebuild.
# 1.0 / 'dice' reproduce stage 1 exactly, which is what makes the overlap a regression check
POSITION_WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
DENOMINATOR_GRID = ['dice', 'geo']

# how many of stage 1's best configurations to carry forward, plus evenly spaced samples from
# the rest so the regression check covers weak configurations too, not just the winners
STAGE2_TOP_N = 8
STAGE2_SPREAD = 4


def corrupt(word: str, edits: int, rng: random.Random) -> str:
    """identical to word_lookup_eval.corrupt, so the query set matches that script's"""
    for _ in range(edits):
        if len(word) < 2:
            break
        op = rng.choice(['insert', 'delete', 'substitute', 'transpose'])
        i = rng.randrange(len(word))
        if op == 'insert':
            word = word[:i] + rng.choice(ALPHABET) + word[i:]
        elif op == 'delete':
            word = word[:i] + word[i + 1:]
        elif op == 'substitute':
            word = word[:i] + rng.choice(ALPHABET) + word[i + 1:]
        else:
            j = min(i + 1, len(word) - 1)
            chars = list(word)
            chars[i], chars[j] = chars[j], chars[i]
            word = ''.join(chars)
    return word


def load_typo(dict_size: int = 8000, num_queries: int = 250, seed: int = 0):
    """returns (documents, [(query, relevant_set), ...]) -- mirrors word_lookup_eval.main"""
    rng = random.Random(seed)
    with open(WORDS, encoding='utf8') as f:
        vocab = sorted({w.strip().casefold() for w in f
                        if w.strip().isalpha() and 3 <= len(w.strip()) <= 15})
    targets = rng.sample(vocab, num_queries)
    rest = [w for w in vocab if w not in set(targets)]
    documents = sorted(set(targets) | set(rng.sample(rest, max(0, dict_size - num_queries))))
    queries = [(corrupt(w, rng.choice([1, 1, 2]), rng), w) for w in targets]
    return documents, [(q, {w}) for q, w in queries if q != w]


def read_csv(path: Path) -> List[dict]:
    for encoding in ('utf-8', 'latin-1'):
        try:
            with open(path, encoding=encoding, newline='') as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(path)


def load_abtbuy(num_queries: int = 100, seed: int = 0):
    """
    returns (documents, [(query, relevant_set), ...]) -- mirrors baseline_eval.main part A

    the index is keyed by string, not by row id, so several Buy rows sharing a name collapse to
    one document. relevance is therefore resolved on the casefolded name: a returned name counts
    as correct if *any* row carrying it is a true match for that query
    """
    abt = {row['id']: row['name'] for row in read_csv(DATA / 'Abt.csv')}
    buy = {row['id']: row['name'] for row in read_csv(DATA / 'Buy.csv')}
    truth: Dict[str, Set[str]] = defaultdict(set)
    for row in read_csv(DATA / 'abt_buy_perfectMapping.csv'):
        truth[row['idAbt']].add(row['idBuy'])

    name_to_ids: Dict[str, Set[str]] = defaultdict(set)
    for bid, name in buy.items():
        key = clean(name)
        if key:
            name_to_ids[key].add(bid)
    documents = sorted(name_to_ids)

    rng = random.Random(seed)
    query_ids = rng.sample(sorted(truth), num_queries)
    queries = []
    for aid in query_ids:
        query = clean(abt[aid])
        relevant = {name for name, ids in name_to_ids.items() if ids & truth[aid]}
        if query and relevant:
            queries.append((query, relevant))
    return documents, queries


def clean(text: str) -> str:
    """casefold and strip the two control characters V7 reserves as START/END markers"""
    return text.replace('\2', ' ').replace('\3', ' ').strip().casefold()


def evaluate(index: ApproxWordListV7, queries, dim: float, normalize: bool,
             position_weight: float = 1.0, denominator: str = 'dice') -> Dict[str, float]:
    """run every query and average each metric over them"""
    totals = defaultdict(float)
    for query, relevant in queries:
        ranked = [word for word, _score in
                  index.lookup(query, top_k=TOP_K, dim=dim, normalize=normalize,
                               position_weight=position_weight, denominator=denominator)]
        totals['map'] += average_precision(ranked, relevant)
        totals['ndcg@10'] += ndcg_at_k(ranked, relevant, k=10)
        totals['mrr'] += reciprocal_rank(ranked, relevant)
        totals['hit@1'] += hit_at_k(ranked, relevant, k=1)
        totals['recall@10'] += recall_at_k(ranked, relevant, k=10)
    return {key: value / len(queries) for key, value in totals.items()}


def run_task(task: str, documents: List[str], queries, quick: bool) -> List[dict]:
    n_grids = N_GRIDS[:3] if quick else N_GRIDS
    idf_grid = IDF_GRID[:2] if quick else IDF_GRID
    dim_grid = DIM_GRID[:2] if quick else DIM_GRID

    # dim is a no-op for a single n (_generalized_mean short-circuits on one part), so sweeping
    # it there would re-measure the same configuration len(dim_grid) times
    builds = list(itertools.product(n_grids, idf_grid))
    total = sum(len(dim_grid if len(n) > 1 else [1.0]) * len(NORMALIZE_GRID) for n, _ in builds)
    print(f'\n=== {task}: {len(documents)} documents, {len(queries)} queries, '
          f'top_k={TOP_K} ===')
    print(f'{len(builds)} index builds, {total} configurations')
    print(f'{"n":<12} {"idf":>5} {"dim":>5} {"norm":>6} {"MAP":>7} {"NDCG@10":>8} '
          f'{"MRR":>7} {"hit@1":>7} {"rec@10":>7} {"sec":>7}', flush=True)

    rows = []
    for n, idf_exponent in builds:
        t_build = time.perf_counter()
        index = ApproxWordListV7(n=n, idf_exponent=idf_exponent).add_words(documents)
        index._freeze()  # pay the build cost here rather than inside the first timed lookup
        build_sec = time.perf_counter() - t_build

        for dim in (dim_grid if len(n) > 1 else [1.0]):
            for normalize in NORMALIZE_GRID:
                t0 = time.perf_counter()
                scores = evaluate(index, queries, dim, normalize)
                elapsed = time.perf_counter() - t0
                row = dict(task=task, n=repr(n), idf_exponent=idf_exponent, dim=dim,
                           normalize=normalize, build_sec=round(build_sec, 2),
                           eval_sec=round(elapsed, 2), **{k: round(v, 5) for k, v in scores.items()})
                rows.append(row)
                print(f'{repr(n):<12} {idf_exponent:>5} {dim:>5} {str(normalize):>6} '
                      f'{scores["map"]:>7.4f} {scores["ndcg@10"]:>8.4f} {scores["mrr"]:>7.4f} '
                      f'{scores["hit@1"]:>7.4f} {scores["recall@10"]:>7.4f} {elapsed:>7.1f}',
                      flush=True)
        del index
    return rows


def select_stage2_configs(rows: List[dict]) -> List[tuple]:
    """
    stage 1's best configurations, plus the library default and an even spread of the rest

    the spread matters for the regression check rather than for the search: a knob that only
    reproduced stage 1 on the strongest configurations would not have been checked anywhere
    the scores are small
    """
    ranked = sorted(rows, key=lambda r: -r['map'])
    chosen = ranked[:STAGE2_TOP_N]
    step = max(1, len(ranked) // (STAGE2_SPREAD + 1))
    chosen += [ranked[i] for i in range(step, len(ranked), step)][:STAGE2_SPREAD]
    chosen += [r for r in rows if r['n'] == repr((2, 4)) and r['idf_exponent'] == 0.0
               and r['dim'] == 1.0 and not r['normalize']]
    seen, out = set(), []
    for row in chosen:
        key = (row['n'], row['idf_exponent'], row['dim'], row['normalize'])
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def run_stage2(task: str, documents: List[str], queries, stage1_rows: List[dict]) -> List[dict]:
    """sweep the two new knobs over stage 1's selected configurations"""
    configs = select_stage2_configs(stage1_rows)
    # group by the constructor args so each index is built once
    by_build: Dict[tuple, List[tuple]] = defaultdict(list)
    for n_repr, idf_exponent, dim, normalize in configs:
        by_build[(n_repr, idf_exponent)].append((dim, normalize))

    total = len(configs) * len(POSITION_WEIGHT_GRID) * len(DENOMINATOR_GRID)
    print(f'\n=== {task} stage 2: {len(configs)} carried-forward configurations x '
          f'{len(POSITION_WEIGHT_GRID)} position_weight x {len(DENOMINATOR_GRID)} denominator '
          f'= {total} runs ===')
    print(f'{"n":<12} {"idf":>5} {"dim":>5} {"norm":>6} {"pos_w":>6} {"denom":>6} '
          f'{"MAP":>7} {"NDCG@10":>8} {"hit@1":>7} {"sec":>6}', flush=True)

    rows = []
    for (n_repr, idf_exponent), variants in sorted(by_build.items()):
        n = ast.literal_eval(n_repr)  # written by repr() in this same script
        index = ApproxWordListV7(n=n, idf_exponent=idf_exponent).add_words(documents)
        index._freeze()
        for dim, normalize in variants:
            for position_weight in POSITION_WEIGHT_GRID:
                for denom in DENOMINATOR_GRID:
                    # 'geo' only enters through the normalizing denominator
                    if denom == 'geo' and not normalize:
                        continue
                    t0 = time.perf_counter()
                    scores = evaluate(index, queries, dim, normalize, position_weight, denom)
                    elapsed = time.perf_counter() - t0
                    rows.append(dict(task=task, n=n_repr, idf_exponent=idf_exponent, dim=dim,
                                     normalize=normalize, position_weight=position_weight,
                                     denominator=denom, eval_sec=round(elapsed, 2),
                                     **{k: round(v, 5) for k, v in scores.items()}))
                    print(f'{n_repr:<12} {idf_exponent:>5} {dim:>5} {str(normalize):>6} '
                          f'{position_weight:>6} {denom:>6} {scores["map"]:>7.4f} '
                          f'{scores["ndcg@10"]:>8.4f} {scores["hit@1"]:>7.4f} {elapsed:>6.1f}',
                          flush=True)
        del index
    return rows


def check_regression(stage1_rows: List[dict], stage2_rows: List[dict]) -> bool:
    """
    every stage 2 run at position_weight=1.0 / denominator='dice' must reproduce stage 1 exactly

    this is the whole reason stage 2 re-runs configurations that were already measured. the two
    new knobs are supposed to be inert at their defaults, and an approximate match would not
    prove that -- summation order is unchanged, so the numbers must be equal to the last digit
    """
    index = {(r['n'], r['idf_exponent'], r['dim'], r['normalize']): r for r in stage1_rows}
    metrics = ('map', 'ndcg@10', 'mrr', 'hit@1', 'recall@10')
    checked = mismatched = 0
    for row in stage2_rows:
        if row['position_weight'] != 1.0 or row['denominator'] != 'dice':
            continue
        before = index.get((row['n'], row['idf_exponent'], row['dim'], row['normalize']))
        if before is None:
            continue
        checked += 1
        diffs = [(m, before[m], row[m]) for m in metrics if before[m] != row[m]]
        if diffs:
            mismatched += 1
            print(f'  REGRESSION {row["n"]} idf={row["idf_exponent"]} dim={row["dim"]} '
                  f'normalize={row["normalize"]}: {diffs}')
    print(f'\nregression check: {checked - mismatched}/{checked} carried-forward configurations '
          f'reproduce stage 1 exactly at position_weight=1.0, denominator=dice')
    return mismatched == 0


def write_results(task: str, rows: List[dict], suffix: str = '') -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f'index_param_search_{task}{suffix}.csv'
    # stage 1 rows carry build_sec and stage 2 rows do not, so a merged write needs the union
    # of keys rather than whichever row happens to be first
    fieldnames: List[str] = []
    for row in rows:
        fieldnames += [key for key in row if key not in fieldnames]
    with open(path, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_results(task: str, suffix: str = '') -> List[dict]:
    """read a results csv back, restoring the types csv flattened to strings"""
    path = RESULTS / f'index_param_search_{task}{suffix}.csv'
    rows = []
    for row in read_csv(path):
        row['normalize'] = row['normalize'] == 'True'
        for key in ('idf_exponent', 'dim', 'position_weight', 'build_sec', 'eval_sec',
                    'map', 'ndcg@10', 'mrr', 'hit@1', 'recall@10'):
            if row.get(key) not in (None, ''):
                row[key] = float(row[key])
        # stage 1 predates both new knobs, and ran at exactly their defaults
        row.setdefault('position_weight', 1.0)
        row.setdefault('denominator', 'dice')
        rows.append(row)
    return rows


def summarize(task: str, rows: List[dict], label: str = '') -> None:
    print(f'\n--- {task}{label}: top 10 by MAP ---')
    print(f'{"n":<12} {"idf":>5} {"dim":>5} {"norm":>6} {"pos_w":>6} {"denom":>6} '
          f'{"MAP":>7} {"NDCG@10":>8} {"hit@1":>7}')
    for row in sorted(rows, key=lambda r: -r['map'])[:10]:
        print(f'{row["n"]:<12} {row["idf_exponent"]:>5} {row["dim"]:>5} {str(row["normalize"]):>6} '
              f'{row.get("position_weight", 1.0):>6} {row.get("denominator", "dice"):>6} '
              f'{row["map"]:>7.4f} {row["ndcg@10"]:>8.4f} {row["hit@1"]:>7.4f}')
    baseline = [r for r in rows if r['n'] == repr((2, 4)) and r['idf_exponent'] == 0.0
                and r['dim'] == 1.0 and not r['normalize']
                and r.get('position_weight', 1.0) == 1.0 and r.get('denominator', 'dice') == 'dice']
    if baseline:
        row = baseline[0]
        print(f'\nlibrary default n=(2, 4) idf=0 dim=1 normalize=False: '
              f'MAP {row["map"]:.4f}  NDCG@10 {row["ndcg@10"]:.4f}  hit@1 {row["hit@1"]:.4f}')


def load_task(task: str):
    if task == 'typo':
        return load_typo()
    if not (DATA / 'Abt.csv').exists():
        print(f'\n=== abtbuy: skipped, {DATA / "Abt.csv"} not found ===')
        print('download https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip '
              'and unzip into .scratch/data/')
        return None
    return load_abtbuy()


def main(which: str = 'both', quick: bool = False, stage2: bool = False) -> None:
    tasks = ['typo', 'abtbuy'] if which == 'both' else [which]
    for task in tasks:
        loaded = load_task(task)
        if loaded is None:
            continue
        documents, queries = loaded
        if not stage2:
            rows = run_task(task, documents, queries, quick)
            summarize(task, rows)
            print(f'\nwrote {write_results(task, rows)}')
            continue

        stage1_rows = read_results(task)
        rows = run_stage2(task, documents, queries, stage1_rows)
        clean_run = check_regression(stage1_rows, rows)
        print(f'\nwrote {write_results(task, rows, suffix="_stage2")}')
        if not clean_run:
            print('NOT merging: the new knobs are not inert at their defaults, so stage 1 and '
                  'stage 2 numbers are not comparable and a combined ranking would be wrong')
            continue
        # stage 2 re-measured some stage 1 cells; keep one row per full configuration
        merged = {(r['n'], r['idf_exponent'], r['dim'], r['normalize'],
                   r.get('position_weight', 1.0), r.get('denominator', 'dice')): r
                  for r in stage1_rows + rows}
        summarize(task, list(merged.values()), label=' (stage 1 + stage 2)')
        print(f'\nwrote {write_results(task, list(merged.values()), suffix="_merged")}')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    main(args[0] if args else 'both', quick='--quick' in sys.argv, stage2='--stage2' in sys.argv)
