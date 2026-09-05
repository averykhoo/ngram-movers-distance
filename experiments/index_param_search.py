"""
grid search over `ApproxWordListV7`'s ranking parameters, scored by MAP and NDCG

this is the first eval script in this directory that drives an *index* rather than a pairwise
metric callable. every other one (baseline_eval, word_lookup_eval, segment_eval, benchmark_all,
hyperparam_search) scores `metric(a, b)` over every candidate, which measures the scoring
formula but never the index's own parameters.

the searchable surface is six knobs, since `top_k` only sets the pruning threshold and `invert`
only changes how the score is reported:

    n                n-gram sizes to index                    (constructor)
    idf_exponent     idf ** p as a per-type weight            (constructor)
    dim              power-mean exponent over n               (lookup)
    normalize        length normalization on/off              (lookup)
    position_weight  how much displacement to charge for      (lookup)
    denominator      dice or geo length normalization         (lookup)

`case_sensitive` is held at the default False: the typo dictionaries are already casefolded, and
on product names True would only fragment postings.

five retrieval tasks. an index cannot consume ER-Magellan's labelled pairs directly, so that one
is reframed into retrieval (see load_ermagellan):

    typo         8000 words from words_en.txt, queries are words corrupted by 1-2 edits
    typo_ms      the same protocol on words_ms.txt -- tracked in the repo, read by no other eval
                 script, and the only non-English task here. tests whether the best parameters
                 are a property of the metric or of English
    typo_hard    30000 words from british-english-insane.txt with 2-3 edits. denser dictionary
                 and a bigger edit budget, so near-misses are genuinely ambiguous
    abtbuy       1092 Buy product names indexed, queries are Abt names, truth is a set of Buy ids
    ermagellan   the long-document task: entities are `COL name VAL ... COL description VAL ...`
                 averaging ~143 characters against ~20 for a product name and ~8 for a dictionary
                 word. nothing else here measures that regime

the three typo tasks each have exactly one right answer, so on them MAP is numerically identical
to MRR and NDCG is a log-discounted reciprocal rank. they are still more sensitive than hit@1.
abtbuy and ermagellan have set-valued truth, though both are close to 1:1 in practice, so do not
expect MAP and MRR to diverge far anywhere in this file.

the samplers for `typo` and `abtbuy` reproduce word_lookup_eval.main and baseline_eval.main
exactly (both seed 0), so those numbers line up with what is already recorded.

run:
    python experiments/index_param_search.py [task[,task...]|both|all] [--quick]
    python experiments/index_param_search.py <task> --stage2      # sweep the two newer knobs
    python experiments/index_param_search.py all --unified        # one grid on every task
    python experiments/index_param_search.py all --aggregate      # rank across tasks

--unified is what a cross-task answer needs: the same grid on every benchmark, so configurations
can be ranked against each other. the staged modes exist because stage 1 predates
position_weight and denominator, and re-running its whole grid to add them would have been
wasteful -- stage 2 carries forward stage 1's best and re-measures a spread of the rest as a
regression check.

abtbuy and ermagellan live in gitignored .scratch/data/ and are absent on a fresh clone; those
tasks skip themselves with a message rather than failing. the three typo tasks are self-contained.
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
from typing import Sequence
from typing import Set
from typing import Tuple

# running this as `python experiments/index_param_search.py` puts experiments/ on sys.path but
# not the repo root, so `nmd` would not import
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranking_metrics import average_precision
from ranking_metrics import hit_at_k
from ranking_metrics import ndcg_at_k
from ranking_metrics import precision_at_k
from ranking_metrics import r_precision
from ranking_metrics import recall_at_k
from ranking_metrics import reciprocal_rank

from nmd.nmd_index_v7 import ApproxWordListV7

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / '.scratch' / 'data'
WORDS = ROOT / 'experiments' / 'words_en.txt'
WORDS_MS = ROOT / 'experiments' / 'words_ms.txt'
WORDS_INSANE = ROOT / 'experiments' / 'british-english-insane.txt'
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


def load_typo(path: Path = WORDS, dict_size: int = 8000, num_queries: int = 250,
              seed: int = 0, edits: Sequence[int] = (1, 1, 2)):
    """
    returns (documents, [(query, relevant_set), ...]) -- mirrors word_lookup_eval.main

    `edits` is sampled per query, so the default (1, 1, 2) means "one edit two times out of
    three". the same filter as word_lookup_eval is applied to whatever word list is passed, so
    the three typo tasks differ only in their vocabulary and edit budget
    """
    rng = random.Random(seed)
    with open(path, encoding='utf8') as f:
        vocab = sorted({w.strip().casefold() for w in f
                        if w.strip().isalpha() and 3 <= len(w.strip()) <= 15})
    targets = rng.sample(vocab, num_queries)
    rest = [w for w in vocab if w not in set(targets)]
    documents = sorted(set(targets) | set(rng.sample(rest, max(0, dict_size - num_queries))))
    queries = [(corrupt(w, rng.choice(list(edits)), rng), w) for w in targets]
    return documents, [(q, {w}) for q, w in queries if q != w]


def load_ermagellan(seed: int = 0):
    """
    ER-Magellan Abt-Buy reframed from labelled pairs into retrieval

    the split ships as (left, right, 0/1) rows, which no index can consume directly. the right
    entities of every split become the document corpus, and each left entity that has a positive
    becomes a query whose relevant set is its positives. pooling all three splits for the corpus
    (but drawing queries only from valid.txt, which no other script in this directory reads) puts
    real distractors in the index rather than only the 693 right entities valid.txt happens to
    mention.

    this is the only long-document task here: entities are `COL name VAL ... COL description
    VAL ...`, averaging ~143 characters against ~20 for a product name and ~8 for a dictionary
    word, so it is the one place n-gram counts per document get large
    """
    rights, pairs = set(), []
    for split in ('train', 'valid', 'test'):
        path = DATA / 'er_magellan' / f'{split}.txt'
        for line in path.read_text(encoding='utf8').splitlines():
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            left, right, label = parts
            rights.add(clean(right))
            if split == 'valid' and label == '1':
                pairs.append((clean(left), clean(right)))

    truth: Dict[str, Set[str]] = defaultdict(set)
    for left, right in pairs:
        truth[left].add(right)
    documents = sorted(r for r in rights if r)
    queries = [(query, relevant) for query, relevant in sorted(truth.items()) if query and relevant]
    random.Random(seed).shuffle(queries)
    return documents, queries


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
        totals['ndcg@5'] += ndcg_at_k(ranked, relevant, k=5)
        totals['mrr'] += reciprocal_rank(ranked, relevant)
        totals['hit@1'] += hit_at_k(ranked, relevant, k=1)
        totals['hit@10'] += hit_at_k(ranked, relevant, k=10)
        totals['recall@10'] += recall_at_k(ranked, relevant, k=10)
        totals['p@10'] += precision_at_k(ranked, relevant, k=10)
        totals['rprec'] += r_precision(ranked, relevant)
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


# --- unified sweep -------------------------------------------------------------------------
# one grid run identically on every benchmark, so configurations can be ranked across tasks.
# the two-stage split was right for adding knobs to an existing sweep, but a "best overall"
# answer needs every task measured over the same points, not a union of differently-shaped grids.
# trimmed from stage 1's grid using what stage 1 and 2 showed: dim beyond 2 never won, idf above
# 2 collapsed on typo, and the n-lists dropped here were dominated on both tasks
UNIFIED_N = [(1,), (2,), (1, 2), (2, 3), (2, 4)]
# stage 1 put abtbuy's optimum at the *edge* of its grid (best MAP climbed monotonically to
# 0.9462 at idf 3.0), and part 6 saw train AP still rising at p=15, so capping this at 2.0 would
# truncate one task's answer. typo collapses above 1.0, so the grid has to span both
UNIFIED_IDF = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
UNIFIED_DIM = [1.0, 2.0]
UNIFIED_POSITION_WEIGHT = [0.0, 0.5, 1.0]
# normalize and denominator are not independent: 'geo' only enters through the normalizing
# denominator, so pairing them avoids 25% of the grid being duplicate rows
UNIFIED_NORM = [(False, 'dice'), (True, 'dice'), (True, 'geo')]


def unified_configs() -> List[tuple]:
    """(n, idf, dim, normalize, position_weight, denominator) for the whole unified grid"""
    out = []
    for n in UNIFIED_N:
        # dim is a no-op for a single n, so sweeping it there would duplicate rows
        for idf in UNIFIED_IDF:
            for dim in (UNIFIED_DIM if len(n) > 1 else [1.0]):
                for normalize, denom in UNIFIED_NORM:
                    for position_weight in UNIFIED_POSITION_WEIGHT:
                        out.append((n, idf, dim, normalize, position_weight, denom))
    return out


def load_shortlist(limit: int) -> List[tuple]:
    """
    the best `limit` configurations from a previous --aggregate, plus the library default

    exists for `ermagellan`, where the full grid is not affordable: at n=(1,) one configuration
    costs 341s and scores MAP 0.008, because every 138-character document contains every letter,
    so unigrams have no signal at all there. spending five hours measuring that is not worth it.
    running the shortlist instead still places ermagellan in the cross-task ranking -- aggregate()
    intersects on configurations measured everywhere, so it just narrows the field to these
    """
    path = RESULTS / 'index_param_search_overall.csv'
    if not path.exists():
        raise SystemExit(f'{path} not found -- run --aggregate on the other tasks first')
    chosen = []
    for row in read_csv(path)[:limit]:
        chosen.append((ast.literal_eval(row['n']), float(row['idf_exponent']),
                       float(row['dim']), row['normalize'] == 'True',
                       float(row['position_weight']), row['denominator']))
    default = ((2, 4), 0.0, 1.0, False, 1.0, 'dice')
    if default not in chosen:
        chosen.append(default)
    return chosen


def run_unified(task: str, documents: List[str], queries, configs: List[tuple] = None) -> List[dict]:
    configs = configs if configs is not None else unified_configs()
    by_build: Dict[tuple, List[tuple]] = defaultdict(list)
    for n, idf, dim, normalize, position_weight, denom in configs:
        by_build[(n, idf)].append((dim, normalize, position_weight, denom))

    print(f'\n=== {task}: {len(documents)} documents, {len(queries)} queries, top_k={TOP_K} ===')
    print(f'{len(by_build)} index builds, {len(configs)} configurations', flush=True)
    rows, done, started = [], 0, time.perf_counter()
    for (n, idf), variants in by_build.items():
        index = ApproxWordListV7(n=n, idf_exponent=idf).add_words(documents)
        index._freeze()
        for dim, normalize, position_weight, denom in variants:
            t0 = time.perf_counter()
            scores = evaluate(index, queries, dim, normalize, position_weight, denom)
            rows.append(dict(task=task, n=repr(n), idf_exponent=idf, dim=dim,
                             normalize=normalize, position_weight=position_weight,
                             denominator=denom, eval_sec=round(time.perf_counter() - t0, 2),
                             **{k: round(v, 5) for k, v in scores.items()}))
            done += 1
        elapsed = time.perf_counter() - started
        print(f'  {done:>4}/{len(configs)}  n={repr(n):<10} idf={idf:<4} '
              f'best MAP so far {max(r["map"] for r in rows):.4f}  '
              f'[{elapsed:.0f}s elapsed, ~{elapsed / done * (len(configs) - done):.0f}s left]',
              flush=True)
        del index
    return rows


def aggregate(tasks: List[str]) -> None:
    """
    rank configurations across every benchmark that has a unified result file

    raw MAP is not comparable between tasks -- typo tops out near 0.96 and abtbuy near 0.85 --
    so a plain mean would silently weight whichever task has the widest spread. two aggregations
    are reported instead, and they are only trustworthy where they agree:

      mean rank        each config's rank within its task, averaged. scale-free, but treats the
                       gap between 1st and 2nd as identical to 50th and 51st
      mean normalized  MAP rescaled per task so the task's worst config is 0 and its best is 1.
                       keeps the size of the gaps, but a task where everything scores similarly
                       gets its differences stretched to the same range as a task where they
                       do not

    worst rank is the tie-breaker that matters for a default: a configuration that wins on
    average by being excellent on three tasks and terrible on a fourth is not a good default
    """
    per_task: Dict[str, Dict[tuple, float]] = {}
    for task in tasks:
        path = RESULTS / f'index_param_search_{task}_unified.csv'
        if not path.exists():
            print(f'skipping {task}: {path.name} not found')
            continue
        rows = read_results(task, '_unified')
        per_task[task] = {(r['n'], r['idf_exponent'], r['dim'], r['normalize'],
                           r['position_weight'], r['denominator']): r['map'] for r in rows}
    if len(per_task) < 2:
        print('need at least two tasks with unified results to aggregate')
        return

    shared = set.intersection(*(set(scores) for scores in per_task.values()))
    print(f'\n=== best overall across {len(per_task)} benchmarks '
          f'({", ".join(sorted(per_task))}) ===')
    print(f'{len(shared)} configurations measured on every task')

    ranks: Dict[tuple, List[int]] = defaultdict(list)
    normalized: Dict[tuple, List[float]] = defaultdict(list)
    for task, scores in per_task.items():
        ordered = sorted(shared, key=lambda cfg: -scores[cfg])
        for position, cfg in enumerate(ordered, 1):
            ranks[cfg].append(position)
        low = min(scores[cfg] for cfg in shared)
        high = max(scores[cfg] for cfg in shared)
        span = (high - low) or 1.0
        for cfg in shared:
            normalized[cfg].append((scores[cfg] - low) / span)

    summary = []
    for cfg in shared:
        summary.append(dict(cfg=cfg,
                            mean_rank=sum(ranks[cfg]) / len(ranks[cfg]),
                            worst_rank=max(ranks[cfg]),
                            mean_norm=sum(normalized[cfg]) / len(normalized[cfg]),
                            per_task={t: per_task[t][cfg] for t in per_task}))

    task_names = sorted(per_task)
    header = (f'{"n":<10} {"idf":>4} {"dim":>4} {"norm":>6} {"pos_w":>6} {"den":>5} '
              f'{"mRank":>6} {"wRank":>6} {"mNorm":>6} ' +
              ' '.join(f'{t[:9]:>9}' for t in task_names))
    for label, key in (('mean rank (lower is better)', lambda s: s['mean_rank']),
                       ('mean normalized MAP (higher is better)', lambda s: -s['mean_norm'])):
        print(f'\n--- top 10 by {label} ---')
        print(header)
        for s in sorted(summary, key=key)[:10]:
            n, idf, dim, normalize, pw, den = s['cfg']
            print(f'{n:<10} {idf:>4} {dim:>4} {str(normalize):>6} {pw:>6} {den:>5} '
                  f'{s["mean_rank"]:>6.1f} {s["worst_rank"]:>6} {s["mean_norm"]:>6.3f} ' +
                  ' '.join(f'{s["per_task"][t]:>9.4f}' for t in task_names))

    print('\n--- best per task, for comparison with any single overall choice ---')
    for task in task_names:
        best_cfg = max(shared, key=lambda cfg: per_task[task][cfg])
        n, idf, dim, normalize, pw, den = best_cfg
        print(f'{task:<12} MAP {per_task[task][best_cfg]:.4f}  '
              f'n={n} idf={idf} dim={dim} normalize={normalize} '
              f'position_weight={pw} denominator={den}')

    default_cfg = (repr((2, 4)), 0.0, 1.0, False, 1.0, 'dice')
    if default_cfg in shared:
        print(f'\nlibrary default n=(2, 4) idf=0 dim=1 normalize=False:')
        for task in task_names:
            rank = sorted(shared, key=lambda cfg: -per_task[task][cfg]).index(default_cfg) + 1
            print(f'  {task:<12} MAP {per_task[task][default_cfg]:.4f}  rank {rank}/{len(shared)}')

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / 'index_param_search_overall.csv'
    with open(out, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'idf_exponent', 'dim', 'normalize', 'position_weight',
                         'denominator', 'mean_rank', 'worst_rank', 'mean_normalized_map']
                        + [f'map_{t}' for t in task_names])
        for s in sorted(summary, key=lambda s: s['mean_rank']):
            writer.writerow(list(s['cfg']) + [round(s['mean_rank'], 2), s['worst_rank'],
                                              round(s['mean_norm'], 4)]
                            + [s['per_task'][t] for t in task_names])
    print(f'\nwrote {out}')


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
                    'map', 'ndcg@10', 'ndcg@5', 'mrr', 'hit@1', 'hit@10', 'recall@10',
                    'p@10', 'rprec'):
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


# every benchmark, with the file each one needs. the two typo variants and the long-document
# task were added after the first sweep: words_ms.txt and british-english-insane.txt are both
# tracked in the repo and were read by no eval script, and er_magellan/valid.txt was on disk
# but referenced by zero lines of code
TASKS = {
    'typo': (lambda: load_typo(WORDS), WORDS),
    'typo_ms': (lambda: load_typo(WORDS_MS), WORDS_MS),
    'typo_hard': (lambda: load_typo(WORDS_INSANE, dict_size=30000, edits=(2, 2, 3)), WORDS_INSANE),
    'abtbuy': (load_abtbuy, DATA / 'Abt.csv'),
    'ermagellan': (load_ermagellan, DATA / 'er_magellan' / 'valid.txt'),
}

ALL_TASKS = list(TASKS)


def load_task(task: str):
    if task not in TASKS:
        raise SystemExit(f'unknown task {task!r}; choose from {ALL_TASKS}')
    loader, required = TASKS[task]
    if not Path(required).exists():
        print(f'\n=== {task}: skipped, {required} not found ===')
        if 'Abt' in str(required) or 'magellan' in str(required):
            print('the product datasets live in gitignored .scratch/data/ -- see the docstrings '
                  'in baseline_eval.py and er_magellan_eval.py for where to download them')
        return None
    return loader()


def main(which: str = 'both', quick: bool = False, stage2: bool = False,
         unified: bool = False, do_aggregate: bool = False, shortlist: int = 0) -> None:
    if which in ('both', 'all'):
        tasks = ALL_TASKS if which == 'all' else ['typo', 'abtbuy']
    else:
        tasks = which.split(',')

    if do_aggregate:
        aggregate(tasks)
        return

    for task in tasks:
        loaded = load_task(task)
        if loaded is None:
            continue
        documents, queries = loaded
        if unified:
            configs = load_shortlist(shortlist) if shortlist else None
            rows = run_unified(task, documents, queries, configs)
            summarize(task, rows)
            print(f'\nwrote {write_results(task, rows, suffix="_unified")}')
            continue
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
    shortlist_n = 0
    for arg in sys.argv[1:]:
        if arg.startswith('--shortlist='):
            shortlist_n = int(arg.split('=', 1)[1])
    main(args[0] if args else 'both', quick='--quick' in sys.argv, stage2='--stage2' in sys.argv,
         unified='--unified' in sys.argv, do_aggregate='--aggregate' in sys.argv,
         shortlist=shortlist_n)
