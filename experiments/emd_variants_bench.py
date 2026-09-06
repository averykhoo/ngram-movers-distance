"""
profile every 1-D EMD variant on the workload that actually reaches them

`nmd/emd_1d.py` ships five implementations; this times the three that are not dominated. Which is fastest depends entirely on the *shape* of
the inputs, and the shape depends on the corpus:

    short words (8 chars)      98.0% of calls are 1-vs-1, 1.8% are 1-vs-many. `emd_1d_fast`
                               closed-forms exactly those, which is why it ships.
    long documents (138 chars) 74.5% of unigram postings are multi-occurrence, so the general
                               dp fallback runs constantly. measured 2026-09-06 it is 78% of an
                               `ermagellan` lookup at n=(2,4) and 97% at n=(1,).

so the "emd is not the bottleneck, it was ~7% of a V6 lookup" note in HANDOFF.md is a
short-word result and does not carry over. this script builds the real (x, y) location-list
workload out of an index and times every variant on it.

`emd_1d_hybrid` is the interesting one here: it carries hand-written preprocessing the dp does
not have -- an equal-length shortcut (1-D optimal transport with equal masses is just the sorted
monotone matching), cancellation of points common to both lists, and single-point shortcuts.
Those rules were written when nothing exercised them.

run:  python experiments/emd_variants_bench.py [task] [n]
      python experiments/emd_variants_bench.py ermagellan 1
      python experiments/emd_variants_bench.py typo 2
"""
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable
from typing import List
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nmd.emd_1d import emd_1d_dp
from nmd.emd_1d import emd_1d_fast
from nmd.emd_1d import emd_1d_hybrid
from nmd.nmd_index_v7 import ApproxWordListV7
from nmd.nmd_index_v7 import n_grams

VARIANTS = {
    'emd_1d_dp': emd_1d_dp,
    'emd_1d_fast': emd_1d_fast,
    'emd_1d_hybrid': emd_1d_hybrid,
}

# two variants are deliberately excluded, both strictly dominated:
#   emd_1d_slow  enumerates C(len_x, len_y) combinations -- factorial worst case (HANDOFF #11)
#   emd_1d_old   is emd_1d_hybrid's preprocessing followed by that same brute-force enumeration,
#                where hybrid substitutes a dp. same rules, worse tail, so hybrid dominates it.
#                it stays in nmd/emd_1d.py because tests/test_emd_correctness.py uses it as an
#                independent oracle -- being slow and obviously-correct is the point there


def collect_workload(task: str, n: Tuple[int, ...], max_pairs: int = 20000,
                     max_queries: int = 40) -> List[Tuple[tuple, tuple]]:
    """
    the exact (query locations, indexed locations) pairs `_similarity_vectors` sends to the dp

    only the both-sides-repeat case, since that is the branch that calls emd_1d_dp at all --
    the single-point branches are closed-formed inline in nmd_index_v7 and never reach here
    """
    import index_param_search as search

    loaded = search.load_task(task)
    if loaded is None:
        raise SystemExit(f'task {task} unavailable')
    documents, queries = loaded
    word_list = ApproxWordListV7(n=n).add_words(documents)
    word_list._freeze()

    pairs = []
    for query, _relevant in queries[:max_queries]:
        for n_idx, size in enumerate(n):
            grams = n_grams(query, size)
            divisor = len(grams) - 1
            locations = {}
            for idx, gram in enumerate(grams):
                locations.setdefault(gram, []).append(idx / divisor if divisor > 0 else 0.0)
            for gram, locs in locations.items():
                if len(locs) == 1:
                    continue  # handled by a closed form in the index, never reaches the dp
                for _word_index, other_locs in word_list._multi[n_idx].get(gram, ()):
                    pairs.append((tuple(locs), tuple(other_locs)))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def verify(pairs: List[Tuple[tuple, tuple]], limit: int = 3000) -> None:
    """every variant must agree with emd_1d_dp before any timing is worth reading"""
    print('correctness against emd_1d_dp:')
    for name, fn in VARIANTS.items():
        worst = 0.0
        for x, y in pairs[:limit]:
            worst = max(worst, abs(fn(x, y) - emd_1d_dp(x, y)))
        verdict = 'OK' if worst < 1e-9 else 'DISAGREES'
        print(f'  {name:<14} worst |diff| {worst:>9.3g}  {verdict}  ({min(limit, len(pairs))} checked)')


def benchmark(pairs: List[Tuple[tuple, tuple]]) -> None:
    shapes = Counter((len(x), len(y)) for x, y in pairs)
    grid = sum(len(x) * len(y) for x, y in pairs) / len(pairs)
    print(f'\nworkload: {len(pairs)} (x, y) pairs, mean dp grid {grid:.0f} cells')
    print(f'  most common shapes: {shapes.most_common(5)}')
    equal = sum(1 for x, y in pairs if len(x) == len(y))
    print(f'  equal-mass {100 * equal / len(pairs):.1f}%  (the sorted-matching shortcut applies)')

    print(f'\n{"variant":<16} {"total":>9} {"per call":>10} {"vs dp":>8}')
    baseline = None
    for name, fn in VARIANTS.items():
        start = time.perf_counter()
        for x, y in pairs:
            fn(x, y)
        elapsed = time.perf_counter() - start
        per_call = elapsed / len(pairs)
        if baseline is None:
            baseline = per_call
        print(f'{name:<16} {elapsed:>8.3f}s {per_call * 1e6:>9.1f}us {baseline / per_call:>7.2f}x')


def main(task: str = 'ermagellan', n_arg: str = '1') -> None:
    n = tuple(int(part) for part in n_arg.split(','))
    print(f'=== {task}, n={n} ===')
    pairs = collect_workload(task, n)
    if not pairs:
        print('no both-sides-repeat pairs in this workload -- the dp is never called here')
        return
    verify(pairs)
    benchmark(pairs)


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['ermagellan', '1']))
