"""
load test for `ApproxWordListV7` at a large vocabulary of multi-word phrases

the 300k single-word run recorded in HANDOFF.md ("Large vocabularies: what actually costs
time") had no tracked script behind it -- its only trace was a log in the gitignored
`.scratch/`. this is that benchmark, kept, and pushed to where the index is expected to hurt:

    * 1M documents instead of 300k, so every vocabulary-sized array op is 3.3x larger
    * phrases of 2-5 dictionary words instead of single words, so documents are ~3x longer,
      and the space character makes n-grams like "e " and " t" repeat *within* a document.
      that moves postings off the vectorised single-occurrence path and onto `_multi`, which is
      the regime HANDOFF says the index was never tuned for

the phrases are synthetic: random draws from british-english-insane.txt joined with spaces, so
the run is self-contained. queries are indexed phrases corrupted by character edits (the same
`corrupt` as the typo tasks) and, for a third of them, by dropping a word as well, so there is
one right answer per query and MAP == MRR.

reports build time, resident memory, posting-list shape (share of postings on the single vs
multi path), lookup time per configuration, retrieval quality as a sanity check, the share of
the vocabulary touched / surviving pruning, and a cProfile of the tuned configuration.

run:
    python experiments/large_vocab_bench.py                       # 1M phrases
    python experiments/large_vocab_bench.py --size 100000         # smaller
    python experiments/large_vocab_bench.py --no-profile
"""
import argparse
import cProfile
import ctypes
import pstats
import random
import sys
import time
from pathlib import Path
from typing import List
from typing import Sequence
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from index_param_search import WORDS_INSANE
from index_param_search import corrupt
from ranking_metrics import reciprocal_rank

from nmd.nmd_index_v7 import ApproxWordListV7

TOP_K = 10

# (label, lookup kwargs). the first is the shipped default; the second is what HANDOFF
# recommends per domain and is slower at scale because both knobs add vocabulary-sized math
CONFIGS = [
    ('d1/dice', dict(dim=1, denominator='dice')),
    ('d2/geo', dict(dim=2, denominator='geo')),
]


def rss_mb() -> float:
    """resident set size in MB; windows has no `resource` module so this goes through psapi"""
    if sys.platform != 'win32':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    class _Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_uint32), ('PageFaultCount', ctypes.c_uint32),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]

    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                                      counters.cb):
        return float('nan')
    return counters.WorkingSetSize / 2 ** 20


def load_dictionary() -> List[str]:
    with open(WORDS_INSANE, encoding='utf8') as f:
        return sorted({w.strip().casefold() for w in f
                       if w.strip().isalpha() and 3 <= len(w.strip()) <= 15})


def make_phrases(vocab: Sequence[str], size: int, words_per_phrase: Tuple[int, int],
                 rng: random.Random) -> List[str]:
    """`size` distinct phrases, each `lo..hi` dictionary words joined by single spaces"""
    lo, hi = words_per_phrase
    phrases = set()
    while len(phrases) < size:
        phrases.add(' '.join(rng.choice(vocab) for _ in range(rng.randint(lo, hi))))
    return sorted(phrases)


def make_queries(phrases: Sequence[str], num_queries: int, rng: random.Random,
                 edits: Sequence[int] = (1, 2, 3)) -> List[Tuple[str, str]]:
    """(query, answer) pairs: character edits, and for every third query a dropped word too"""
    out = []
    for i, answer in enumerate(rng.sample(phrases, num_queries)):
        words = answer.split(' ')
        if i % 3 == 0 and len(words) > 2:
            del words[rng.randrange(len(words))]
        query = corrupt(' '.join(words), rng.choice(list(edits)), rng)
        if query != answer:
            out.append((query, answer))
    return out


def posting_shape(index: ApproxWordListV7) -> None:
    """how much of the index is on the vectorised path vs the multi-occurrence fallback"""
    for n_idx, n in enumerate(index.n_list):
        single = sum(locs.size for locs in index._locs[n_idx].values())
        multi_postings = sum(len(entries) for entries in index._multi[n_idx].values())
        multi_locs = sum(len(locs) for entries in index._multi[n_idx].values()
                         for _, locs in entries)
        total = single + multi_postings
        print(f'  n={n}: {len(index._words[n_idx])} distinct n-grams, {total} postings, '
              f'{100 * single / total:.1f}% single-occurrence, '
              f'{multi_postings} multi ({multi_locs} locations, '
              f'{multi_locs / max(1, multi_postings):.2f} per posting)')


def selectivity(index: ApproxWordListV7, queries, **kwargs) -> None:
    """share of the vocabulary that shares at least one n-gram with the query, i.e. gets scored"""
    touched = 0
    for query, _ in queries:
        found = index._similarity_vectors(query, normalize=True, **kwargs)
        if found is None:
            continue
        _scores, combined = found
        touched += int(np.count_nonzero(combined > 0))
    print(f'  touched {100 * touched / (len(index) * len(queries)):.1f}% of the vocabulary per query')


def time_lookups(index: ApproxWordListV7, queries, repeats: int) -> None:
    """
    interleave the configurations query by query and keep the minimum over `repeats`, per the
    HANDOFF measurement note: contention only ever adds time, so the minimum is the estimate
    """
    best = {label: [float('inf')] * len(queries) for label, _ in CONFIGS}
    mrr = {label: 0.0 for label, _ in CONFIGS}
    for rep in range(repeats):
        for qi, (query, answer) in enumerate(queries):
            for label, kwargs in CONFIGS:
                start = time.perf_counter()
                ranked = index.lookup(query, top_k=TOP_K, **kwargs)
                elapsed = time.perf_counter() - start
                best[label][qi] = min(best[label][qi], elapsed)
                if rep == 0:
                    mrr[label] += reciprocal_rank([w for w, _ in ranked], {answer})
    for label, _ in CONFIGS:
        times = np.array(best[label]) * 1000
        print(f'  {label:8s} mean {times.mean():8.2f} ms  median {np.median(times):8.2f} ms  '
              f'max {times.max():8.2f} ms   MRR {mrr[label] / len(queries):.3f}')


def profile(index: ApproxWordListV7, queries, num: int, **kwargs) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for query, _ in queries[:num]:
        index.lookup(query, top_k=TOP_K, **kwargs)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime').print_stats(22)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1_000_000)
    parser.add_argument('--queries', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--words', type=int, nargs=2, default=(2, 5), metavar=('LO', 'HI'))
    parser.add_argument('--n', type=int, nargs='+', default=(2, 4))
    parser.add_argument('--idf', type=float, default=0.0)
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    vocab = load_dictionary()
    print(f'dictionary: {len(vocab)} words')
    phrases = make_phrases(vocab, args.size, tuple(args.words), rng)
    queries = make_queries(phrases, args.queries, rng)
    lengths = [len(p) for p in phrases]
    print(f'phrases: {len(phrases)}, {min(lengths)}-{max(lengths)} chars, '
          f'mean {np.mean(lengths):.1f}; queries: {len(queries)}')
    print(f'  e.g. {queries[:3]}')

    before = rss_mb()
    start = time.perf_counter()
    index = ApproxWordListV7(tuple(args.n), idf_exponent=args.idf)
    index.add_words(phrases)
    index.lookup(queries[0][0])  # forces _freeze so the build time includes it
    print(f'build {time.perf_counter() - start:.1f}s, rss {before:.0f} -> {rss_mb():.0f} MB')
    posting_shape(index)

    print('selectivity at n=%s top_k=%d:' % (index.n_list, TOP_K))
    selectivity(index, queries[:20], dim=1, denominator='dice')

    print(f'lookup, min of {args.repeats} over {len(queries)} queries:')
    time_lookups(index, queries, args.repeats)

    if not args.no_profile:
        for label, kwargs in CONFIGS:
            print(f'=== profile {label}, {min(20, len(queries))} queries ===')
            profile(index, queries, 20, **kwargs)


if __name__ == '__main__':
    main()
