"""
`ApproxWordListV6` (the shipped `WordList`) against `ApproxWordListV7`, on build time,
index size and lookup time.

this exists because the README carried a V6-vs-V7 table whose numbers had no date and no
script behind them -- nothing in the tree regenerated them, which is exactly what CLAUDE.md
forbids. this is that table, made reproducible.

each cell runs in its own subprocess, so one index is never sized while the other is alive.
`--all` spawns the cells and prints the markdown table; a single cell can also be run alone.

vocabularies come from the tracked corpora: words_en.txt for anything up to its ~41.5k
distinct words, british-english-insane.txt above that. queries are indexed words corrupted by
1-2 character edits, using the same `corrupt` as the typo benchmarks, so the timings are taken
on the kind of query the index is for rather than on exact hits.

index size is a recursive walk of the object graph, not process RSS: RSS never shrinks when
the allocator frees, so it would charge V7 for the python dicts it builds and then throws away
in `_freeze`, which is the entire point of the class. the walk is cut down from
`experiments/sizeof.py::deep_sizeof` -- same numpy `nbytes` special case, same id-based
deduplication, minus the pandas and `__slots__` branches that nothing here has.

timing convention follows large_vocab_bench.py: the minimum over repeats per query, because
contention from other processes can only ever add time.

⚠ sizing V6 is itself expensive -- the walk visits every posting tuple, so it takes ~13s and
a few GB at 41.5k words and considerably more at 250k. `--no-size` skips it.

run:
    python experiments/v6_vs_v7_bench.py --all
    python experiments/v6_vs_v7_bench.py --all --sizes 41554 250000 --queries 20
    python experiments/v6_vs_v7_bench.py --index v6 --vocab-size 41554
"""
import argparse
import io
import json
import random
import subprocess
import sys
import time
from numbers import Number
from pathlib import Path
from typing import Callable
from typing import Generator
from typing import List
from typing import Mapping
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from index_param_search import WORDS
from index_param_search import WORDS_INSANE
from index_param_search import corrupt

N_LIST = (2, 4)
TOP_K = 10

NOT_ITERABLE = (io.IOBase, str, bytes, bytearray, range, Number, Generator, Callable, type)


def deep_sizeof(obj) -> int:
    """
    total bytes reachable from `obj`, counting each object once

    cut down from experiments/sizeof.py::deep_sizeof. numpy arrays report `nbytes` and are not
    recursed into (their `dir()` exposes views like `.T`, which recurse forever); everything
    else is `sys.getsizeof` plus its keys, values, elements and `__dict__`. name-mangled
    private attributes are reached through `__dict__`, which is how V6's postings get counted.
    """
    sizes = {}
    stack = {id(obj): obj}

    def push(thing):
        if id(thing) not in sizes:
            stack[id(thing)] = thing

    while stack:
        item_id, item = stack.popitem()
        if item_id in sizes:
            continue

        nbytes = getattr(item, 'nbytes', None)
        if isinstance(nbytes, int):
            sizes[item_id] = nbytes
            continue

        sizes[item_id] = sys.getsizeof(item)
        if isinstance(item, NOT_ITERABLE):
            continue

        if isinstance(item, (Mapping, dict)):
            for key in item.keys():
                push(key)
            for value in item.values():
                push(value)
        if hasattr(item, '__iter__') and hasattr(item, '__getitem__'):
            for element in item:
                push(element)
        for key, value in getattr(item, '__dict__', {}).items():
            push(key)
            push(value)

    return sum(sizes.values())


def load_vocab(size: int) -> List[str]:
    """`size` distinct words, from words_en.txt if it is big enough and the insane list if not"""
    path = WORDS if size <= 41554 else WORDS_INSANE
    with open(path, encoding='utf8') as f:
        words = sorted({w.strip().casefold() for w in f if w.strip()})
    if len(words) < size:
        raise SystemExit(f'{path} has only {len(words)} distinct words, need {size}')
    return words[:size]


def make_queries(vocab: Sequence[str], num_queries: int, rng: random.Random) -> List[str]:
    """indexed words corrupted by 1-2 edits, so every query is a near-miss rather than a hit"""
    out = []
    for word in rng.sample(list(vocab), min(len(vocab), num_queries * 4)):
        if len(word) < 4:
            continue
        query = corrupt(word, rng.choice([1, 2]), rng)
        if query != word:
            out.append(query)
        if len(out) == num_queries:
            break
    return out


def run_cell(index_name: str, vocab_size: int, num_queries: int, repeats: int, seed: int,
             measure_size: bool = True) -> dict:
    """build one index, size it, then time lookups on it; returns the row as a dict"""
    rng = random.Random(seed)
    vocab = load_vocab(vocab_size)
    queries = make_queries(vocab, num_queries, rng)

    start = time.perf_counter()
    if index_name == 'v6':
        from nmd.nmd_index import ApproxWordListV6
        index = ApproxWordListV6(N_LIST, filter_n=3)
        for word in vocab:
            index.add_word(word)
    elif index_name == 'v7':
        from nmd.nmd_index_v7 import ApproxWordListV7
        index = ApproxWordListV7(N_LIST)
        index.add_words(vocab)
    else:
        raise SystemExit(f'unknown index {index_name}')
    index.lookup(queries[0], top_k=TOP_K)  # V7 freezes lazily, so the first lookup is part of the build
    build_s = time.perf_counter() - start

    # time before sizing, never after: the walk allocates and frees a dict with one entry per
    # reachable object, which on V6 is gigabytes, and it leaves the heap fragmented enough to
    # move the lookup numbers by more than 2x. measured 2026-09-18, the reason for this order
    best = [float('inf')] * len(queries)
    for _ in range(repeats):
        for qi, query in enumerate(queries):
            start = time.perf_counter()
            index.lookup(query, top_k=TOP_K)
            best[qi] = min(best[qi], time.perf_counter() - start)
    lookup_ms = 1000 * sum(best) / len(best)

    size_mb = deep_sizeof(index) / 2 ** 20 if measure_size else float('nan')

    return dict(index=index_name, vocab=len(vocab), queries=len(queries),
                build_s=build_s, size_mb=size_mb, lookup_ms=lookup_ms)


def run_all(sizes: Sequence[int], num_queries: int, repeats: int, seed: int,
            measure_size: bool) -> None:
    rows = []
    for size in sizes:
        for index_name in ('v6', 'v7'):
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   '--index', index_name, '--vocab-size', str(size),
                   '--queries', str(num_queries), '--repeats', str(repeats),
                   '--seed', str(seed), '--json']
            if not measure_size:
                cmd.append('--no-size')
            print(f'... {index_name} at {size}', file=sys.stderr, flush=True)
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode != 0:
                raise SystemExit(f'cell {index_name}@{size} failed:\n{out.stdout}\n{out.stderr}')
            rows.append(json.loads(out.stdout.strip().splitlines()[-1]))

    print()
    print(f'n={N_LIST}, top_k={TOP_K}, {rows[0]["queries"]} corrupted queries, '
          f'min of {repeats}, python {sys.version.split()[0]}')
    print()
    print('| vocabulary | build | index size | lookup |')
    print('|------------|-------|------------|--------|')
    for row in rows:
        print(f'| {row["vocab"] / 1000:.1f}k, {row["index"].upper()} | {row["build_s"]:.1f}s '
              f'| {row["size_mb"]:.1f} MB | {row["lookup_ms"]:.1f} ms |')
    print()
    for size in sizes:
        v6 = next(r for r in rows if r['vocab'] == size and r['index'] == 'v6')
        v7 = next(r for r in rows if r['vocab'] == size and r['index'] == 'v7')
        print(f'{size}: lookup {v6["lookup_ms"] / v7["lookup_ms"]:.1f}x, '
              f'size {v6["size_mb"] / v7["size_mb"]:.1f}x')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='run every cell in its own subprocess')
    parser.add_argument('--sizes', type=int, nargs='+', default=(41554, 250000))
    parser.add_argument('--index', choices=('v6', 'v7'))
    parser.add_argument('--vocab-size', type=int, default=41554)
    parser.add_argument('--queries', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-size', action='store_true', help='skip the expensive object-graph walk')
    parser.add_argument('--json', action='store_true', help='print the row as json, for --all')
    args = parser.parse_args()

    if args.all:
        run_all(args.sizes, args.queries, args.repeats, args.seed, not args.no_size)
        return
    if not args.index:
        raise SystemExit('pass --index v6|v7, or --all')
    row = run_cell(args.index, args.vocab_size, args.queries, args.repeats, args.seed,
                   not args.no_size)
    if args.json:
        print(json.dumps(row))
    else:
        print(f'{row["index"]} at {row["vocab"]}: build {row["build_s"]:.1f}s, '
              f'index {row["size_mb"]:.1f} MB, lookup {row["lookup_ms"]:.1f} ms')


if __name__ == '__main__':
    main()
