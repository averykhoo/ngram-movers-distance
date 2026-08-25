# Handoff

State as of commit `cbde8e7` on `master`. Working tree clean, local and remote in sync.

## Environment

Conda env named after the repo folder:
`C:\Users\avery\anaconda3\envs\ngram-movers-distance\python.exe`

```bash
"C:/Users/avery/anaconda3/envs/ngram-movers-distance/python.exe" -m pytest -q   # 192 passing
```

Installed and used: `numpy` 2.2.4, `scipy` 1.15.2, `pyroaring`, `regex`. `numba` 0.61.2 is
installed but **nothing in `nmd/` imports it** — it was only used to evaluate a prototype.

## What changed recently

| commit | what |
|---|---|
| `f72e7d1` | added `nmd/nmd_index_v7.py` (numpy-backed index, ~20-29x faster than V6, ~11-14x less memory) + `tests/test_index_v7.py`; dropped the `lru_cache` from `get_n_grams` |
| `cbde8e7` | hoisted n-gram splitting out of the `nmd_bow` cost matrix loop (2-5.7x) + `tests/test_bow.py`; added `emd_1d_fast` |

Benchmarks and profiling scripts behind those numbers are in `.scratch/` (gitignored, see
`.scratch/README.md` for which script proved what).

## Open items

### Decisions waiting on you

1. **Should `WordList` point at V7?** `nmd/__init__.py` still has `WordList = ApproxWordListV6`.
   Flipping it makes `numpy` a hard dependency of a package that currently has none. Left as
   opt-in (`from nmd.nmd_index_v7 import ApproxWordListV7`), mirroring how `nmd_bow` handles
   scipy. Not flipped without a call from you.

2. **Packaging declares no dependencies at all.** `pyproject.toml` has no `dependencies` key,
   but `nmd_bow` needs `scipy`, `nmd_index_v7` needs `numpy`, and `nmd_word_set` needs
   `pyroaring` + `regex`. Anyone installing from PyPI and following the README hits an
   ImportError. Optional extras would fix it:
   ```toml
   [project.optional-dependencies]
   bow = ["scipy"]
   index = ["numpy"]
   wordset = ["pyroaring", "regex"]
   ```
   Also `__version__` is still `0.0.6` and was **not** bumped for either commit — needs
   incrementing before `flit publish`.

### Cheap wins, not done

3. **`nmd_core.ngram_movers_distance` still calls `emd_1d_dp`.** Switching it to the new
   `emd_1d_fast` is a one-line import change, measured at ~1.15x on pairwise comparisons.
   Left alone because the request was scoped to the bow path. `emd_1d_fast` is proven
   equivalent to `emd_1d_dp` in `tests/test_bow.py::TestEmd1dFast` (exhaustive over a
   quantized grid, random input, and outside the unit interval).

4. **Cross-call n-gram cache for `nmd_bow`.** Measured at only ~1.2x on top of what landed
   (79.5% hit rate over a 3000-candidate sweep), so it was deliberately left out — it is the
   same globally-scoped pattern that was removed from `get_n_grams`. If it ever matters, the
   right shape is an explicit indexed API over the candidate corpus, not a hidden global.
   Prototype: `.scratch/bench_bow_many.py`.

5. **numba kernel for the bow cost matrix.** Prototype in `.scratch/kernel_bow.py`, correct
   and verified to 1.8e-15. **Not landed on purpose**: 4.5-9x above ~20 words per bag, but a
   *loss* (0.8x) at 5 words, which is the README's own use case. Costs ~143 MB of dependency
   (numba 25 + llvmlite 86 + numpy 32), a `numpy<2.3` pin, and ~1.8s JIT compile on first
   call. Only worth it for document-length token sequences.

### Known bugs, documented but deliberately not fixed

Kept for archaeological reasons — the old index classes are intentionally frozen. All are
listed under "known bugs in the older index classes" in `README.md`, and V7 has none of them.

6. `ApproxWordListV3.vocabulary` is `return sorted(self.vocabulary)` — infinite recursion.
7. `ApproxWordListV5.lookup(invert=False)` returns `normalize - match_score`, i.e. bool
   arithmetic, so distances are negative and sorted worst-first; also returns `top_k * 2`
   results instead of `top_k`.
8. `ApproxWordListV6.lookup()` raises `ZeroDivisionError` for any query of length `n - 2`
   (e.g. a 2-char query when 4-grams are indexed). **This one affects the shipped
   `WordList`**, so it is the most likely to bite a real user.
9. `num_grams()` disagrees with `get_n_grams()`: over-counts by 2 at `n=1`, and returns a
   negative count when the word is shorter than `n - 2`.

### Not looked at

10. **`nmd/nmd_word_set.py` has no tests and was never reviewed this session.** It imports
    `ApproxWordListV5`, which carries bug #7 above. It is ~530 lines and imports fine, but
    whether `WordSet` is correct is unverified.
11. `emd_1d_slow` and `emd_1d_old` are dead code using `itertools.combinations`, i.e.
    factorial blowup in the worst case. Nothing imports them.
12. Pre-existing README todos, untouched: `remove()`/`discard()` for a set-like container
    (needs index compaction), prefix lookup, a `min_similarity` filter on lookup, trying
    cython, and `from nmd import nmd` returning a module rather than a function.

## Things worth knowing before optimizing further

* **emd is not the bottleneck.** It was ~7% of a V6 lookup. 98.0% of emd calls are 1-vs-1
  and 1.8% are 1-vs-many, which is what `emd_1d_fast` exploits. Profile before assuming the
  transport math is the expensive part — it usually isn't.
* **`__ngram_counts` is redundant data but not redundant performance.** It is fully derivable
  from `__ngram_positions`, yet measurably faster to scan (3.9ms vs 5.2ms) because the tuples
  are smaller. Confirmed, not a python-version artifact. V7 sidesteps it with flat int32
  arrays.
* **A/B a function by running separate processes, not by rebinding a module global.** Cache
  contents and GC state carry over between in-process variants and gave a misleading result
  once already (`.scratch/check_cache.py` vs `check_cache2.py`).
* **V7's pruning is exact, not lossy.** Similarity for a shared n-gram at counts `(c1, c2)`
  lies in `[min(c1,c2), 2*min(c1,c2)]`, so summed min-counts bracket the true score. If you
  change the scoring, that two-sided bound is the invariant to preserve, and
  `tests/test_index_v7.py::test_pruning_never_drops_a_true_top_k` is what checks it.
