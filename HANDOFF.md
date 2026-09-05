# Handoff

State as of `master` on 2026-09-05, one commit ahead of `origin/master`. The open-item list
below was triaged with the owner on 2026-09-05; every item now carries a decision.

⚠ The "What changed recently" table and the "Things worth knowing" section still describe the
tree as of `cbde8e7`. Everything in `docs/bow-plan.md` parts 3-8 (segment mover's distance,
the hyperparameter search, idf on both `WordSet` and V7, the all-metrics benchmark) landed
after that and is **not** summarized here yet.

## Environment

Conda env named after the repo folder:
`C:\Users\user\anaconda3\envs\ngram-movers-distance\python.exe`

```bash
"C:/Users/user/anaconda3/envs/ngram-movers-distance/python.exe" -m pytest -q
# 537 passed, 1 xfailed (2026-09-05)
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

## The standing constraint, decided 2026-09-05

**`nmd` stays dependency-free, and the default path stays pure python.** `import nmd` exports
exactly `ngram_movers_distance` and `WordList` and pulls in nothing third-party — verified by
importing it with `numpy` / `scipy` / `pyroaring` / `regex` / `numba` blocked at the meta-path,
which loads zero of them. Anything needing a third-party package (V7, `nmd_bow`,
`nmd_segments`, `WordSet`) is reached by its full module path and is documented in the README's
dependency table, **not** exported from `nmd/__init__.py` and **not** added to
`pyproject.toml` as a dependency or an extra.

**The old index classes are frozen, not deleted.** V3 and V5 keep their bugs on purpose, for
comparison. V6 is the shipped `WordList` and is therefore *not* frozen — bugs that reach it
get fixed.

## Open items

### Closed on 2026-09-05

1. **Should `WordList` point at V7?** No — that would make numpy a hard dependency. Stays
   `ApproxWordListV6`; V7 remains opt-in via `from nmd.nmd_index_v7 import ApproxWordListV7`.
2. **Should `pyproject.toml` declare dependencies or extras?** No. Declaring none is correct
   and intended. The optional modules and what they each need are now a table in the README
   instead. (`__version__` is still `0.0.6` and still needs bumping before any `flit publish`
   — that is the one part of this item left open, see #13.)
3. **`nmd_core` now calls `emd_1d_fast`** instead of `emd_1d_dp`. Approved because
   `emd_1d_fast` is pure python and lives in the same module, so it costs no dependency.
   Same values, ~1.15x on pairwise comparisons; equivalence pinned by
   `tests/test_bow.py::TestEmd1dFast`.
8. **Fixed: `ApproxWordListV6.lookup()` `ZeroDivisionError` for a query of length `n - 2`.**
   `add_word()` had always guarded this; only the lookup path had not.
9. **Fixed: `num_grams()` on the default path.** V6 now uses a new corrected `num_n_grams()`
   in the same module. `num_grams()` itself is untouched, because V3, V5 and
   `tests/test_index_v7.py` all still depend on its behaviour.

   Both fixes are pinned by `tests/test_index_v6.py`, which was re-run against the pre-fix
   tree to confirm exactly four of its tests actually go red (listed in that file's docstring).
   ⚠ Correcting the counts made a `0 / 0` reachable that the old *negative* counts had been
   masking, so the normalizing denominators are now clamped with `max(1, ...)`; deleting either
   clamp makes `TestShortWordDenominatorClamp` raise. Neither fix changes any score on the
   `n=(2, 4)` default, where the old and new counts agree for every non-empty word.

15. **Fixed: `WordSet.find_similar(min_similarity=...)` was validated and then ignored.** The
    only code applying it was inside the commented-out exact-rescoring block, so asking for
    `>= 0.99` returned whatever the top k happened to be. It now filters on the score the
    method actually returns (the normalized approximate similarity, already clamped to
    `[0, 1]`), breaking early since the list is sorted best-first.
16. **Fixed: `WordSet`'s default `ngram_sizes` was `(2, 3, 4)` while its docstring said
    `(2, 4)`.** Changed the code to match the docs. ⚠ This also moves `_effective_filter_n`
    from 3 to 2, because the filter is `3 if 3 in n_list else min(n_list)`. The filter is
    therefore less selective: more candidates get scored (slower) but recall goes up — a
    1-character query returned nothing at all under the old default and now returns matches.
17. **The no-op loop in `find_similar` is commented out, not deleted.** It computed
    `norm_factor_for_n` and discarded it, once per n per candidate word. It is left in place
    because it records an abandoned design — scoring and normalizing per n, as V6 and V7 both
    do, rather than pooling every n into one `ngrams_pos` dict. Picking that up means splitting
    `ngrams_pos` by n.
18. **`ApproxWordListV6` now calls `emd_1d_fast`** too, not just `nmd_core`. Verified by
    dumping top-5 lookups for 8 queries x 4 n-lists over a 4000-word vocabulary in two separate
    processes (not by rebinding a global): **0 ranking differences**, largest score difference
    1.8e-15 — summation-order noise, since `emd_1d_fast` adds `(m - 1)` once where the dp
    accumulates `1.0` that many times.
19. **`build_system_prompt.py` deleted** from the repo root. It was untracked, unreferenced,
    unrelated to nmd, and published as a gist (URL in its own docstring), so this was not the
    only copy. No `system prompt.txt` existed.

### Deliberately not doing, keep documented

4. **Cross-call n-gram cache for `nmd_bow`.** Measured at only ~1.2x on top of what landed
   (79.5% hit rate over a 3000-candidate sweep), so it was deliberately left out — it is the
   same globally-scoped pattern that was removed from `get_n_grams`. If it ever matters, the
   right shape is an explicit indexed API over the candidate corpus, not a hidden global.
   Prototype: `.scratch/bench_bow_many.py`. **Reaffirmed 2026-09-05: skip, keep written down.**

5. **numba kernel for the bow cost matrix.** Prototype in `.scratch/kernel_bow.py`, correct
   and verified to 1.8e-15. **Not landed on purpose**: 4.5-9x above ~20 words per bag, but a
   *loss* (0.8x) at 5 words, which is the README's own use case. Costs ~143 MB of dependency
   (numba 25 + llvmlite 86 + numpy 32), a `numpy<2.3` pin, and ~1.8s JIT compile on first
   call. Only worth it for document-length token sequences. **Reaffirmed 2026-09-05: skip,
   keep documented.** Also now blocked by the dependency-free constraint above.

6. `ApproxWordListV3.vocabulary` is `return sorted(self.vocabulary)` — infinite recursion.
   **Won't fix**: V3 is frozen, is not reachable from the default namespace, and adds nothing
   over V5/V6 (it predates the pruning bound and does not even use the same scoring formula).
7. `ApproxWordListV5.lookup(invert=False)` returns `normalize - match_score`, i.e. bool
   arithmetic, so distances are negative and sorted worst-first; also returns `top_k * 2`
   results instead of `top_k`. **Won't fix**: frozen. V5 also still raises `ZeroDivisionError`
   for a query of length `n - 2` (confirmed 2026-09-05) — that is the same bug #8 was, and it
   stays in V5 on purpose.

### Still open

10. **`nmd/nmd_word_set.py` core scoring is still unverified.** `tests/test_word_set.py` (new,
    2026-09-05) now covers the API surface — `min_similarity`, defaults, set protocol, unicode,
    edge cases — and `test_word_set_idf.py` covers idf, but **whether `find_similar`'s scoring
    is correct is still untested**. It returns the approximate score directly, with the exact
    rescoring pass commented out, and there is no parity test against
    `ngram_movers_distance`. That parity test is the obvious next thing to write here.
    It imports `ApproxWordListV5` for the `__main__` benchmark at the bottom of the file (so
    the import is live, not dead), which means that benchmark compares against a class
    carrying bug #7.
11. `emd_1d_slow` is dead code using `itertools.combinations`, i.e. factorial blowup in the
    worst case. (`emd_1d_old` is no longer dead — `tests/test_emd_correctness.py` imports it
    as an oracle.)
12. Pre-existing README todos, untouched: `remove()`/`discard()` on the V7 index (needs index
    compaction; `WordSet` already has them), prefix lookup, a `min_similarity` filter on
    lookup, trying cython, and `from nmd import nmd` returning a module rather than a function.
13. **`__version__` is `0.0.6` and has not been bumped** for anything since. Needs
    incrementing before `flit publish`; not bumped unprompted, since it is only meaningful at
    publish time.
14. **The Part 7 finding has no corresponding code change.** `n=(1, 2)` beats the README's
    `(2, 4)` default by 11 points of hit@1 on typo correction, but unigrams have almost no
    selectivity as an index key. The suggested design — prune on 2- or 3-grams, rescore with
    `n=(1, 2)` — is not implemented anywhere, and the defaults are unchanged.

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
