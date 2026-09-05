# Handoff

State as of `master` on 2026-09-06. The open-item list below was triaged with the owner on
2026-09-05; every item carries a decision.

⚠ The "What changed recently" table and the "Things worth knowing" section still describe the
tree as of `cbde8e7`. Everything in `docs/bow-plan.md` parts 3-8 (segment mover's distance,
the hyperparameter search, idf on both `WordSet` and V7, the all-metrics benchmark) landed
after that and is **not** summarized here yet.

## Session 2026-09-06: index parameter search

**V7 gained two lookup knobs and flipped one default.** `position_weight` (how much of the
positional displacement to charge, 0.0 == order-blind dice-multiset, 1.0 == nmd) and
`denominator` (`dice` additive vs `geo` geometric). Both are pure read-path, so sweeping them
needs no rebuild. `lookup(normalize=...)` now defaults to **True**, unlike V6 and
`ngram_movers_distance`; that divergence is deliberate and pinned by
`tests/test_index_v7_knobs.py::TestDefaults`.

**Six retrieval benchmarks now exist**, in `experiments/index_param_search.py`: `typo`,
`typo_ms`, `typo_hard`, `typo_brutal`, `abtbuy`, `ermagellan`. Real MAP and NDCG live in
`experiments/ranking_metrics.py` -- neither existed anywhere in the tree before; every eval
script inlined a `rank = next(...)` loop that took only the first relevant item's rank.
Results are tracked CSVs under `experiments/results/`.

**The headline result: the tasks split into two clusters that disagree on every knob.**

| cluster | tasks | `n` | idf | dim | normalize | position_weight |
|---|---|---|---|---|---|---|
| short strings | typo, typo_ms, typo_hard, typo_brutal | (1, 2) | 0-0.5 | 2 | True | 1.0 |
| product matching | abtbuy, ermagellan | (2, 3) | 3.0 | 1 | False | 0.0 |

Best single compromise is `n=(1,2) idf=0.5 dim=2 normalize=True position_weight=1.0
denominator='geo'` -- top of the six-task ranking, best mean rank (51.2/432) of the five-task
one. Its worst rank is 234/432, on abtbuy. **The old default `n=(2,4) idf=0 dim=1
normalize=False` ranks 334 / 161 / 99 / 94 / 98 of 432 across the five full-grid tasks**: not a
compromise, just unchosen.

⚠ **`normalize` and `idf_exponent` substitute for each other** -- both counteract length bias,
so applying both over-corrects. On abtbuy `normalize=True` is worth +0.145 MAP at
`idf_exponent=0` and **-0.079** at `idf_exponent=3`. Any future default change to one has to
re-check the other.

**The eval scripts now drive V7 instead of scoring pairwise** (`experiments/v7_ranking.py`).
Verified identical against the pre-port code at matched seeds, except three
`benchmark_all` task-A idf rows that differ in the third decimal because `make_metrics` fits idf
over Abt+Buy while the index fits over Buy alone -- those are relabelled `[V7 idf]`.
`benchmark_all` task A2, `baseline_eval` part B and `segment_eval` are **not** ported on
purpose: they rerank 20 candidates, where building an index costs more than it saves.
`benchmark_all` task B is not retrieval at all.

⚠ **V7's "~98% of postings are single-occurrence" claim is short-word-specific.** Measured
2026-09-06: 99.5% on typo (8-char), 92.3% on abtbuy (53-char), **78.8% on ermagellan
(138-char)**. On long documents 21% of postings fall to the python dp fallback, which is why one
`ermagellan` configuration costs 67s and an `n=(1,)` one costs 341s. If long documents ever
matter, that fallback is the thing to vectorise.

⚠ **`n=(1,)` is useless on long documents** -- MAP 0.008 on ermagellan, because every
138-character document contains every letter. Unigrams only work when documents are short.

## Environment

Conda env named after the repo folder:
`C:\Users\user\anaconda3\envs\ngram-movers-distance\python.exe`

```bash
"C:/Users/user/anaconda3/envs/ngram-movers-distance/python.exe" -m pytest -q
# 992 passed, 1 xfailed (2026-09-05)
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

10. **Closed: `nmd/nmd_word_set.py`'s core scoring is verified.** `tests/test_parity_with_nmd.py`
    (new) scores every index against `ngram_movers_distance` recomputed from the raw strings,
    over 2000 words of `experiments/words_en.txt` and 30 queries. Findings:

    * `WordSet.find_similar` **is** exactly nmd (worst error 2.2e-16), but under a *pooled*
      normalization: one ratio `sum_n similarity / sum_n n-gram count`, where V5/V6/V7 return
      the mean of the per-n ratios. Both are correct; they are different aggregations, and they
      differ by ~0.09 on short words, so `WordSet` and `WordList` are not expected to agree.
    * V6 agrees bit-for-bit (0.0), V7 and V5 to summation noise (worst 3.6e-15). V5's *scores*
      were never wrong — its frozen bugs are in the returned distance and top-k, not the math.
    * V3 genuinely does not compute nmd (`max(c1, c2)` not `c1 + c2`, normalized by the query
      alone); pinned as a freeze guard rather than fixed, per #6.
    * With its lossy `filter_n=3` off, V6 returns the exhaustive top-5 exactly; so do V7 (no
      prefilter) and `WordSet` (whose `min(n) == 2` filter is lossless, since any shared 4-gram
      contains a shared 2-gram). Default V6 recovered 146/150 of the true top-5.
    * Two divergences from nmd are documented and pinned, not fixed: `WordSet` keeps the
      START/END markers at `n == 1`; and V6/V7 score a `0 / 0` normalization as 0.0 where nmd
      falls back to string equality and returns 1.0 (identical words of length `<= n - 3`).

    Sabotage-checked: `+ 1e-6` on each class's similarity accumulator turns exactly that
    class's tests red (164 failures for V5+V6, 232 for V7+`WordSet`, none crossing over). Note
    `nmd_word_set.py` still imports `ApproxWordListV5` for its `__main__` benchmark, which
    therefore compares against a class carrying bug #7.

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

    **Confirmed and strengthened 2026-09-06.** The finding survives real candidate generation
    and MAP scoring, not just pairwise rescoring: `n=(1, 2)` is the best n-list on all four
    typo benchmarks. It is also *still* the wrong index key — `n=(1,)` costs 341s per
    configuration on `ermagellan` and scores MAP 0.008 there. So the split prune/rescore design
    is now the highest-value unimplemented item, and it would need a second n-list parameter
    (`rescore_n`) that V7 does not have.

20. **`n` is doing two jobs and a grid search cannot separate them.** It selects candidates
    *and* scores them, so a single sweep conflates "which n-grams make a good index key" with
    "which make a good scorer". That is exactly what item 14 proposes splitting. Until it is
    split, any recommended `n` is a compromise between the two roles rather than an optimum for
    either.

21. **`ermagellan` was only measured on a 19-configuration shortlist**, not the full 432-point
    grid, because `n=(1,)` there costs 341s per configuration to measure noise. The six-task
    aggregate therefore ranks 19 configurations; the five-task one ranks all 432. Both agree on
    the winner, but do not quote a "rank N/19" as if it came from the full grid.

22. **`idf_exponent`'s optimum on product matching is still at the grid edge.** abtbuy peaks at
    the largest value swept (3.0 → 0.9500, 4.0 → 0.9467 so it may just have turned over), and
    `docs/bow-plan.md` part 6 saw train AP still climbing at p=15. Part 6 also names three
    guards that were never implemented and would matter at high exponents: an idf floor,
    punctuation normalization, and fitting df on an external corpus.

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
