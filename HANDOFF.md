# Handoff

State as of `master` on 2026-09-18. The open-item list below was triaged with the owner on
2026-09-05; every item carries a decision. Items 23 and 24 were decided 2026-09-07; items 11
and 23 were finished 2026-09-16.

**Dated session write-ups moved to `docs/session-log.md` on 2026-09-16**, newest first. This
file keeps only what is still true and still open: the binding decisions and negative results
are indexed below in one line each, pointing back at the archive for the reasoning.

⚠ The "What changed recently" table still describes the tree as of `cbde8e7` and has not been
refreshed.

## Next session: suggested order

1. **Item 25 — pick a records `n` preset**, if the library is going to ship one at all. Needs
   the pooled-tune-halves re-measurement described in the item, not a lift from one task. It
   also unblocks the last piece of item 23: the README's per-domain "which parameters to use"
   table still carries `(2, 4)` for records on one task's evidence.
2. **Items 14/20 — prune-then-rescore (`rescore_n`).** The highest-value remaining code change:
   `n=(1,2)` wins every short-string task but is a poor index key, and this is the only item
   that would let V7 use it as one anyway. Real API surface work, not a probe.
3. **Item 13 — bump `__version__`**, which is no longer hygiene. `0.0.6` is already on PyPI, so
   the publish pipeline cannot succeed until it moves. As of the 2026-09-18 hand dry run it is
   the **only** thing known to be blocking a release: the build, the metadata, the
   dependency-free install, the README examples and the 3.10 floor were all verified off a real
   wheel. See `docs/releasing.md`.
4. **Hygiene whenever convenient**: 12 (README todos), 27 (BEIR caveat, already written, no
   action needed), 28 (re-measure Part 10 timings on an idle machine before quoting them
   again).
5. **Low priority / speculative, only if someone is already deep in this code**: 26 (mix
   normalization knob) and 29 (real WAND) -- both are documented well enough to pick up cold,
   neither is blocking anything else.

## Decisions and negative results that still bind

**Do not re-derive these.** Each was built and benchmarked before being rejected, or decided
against evidence that is still good. Full reasoning is in `docs/session-log.md` under the dated
entry in the last column.

| thing | standing verdict | written up under |
|---|---|---|
| `dim=2` + `denominator='geo'` as library defaults | **not shipping**, although they are worth ~79% of each task's achievable MAP against the defaults' ~76%. They break 302 tests in `tests/test_parity_with_nmd.py` and correctly so: that file pins `lookup()` == nmd, and these two knobs make it something else. Per-domain README advice instead, opt-in | 2026-09-06 |
| restricting V7's pass 2 to the pruning candidates | built, benchmarked, **rejected**. Selective exactly where the dp is rarely called: 0.9% of 30k words survive on short corpora, 90-100% on ermagellan | 2026-09-06 |
| banded dp for emd | built, benchmarked, **rejected** — reduces dp work as predicted, does not move the wall clock | 2026-09-06 |
| global grouping across n-grams | built, benchmarked, **rejected**, same shape as banding | 2026-09-06 |
| V7's pruning pass | **removed.** V7 now scores every posting exactly and partial-sorts; 1M phrases went 687 → 175 ms. The invariant is "top k equals an exhaustive scan, ties alphabetical" | 2026-09-06 |
| the cheap WAND-style bound | did not work. Real WAND is still open as item 29, and needs a compiled pivot walk | 2026-09-06 |
| a cosine-shaped V7 variant carrying V7's position term | **rejected.** Position buys at most +0.004 MAP and is flat at V7's own `position_weight=1.0`. A product-numerator, L2-normalized variant is tf-idf cosine with extra steps | 2026-09-07 |
| V7 as a reranker / RRF partner / tuned interpolant over BM25 | measured and **false**: never more than +0.005 MAP, i.e. one query. Tuned λ sits at the endpoints | 2026-09-06 (later) |
| shipping `weights` with a default of `p=2` (Part 6's recommendation) | **overruled by Part 7.** idf is opt-in, off by default: p=2 costs 8 points of hit@1 on the primary use case | `docs/bow-plan.md` part 7 |

## What `docs/bow-plan.md` parts 3-8 say (summarized 2026-09-16)

Parts 9 and 10 are covered by the 2026-09-06 and 2026-09-07 entries in `docs/session-log.md`.
Parts 3-8 landed after `cbde8e7` and had never been summarized here. Between them they are why
this library's advice is per-domain rather than global.

**Part 3 — why segment matching is a wrapper, not a change to nmd's ground cost.** The
alternative was worked out in full and rejected as a *final* score: give every n-gram occurrence
`(global_loc, word_id, within_word_offset)` and charge `min(|Δglobal|, γ + |Δoffset|)`. Because
n-grams then jump independently, a common bigram in two unrelated words matches for `γ`, and no
global control prevents it — every jump budget Lagrangian-relaxes back to a constant per-jump
price, i.e. `γ` again. The fix is *coupling* the "word i ↔ word j" decision, which a sum of
independent per-n-gram EMDs cannot express. It survives only as the *relaxed* stage of the Part 1
cascade, where it is a real upper bound on the coupled score. Also rejected: run-length bonuses
inside linear sum assignment — that is a quadratic assignment problem and LSA cannot represent it.

**Part 4 — the segment prototype, and the mass bug that set its default.** `nmd/nmd_segments.py`,
solved with `scipy.optimize.milp`. With `gain = (p+q)·sim`, token mass let junk pay for itself:
`[theater in a] <-> [theater]` scored *higher* than `[theater] <-> [theater]`. Hence the
`mass='ngrams'` default, with mass counted on the tokens rather than the concatenation — which is
what makes a perfect split/merge exactly distance 0. `min_sim=0.2` kills bigram-coincidence
matches and halves runtime. On 100 Abt-Buy queries: 0.768 hit@1 / 0.847 MRR against char-nmd(2)'s
0.697 / 0.792. Splits, merges and reordering are found in the wild; λ (the in-order preference)
does not help on product names, which reorder freely, so it is a domain choice and not a free win.

**Part 5 — the alignment machinery is not the weak part; the representation is.** Within the nmd
family the ordering is consistent across both experiments: char-nmd → bow-nmd → segment, +4 to +7
hit@1. But a plain tf-idf char-3gram cosine beats all of it — 0.890 hit@1 against the segment
metric's 0.760, at roughly 400x the speed — and on the ER-Magellan Abt-Buy split it scores 64.88
F1 from one threshold on one attribute, above the published DeepMatcher+ 62.8. Every method above
the segment metric in those tables is idf-weighted and every method below it is not.

**Part 6 — nmd is jaccard *minus* an order allowance, and idf is worth 5-15x the hyperparameters.**
Drop nmd's position term and what is left is Dice over n-gram multisets, which is rank-equivalent
to jaccard; so `normalized nmd = jaccard-family overlap − positional discount − (set→multiset)`.
The discount shaves true matches harder than non-matches (separation 0.166 → 0.152). Grid search
buys ~3 F1 on this data; idf buys +17 F1 at p=2. ⚠ The exponent does not peak here: train AP
climbs to p≈15-25, and the p→∞ limit — "the rarest shared 3-gram", with no mover's distance at
all — scores F1 73.66 on its own. Abt-Buy is substantially a model-number benchmark and a large
exponent is a roundabout rare-substring detector, so that tuning should not be expected to
transfer to names or addresses.

**Part 7 — the dictionary-lookup task reverses almost all of Part 6, and it is the task this
library is for.** 241 corrupted words against an 8000-word dictionary: `nmd n=(1,2)` 0.938 hit@1
against tf-idf cosine's 0.734 and jaccard's 0.834. Three reversals: nmd wins here; **idf hurts**
(idf¹ +0.017, idf² −0.083, idf⁶ −0.502 against plain `n=2`); and **low n wins monotonically**
(0.934 / 0.846 / 0.751 / 0.639 for n = 1/2/3/4, so the README's `(2,4)` is 11 points behind
`(1,2)`). One sentence explains the whole thing: a rare n-gram is the model number in a catalogue
and *the typo itself* in a misspelling, so weighting by rarity amplifies whichever it is. `n=1`
keeps position and discards only adjacency — one substitution destroys two bigrams and three
trigrams but a single unigram, which is exactly the observed decline.

**Part 8 — no metric wins everywhere, and the spread is enormous.** 17 metrics across product
retrieval, product matching and typo correction. The product-matching winner
(`tf-idf char-3gram idf²`) places 15th of 17 on typos, 0.560 against the leader's 0.938; the typo
winner (`nmd n=(1,2)`) places 13th on both product tasks. `nmd n=2 idf¹` is the best all-rounder
— never first, never worse than 7th — because exponent 1 is where the two domains stop
disagreeing. ⚠ Two reading traps in that table: the `sec` columns are **not** a speed benchmark
(pairwise calls with no caching, so tf-idf rebuilds both vectors every comparison), and
`tf-idf cos token` / `soft tf-idf` score ~0 on typos because they are *structurally inapplicable*
— a misspelled word is an unseen token, so its vector is empty — not because they are bad.

## Environment

Conda env named after the repo folder:
`C:\Users\user\anaconda3\envs\ngram-movers-distance\python.exe`

```bash
"C:/Users/user/anaconda3/envs/ngram-movers-distance/python.exe" -m pytest -q
# 1102 passed, 1 xfailed in 7.96s (re-measured 2026-09-16)
```

Installed and used: `numpy` 2.2.4, `scipy` 1.15.2, `pyroaring`, `regex`. `numba` 0.61.2 is
installed but **nothing in `nmd/` imports it** — it was only used to evaluate a prototype.

## Releasing

`docs/releasing.md` has the procedure and the current state. In short, as of 2026-09-18: tag
`v*` triggers `.github/workflows/publish-to-pypi.yml`, which never writes to the repository;
the workflow is pushed and active on GitHub but **has never run**; and item 13 below blocks the
first release.

A **hand dry run on 2026-09-18** exercised everything the pipeline claims to check, off a real
`python -m build` rather than the source tree — artifacts, metadata, a bare-python-3.10 install of
the wheel, every README example, and the suite on the declared floor. All green; write-up in
`docs/session-log.md`. It is repeatable: **`docs/release-dry-run.md`** is the runbook and
`experiments/release_check.py` is the script it drives. ⚠ The runbook creates two throwaway conda
envs, `nmd-release-bare` and `nmd-release-full`; its step 6 deletes them and is not optional. Two tooling facts came out of it and are recorded in `docs/releasing.md`:
`twine check` does **not** render a markdown description (metadata only), and `readme_renderer`
rewrites `#anchor` links but not relative paths — which is why the README's three relative links
became absolute in `0771414`.

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
   instead. (`__version__` is still `0.0.6` and still needs bumping before any release — that
   is the one part of this item left open, see #13.)
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

11. **Closed 2026-09-16: `emd_1d_slow` deleted.** It was dead code using
    `itertools.combinations`, i.e. a factorial worst case, and nothing imported it — the only
    two mentions left in the tree were comments, now updated. `emd_1d_old` was **not** touched
    and is not dead: `tests/test_emd_correctness.py` imports it as an oracle, and
    `experiments/emd_variants_bench.py` still excludes it deliberately as dominated by
    `emd_1d_hybrid`. Gate after the deletion: 1102 passed, 1 xfailed (2026-09-16), unchanged.
12. Pre-existing README todos, untouched: `remove()`/`discard()` on the V7 index (needs index
    compaction; `WordSet` already has them), prefix lookup, a `min_similarity` filter on
    lookup, trying cython, and `from nmd import nmd` returning a module rather than a function.
13. **`__version__` is `0.0.6` and has not been bumped** for anything since. ⚠ **Now a hard
    blocker, not hygiene**: `0.0.6` is already on PyPI and PyPI refuses re-uploads, so the
    publish pipeline cannot succeed at this version. `validate-tag` checks this and stops in
    ~20s rather than at the upload. Still not bumped unprompted, since it is only meaningful at
    release time — declined again on 2026-09-18, when the owner was asked to pick a number and
    chose not to yet. Everything else on the release path is verified as of that date, so this is
    the last thing standing between `master` and a tag. Procedure: `docs/releasing.md`.
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

21. **Closed 2026-09-06: `ermagellan` now covers 270 of the 432 grid points** (the 162
    containing `n=1` are still excluded -- 341 s each to score MAP 0.008). ⚠ The shortlist it
    replaced was actively misleading, not merely narrow: it reported `normalize=False`,
    `position_weight=0.0`, `dice` at MAP 0.7989 where the full grid gives `normalize=True`,
    `position_weight=1.0`, `geo` at **0.9039**. The shortlist had been selected from other tasks'
    rankings, so it inherited their preferences and never sampled the region ermagellan wanted.
    The six-task aggregate ranks those 270 unigram-free configurations; the five-task one ranks
    all 432, and their per-task bests differ wherever the optimum used unigrams. Do not mix rows
    between the two.

22. **Closed 2026-09-06: `idf_exponent` is not grid-limited.** Past the edge on ermagellan
    (`n=(2,3)`, geo, pw=1): 3 → 0.883, 4 → 0.895, 5 → 0.871, 6 → 0.851, 8 → 0.768 test MAP.
    It peaks at 3-4. The original note, for the record: abtbuy peaks at
    the largest value swept (3.0 → 0.9500, 4.0 → 0.9467 so it may just have turned over), and
    `docs/bow-plan.md` part 6 saw train AP still climbing at p=15. Part 6 also names three
    guards that were never implemented and would matter at high exponents: an idf floor,
    punctuation normalization, and fitting df on an external corpus.

23. **Closed 2026-09-07: narrow the pitch, do not chase records parity.** Part 10 says:
    best-in-set for fuzzy lookup of short strings, a wash at 5-10x the cost on records, behind
    BM25 on documents. Item 24's probe forecloses the one redesign that might have changed that
    verdict, so the decision is to narrow rather than keep trying. Reasoning in the 2026-09-07
    entry of `docs/session-log.md`.

    **Done 2026-09-16: the README edit landed.** A "What this is good for, and what it is not"
    section now sits directly under the title, giving the three regimes with Part 10's numbers
    and the date they were measured, plus the "not a useful complement either" result (≤ +0.005
    MAP as reranker / RRF partner / interpolant). ⚠ **Still owed, and deliberately deferred to
    item 25:** the per-domain "which parameters to use" table further down still carries
    `(2, 4)` for the records row on one task's evidence. It was left alone rather than edited
    twice, so item 25 now owns it.

24. **Closed 2026-09-07: the cosine-shaped variant does not beat cosine, and position does not**
    **rescue it.** ermagellan decomposition (Part 10): `min(c_q, c_d)` vs the product costs
    −0.036, L1-geo vs L2 idf-weighted normalization −0.01..−0.04, position +0.008 -- structural,
    but left one question open: would a V7 variant with a product numerator, an L2 norm and
    V7's position term beat plain cosine? Probed in `.scratch/probe_cosine_position.py`
    (self-checked to reproduce `TfidfCosineSystem` at 2.2e-16, so the numbers below are not an
    artifact of a mismatched baseline): on ermagellan, position buys at most +0.004 test MAP
    over no-position (inside the noise this task already showed at 103 queries), and at V7's
    own `position_weight=1.0` it is flat, not positive. Full numbers and the decision are in
    the 2026-09-07 entry of `docs/session-log.md`. **No further design work planned here** -- this was the
    one variant that could have escaped the Part 10 tradeoff, and it does not.

25. **A single "records" configuration has not been chosen.** The selected `n` swings across
    `(2,)`, `(3,)`, `(2,3)`, `(2,4)`, `(3,4)` on the ten Magellan tasks, on ≤150 tune queries
    each. If the library ships a records preset it needs to be picked on the pooled tune halves
    and re-measured, not lifted from one task.

26. **`mix` normalizes by per-query max**; a z-score or top-50 min-max might behave differently
    at λ ≈ 0.5. Low priority — λ chose the endpoints on every task — but it is the one untried
    knob in the "position as tie-breaker" test.

27. **Document tasks are measured without stemming/stopwords and on 150-query halves**, so the
    `bm25_word` numbers are a rough BEIR reproduction (0.665 on scifact exact, 0.294 vs 0.325
    on nfcorpus). Fine for a negative boundary; do not cite them as BEIR-comparable.

28. **Re-measure Part 10's ms/query on an idle machine** before quoting any of them. All were
    taken with 2-3 benchmark processes sharing the laptop.

29. **Real WAND over sorted postings** -- the only idea left that would change the 1M-phrase
    ceiling, and the one code item on this board with no cheaper substitute. Written up in full
    under "⚠ Open: real WAND, since the cheap one did not work" in the 2026-09-06 entry of
    `docs/session-log.md`; this row exists so it is
    ranked rather than buried in prose. Two preconditions, both hard: it can only be measured
    against `experiments/large_vocab_bench.py` (on the small benchmarks the whole vocabulary
    fits in cache and the numpy pass is ~1 ms, so WAND cannot win there), and the pivot walk
    has to be compiled, which the dependency-free constraint makes a V8-shaped project rather
    than a V7 patch. **A python-loop prototype proves nothing and should not be built.**

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
* **V7 has no pruning pass any more** (removed 2026-09-06; see "Large vocabularies, second
  round" in the 2026-09-06 entry of `docs/session-log.md`). It scores every posting exactly and partial-sorts the vocabulary-sized score
  array. The invariant is now simply "the top k equals an exhaustive scan, ties alphabetical",
  pinned by `tests/test_index_v7.py::TestScoring::test_top_k_matches_an_exhaustive_scan` and
  `test_ties_at_the_top_k_boundary_are_ordered_alphabetically`. The two-sided count bound
  (`[min(c1,c2), 2*min(c1,c2)]` per shared n-gram) is still true and is still what keeps
  `position_weight` inside `[0, 1]`; it just no longer drives anything.
