# Session log

Dated write-ups, newest first. Split out of `HANDOFF.md` on 2026-09-16 so that file keeps only
what is still true and still open; nothing here was edited in the move. `HANDOFF.md` indexes the
decisions and negative results that still bind, and points back here for the reasoning.

---

## Session 2026-09-18: release dry run against a genuinely bare environment

No scoring code changed. The publish pipeline has never run (see `docs/releasing.md`), so
everything it claims to check was checked here by hand first, off a real build rather than off
the source tree.

**The artifacts build and the metadata is right.** `python -m build` (flit_core 3.12.0) produced
`nmd-0.0.6-py3-none-any.whl` and `nmd-0.0.6.tar.gz`; `twine check --strict` passes both. METADATA
carries **no `Requires-Dist` at all**, `Requires-Python: >=3.10`, `Description-Content-Type:
text/markdown`, `License-File: LICENSE`, and a Summary lifted from the `nmd/__init__.py`
docstring — i.e. `dynamic = ["version", "description"]` resolves the way `docs/releasing.md`
describes.

⚠ **The sdist ships the package and nothing else**: `nmd/*.py`, `pyproject.toml`, `README.md`,
`LICENSE`, `PKG-INFO`. No `tests/`, no `docs/`, no `experiments/`. So the README's "# Testing →
`pytest`" is a *repository* instruction, not something an sdist download can act on, and
`docs/bow-plan.md` is only ever reachable over the network. That is flit's default behaviour and
was left as is, but it is the reason the README links had to become absolute (below).

**The dependency-free promise holds on the built wheel, not just the tree.** A throwaway conda
env `nmd-release-test` at python **3.10.21** — the declared floor — holding only pip, setuptools,
wheel and packaging, with `numpy` / `scipy` / `pyroaring` / `regex` / `numba` asserted absent
before anything else ran. Installing the wheel alone: `import nmd` leaks none of the five into
`sys.modules`, and `nmd.__all__` is exactly `('ngram_movers_distance', 'WordList')`.

**The README's dependency table is accurate, checked by provoking the failures.** In that bare
env each optional module raises `ImportError` naming a package its README row lists:
`nmd_index_v7` → `numpy`, `nmd_bow` → `scipy`, `nmd_segments` → `numpy` (of `scipy`, `numpy`),
`nmd_word_set` → `pyroaring` (of `pyroaring`, `regex`). Installing those four then makes every
one of them work off the installed wheel.

**Every README code block was run, and every number in them reproduces** (2026-09-18, against the
installed 0.0.6 wheel, not the source tree): the four-word quickstart matches its commented output
term for term; over `experiments/words_en.txt` `WordList.lookup('fotografer')[0]` is
`('photographer', (9.9, 13.87))` and `lookup('beaurocracy')[0]` is `('bureaucracy', (11.73,
17.45))`; `ApproxWordListV7.lookup('fotografer')[0]` is `('photographer', 0.437)`; the bow,
segment and `WordSet` blocks all run. `experiments/readme_example_check.py` is green — nothing
within Damerau-Levenshtein 2, 65 within 5.

**The suite passes on the declared floor.** 1102 passed, 1 xfailed in 10.39s on python 3.10.21;
the repo env is 3.12.14, so `>=3.10` had never actually been exercised. Still only two of the
**eleven** cells the release matrix declares — 2 OSes × 5 pythons plus macos/3.12, counted off the
workflow on 2026-09-18. (An earlier draft of this entry said "seven"; that was never measured.)

**Two facts about the tooling worth not rediscovering:**

* ⚠ **`twine check` does not render a markdown description.**
  `twine/commands/check.py::_RENDERERS` maps `"text/markdown": None` with the comment
  "Rendering cannot fail", so for this package it validates *metadata only* — under `--strict`,
  and regardless of whether a markdown backend is installed. The workflow step is named "Check
  the metadata", which is accurate; do not read it as a README-rendering guard.
* **`readme_renderer` rewrites `#anchor` links but not relative paths.**
  `readme_renderer/markdown.py::_prefix_relative_links` only touches `href="#..."`, to match the
  heading ids it prefixes — and `readme_renderer/clean.py` keeps ids (`"*": {"id"}`), so the
  README's `[what this is good for](#what-this-is-good-for-and-what-it-is-not)` link **does** work
  on PyPI. A relative *path* is left alone and resolves against `https://pypi.org/project/nmd/`.
  Hence `0771414`: `docs/bow-plan.md` (×2) and `tests/README.md` now point at
  `github.com/averykhoo/ngram-movers-distance/blob/master/...`, which works from both places.
  The same commit dropped the `from nmd import nmd should return a function, not a module` todo —
  there is no `nmd.nmd` submodule any more, so that import raises `ImportError` and the todo
  describes a problem that no longer exists.

**It is a runbook now, not a one-off.** The checks first lived only in `.scratch/`, which is the
documented way to lose them; they are tracked as **`experiments/release_check.py`** with
**`docs/release-dry-run.md`** as the runbook. Two modes, because they need contradictory
environments — `bare` asserts numpy / scipy / pyroaring / regex are absent, `full` that they are
present — so the runbook builds two throwaway conda envs (`nmd-release-bare`, `nmd-release-full`)
and runs them in parallel. Both were deleted at the end of this session, and step 6 of the runbook
says to do that.

Three properties are worth keeping if that script is ever edited. It **refuses to run against the
source tree** (`import nmd` must resolve outside the repo), since a file missing from the wheel is
invisible from a tree that still has it. It **asserts its environment before it checks anything**,
so `bare` cannot pass vacuously in an env that happens to have numpy. And every expected number is
also **grepped for as a literal in README.md**, so editing the README without re-measuring goes red
rather than quietly diverging.

Sabotage-checked rather than just run green: perturbing the expected V7 score `0.437` → `0.438`
fails the measurement; deleting `regex` from the README's `nmd_word_set` dependency row fails the
README pin while the import check stays green — which is exactly the drift that pin exists for; and
running `bare` in the full env fails the environment assertion. Run with no wheel installed it
fails `nmd is importable` and stops instead of falling back to the tree.

**General CI now exists** — `.github/workflows/ci.yml`, added the same day. Until then the only
trigger in the repository was `push: tags: v*`, so the matrix could only ever run *as part of a
release*, which is the worst moment to find a red cell. Scope was chosen deliberately small:

| trigger | what runs |
|---|---|
| push, any branch | the suite on ubuntu / **3.10** |
| `master` and PRs | + ubuntu / **3.14**, + the dependency-free install check |

The two pythons are the **ends** — the floor catches a 3.11+ feature sneaking in, the newest
catches a stdlib removal or a dependency without a wheel yet; middle versions rarely break alone
and stay on the release path. ubuntu-only because `nmd` ships no compiled code and this repo is
developed on Windows, so windows/3.12 is already exercised continuously by the local gate — ubuntu
is the platform nothing else covers. The full eleven-cell matrix still gates the upload.

⚠ **`verify-minimal` was extracted into `template-verify-minimal.yml`** rather than copied into
`ci.yml`. Two inline copies of the dependency-free check would be a check that fails by passing
the moment one drifts; `publish-to-pypi.yml` now calls the same file. Its script also gained an
`__all__` assertion while being moved.

**Workflow YAML cannot be run locally, so it was linted instead.** `actionlint` (via `actionlint-py`,
which ships the binary — it installs as `Scripts/actionlint.exe`, not an importable module) is
clean across all four workflows. Sabotage-checked rather than trusted: it catches an input the
called workflow does not declare, and a `uses:` pointing at a file that does not exist, returning
to rc 0 on restore. Those are the two ways the four-file split can break. Commands are in
`docs/release-dry-run.md`. ⚠ A hand-rolled python cross-check written alongside it reported five
false "missing workflow" failures, from `uses.lstrip('./')` — `lstrip` strips *characters*, so it
ate the dot of `.github`. The lesson is the ordinary one: the tool that knows the schema beat the
five-minute script, and the script's failures were in the script.

**Still blocking a release: item 13.** `__version__` is still `'0.0.6'`, which is on PyPI. Left
unbumped on purpose — the owner declined to pick a number in this session, and it is only
meaningful at release time.

## Session 2026-09-16: item 11 and item 23 landed, and this file split off

Documentation and hygiene only; no scoring code changed.

**Item 11 — `emd_1d_slow` deleted from `nmd/emd_1d.py`.** Nothing imported it; the only two
references in the tree were comments (`experiments/correctness-test.py`, a commented-out call and
a stale "assume it exists" note; `experiments/emd_variants_bench.py`, its excluded-variants list),
both updated. `itertools` stays imported because `emd_1d_old` still uses it at what is now
line 558. ⚠ `emd_1d_old` was deliberately **not** deleted in the same pass — it is the oracle in
`tests/test_emd_correctness.py`, so being slow and obviously correct is its job. Gate after:
1102 passed, 1 xfailed in 7.96s, identical to the 2026-09-06 count.

**Item 23 — the README pitch is narrowed.** A "What this is good for, and what it is not" section
now sits directly under the title, before "Why another string matching algorithm?", carrying
Part 10's three regimes with numbers and the 2026-09-06 measurement date: short strings *use it*
(+0.07 to +0.28 MAP, 1-7 ms/query, ~100x faster than a brute-force edit-distance scan), records
*probably not* (±0.02 on nine of ten tasks, −0.057 on the tenth, 5-10x the cost), documents *no*
(nDCG@10 0.650 vs 0.679 on scifact; 0.029 at the library defaults). It also states the negative
complement result explicitly — as reranker, RRF partner or tuned interpolant over BM25 it never
adds more than +0.005 MAP — since that is the claim a reader is most likely to assume still holds.

⚠ **The per-domain "which parameters to use" table was deliberately left alone.** Item 23's own
instruction was to fold item 25's records-preset decision in at the same time so the table is not
edited twice, and item 25 is still undecided (the selected `n` swings across `(2,)`, `(3,)`,
`(2,3)`, `(2,4)`, `(3,4)` on ten Magellan tasks at ≤150 tune queries each). The table still shows
`(2, 4)` for records on one task's evidence. **Item 25 now owns that edit**, and the suggested
order at the top of `HANDOFF.md` says so.

**`docs/bow-plan.md` parts 3-8 are now summarized in `HANDOFF.md`**, closing the ⚠ that had sat
in its banner since Part 9. Parts 9 and 10 were already covered by the two entries below. The
summary is six paragraphs, one per part, and its load-bearing claim is Part 7's: a rare n-gram is
the model number in a catalogue and the typo itself in a misspelling, which is why idf helps on
records and hurts on typos, and therefore why the library's advice is per-domain.

**`origin/experiments` deleted.** Its tip was `b39f9d9 "docstrings"` (Avery, 26 Jul 2022), an
ancestor of `master` 67 commits back, sitting just after `19fc212 move stuff around`. Verified
contained with `git merge-base --is-ancestor` immediately before the push, not just at survey
time. Nothing unique was on it; the tip sha is recorded here in case the label is ever wanted
back. ⚠ `origin/pypi-oidc-attestations` was **not** deleted and should not be: `git cherry` marks
its one commit `+`, and `.github/workflows/publish-to-pypi.yml` (81 lines) exists nowhere else in
the repo -- `master` has no `.github/` at all. That workflow is what item 13's version bump is
for.

⚠ **Two wrong test counts corrected in `tests/README.md`.** Its header claimed
`1052 passed, 1 xfailed` as of 2026-09-06, and its per-file table listed `test_index_v7_knobs.py`
at 60 against an actual 92. Not drift: the corrected table sums to exactly 1103, which is what
`--collect-only` reports, while 1052 never matched even that file's own table (1071 with the bad
row). Both numbers were written down rather than re-run, in `eeffc03`. The header now carries
collected as well as passed, so the next person can check the sum without running anything.

**This file exists as of today.** `HANDOFF.md` had grown to 803 lines, ~530 of them dated session
narrative, which buried the open board. The three dated entries moved here verbatim — no edits in
the move — and `HANDOFF.md` gained a "Decisions and negative results that still bind" table that
indexes each rejection in one line and points back here. ⚠ That table is the part worth
maintaining: the value of the negative results (banded dp, global grouping, candidate
restriction, the cheap WAND bound, the cosine-shaped variant) is entirely in being seen *before*
someone rebuilds one, and an archive nobody opens at session start does not do that.

## Session 2026-09-07: cosine + position probe closes items 23 and 24

Item 24 asked whether a V7 variant shaped like tf-idf cosine (product numerator instead of
`min`, L2 norm instead of L1-geo) but with V7's position term bolted on could beat plain cosine
on records -- if it could, that would be a reason to keep chasing records competitiveness
instead of narrowing the pitch (item 23). Probed in `.scratch/probe_cosine_position.py`
(gitignored, transcribed here in full since it will not survive a clear).

**Self-check, because a wrong reference corner would make every position number meaningless.**
The scorer's `pw=0, product-numerator, log-tf, L2, idf²` corner is required to reproduce
`search_benchmark.TfidfCosineSystem` exactly. First run: worst diff **0.69** -- the L2 norm was
squaring `tf * idf^E`, i.e. accumulating `idf^2E` instead of `idf^E`. Fixed (accumulate
`idf^E * tf^2`, not `(idf^E * tf)^2`); reproduces to **2.2e-16** on both a smoke task
(fodors_zagats) and ermagellan. The other corner (`min`-numerator, raw-tf, L1-geo) matches
`.scratch/probe_ermagellan_gap.py`'s `MinVariant` from Part 10's own decomposition, so both ends
of the axis are cross-checked against code written in a separate session.

**Result, ermagellan, n=(2,3), 103 test queries, idf swept 1-6, best of each cell selected on
tune, test MAP reported:**

| numerator | norm | position weight | test MAP |
|---|---|---:|---:|
| product | L2 | 0.0 | 0.9387 |
| product | L2 | 0.5 | **0.9426** |
| product | L2 | 1.0 (V7's own setting) | 0.9386 |
| product | L2 | 2.0 (past what V7 allows) | 0.9281 |
| min (V7's numerator) | L2 | 0.0 | 0.9295 |
| min | L2 | 1.0 | 0.9223 |
| min | L1-geo (V7's denominator) | 0.0 | 0.9222 |
| min | L1-geo | 1.0 | 0.9174 |

Position buys at most **+0.004 MAP** (pw=0.5 over pw=0, product/L2) -- inside the noise Part 10
already established at this query count (the per-n-vs-pooled comparison swung ±0.030 on the same
103 queries) -- and at **V7's own `position_weight=1.0` it is flat, not positive**, and clearly
negative at 2.0. Switching `min`→`product` and L1-geo→L2 recovers most of the gap on its own
(0.87-0.90 range in Part 10's decomposition → 0.94 here, partly because this probe's idf grid
runs to 6 instead of 3); position adds nothing on top of that fix, in either direction that
would matter.

**Decision: item 24 is closed with "no, this does not close the gap."** A product-numerator,
L2-normalized variant is just tf-idf cosine with extra steps -- building it would not be a new
capability, and V7's positional term does not improve it. There is no remaining design that
keeps V7's transport-distance property (the `min` numerator, which is what makes it EMD rather
than cosine) while also matching cosine's records performance; the two are the same tradeoff
Part 10 already identified, and this probe closes off the one variant that might have escaped
it.

**Item 23, closed on the strength of that: narrow the pitch, do not keep chasing parity.**
`README.md`'s intro (lines ~5-18) still pitches general text search; Part 10 measured
best-in-set for fuzzy short-string lookup, a wash at 5-10x the query cost on records, behind
BM25 on documents. **Not yet done**: the actual README edit. Whoever does it should fold in
item 25's records-preset decision at the same time, so the per-domain table in the README
(currently around the "which parameters to use" section) is not edited twice.

## Session 2026-09-06 (later): V7 against BM25 / tf-idf / edit distance, 17 tasks

`experiments/search_benchmark.py` is the answer to "is this useful for text search". Tune/test
halves per task, every system tuned on tune and reported on test, `top_k=50`, V7's tie rules
applied to all. Written up as `docs/bow-plan.md` **Part 10**; tables in
`experiments/results/search_benchmark_summary.md` (regenerate with `--report`).

**Three regimes.** Short strings (typo ×4, norvig): V7 wins by +0.07 to +0.28 MAP over the best
of char-BM25 / tf-idf / Damerau-Levenshtein, at 1-7 ms/query, position term on. Records
(10 Magellan tasks): within ±0.02 of the best baseline on 9, −0.057 on ermagellan, never
clearly ahead, 5-10x slower. Documents (scifact, nfcorpus): behind BM25 (nDCG@10 0.650 vs
0.679), and only close with `normalize=False`, `n=(4,)`; the default scores 0.029.

**Position helps on short strings (+0.09 typo_brutal) and is noise (±0.02) elsewhere.**
**As a reranker / RRF partner / tuned interpolant on top of BM25 it never adds more than
+0.005 (one query)** -- rerank of a top-50 with recall 1.0 is just the reranker; tuned λ sits
at the endpoints. The weak-form "at least a tie-breaker" hypothesis is measured and false for
this score.

**What V7 lacks vs cosine / BM25 on records is the numerator, not position** (ermagellan
decomposition, Part 10): `min(c_q, c_d)` instead of the product `c_q·c_d` costs −0.036, L1-geo
instead of L2 idf-weighted normalization −0.01 to −0.04, per-n averaging vs pooling ≈ noise;
the position term is +0.008. `min` is what makes the score a transport quantity, so this is
structural.

⚠ **Two protocol lessons.** (1) Textbook BM25 (idf¹) was an under-tuned baseline; tf-idf
cosine beating it exposed that the dot product carries idf². With `idf^{1,2,3}` in the grid
char-BM25 gains up to +0.055 on ermagellan and V7 goes from "ahead of BM25 by 0.016" to
"behind by 0.054". Baselines need the same tuning budget as the system under test. (2) On
1 500-char documents `n=2` costs 550 ms/query for nDCG 0.30 — excluded from the document
tier (`V7_N_DOC`) rather than spend 8 hours on it.

Timing caveat: all ms/query in Part 10 were measured with 2-3 benchmark processes sharing the
laptop; ratios on a task hold, absolute values do not. `.scratch/probe_ermagellan_gap.py`,
`probe_bm25_vs_tfidf.py`, `probe_scifact.py` produced the decomposition numbers; they are
transcribed into Part 10 because `.scratch/` is gitignored.

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

**The headline result: the tasks split into two clusters, differing on `n`, `idf_exponent`
and `dim` -- but agreeing on `normalize=True`, `position_weight=1.0` and (mostly) `geo`.**

| cluster | tasks | `n` | idf | dim | normalize | position_weight | denominator |
|---|---|---|---|---|---|---|---|
| short strings | typo, typo_ms, typo_hard, typo_brutal | (1, 2) | 0-0.5 | 2 | True | 1.0 | either |
| product matching | abtbuy, ermagellan | (2, 3) / (2, 4) | 3.0 | 1 | True | 1.0 | **geo** |

⚠ **Corrected 2026-09-06, after ermagellan's full grid finished.** The product row previously
read `normalize=False`, `position_weight=0.0`, `dice`, and this section previously claimed the
clusters "disagree on every knob". Both came from ermagellan's **19-configuration shortlist**,
which did not contain the winning combinations. On the full 270-point grid ermagellan's best is
`n=(2,4) idf=3.0 dim=1 normalize=True position_weight=1.0 geo` at MAP **0.9039**, against 0.7989
for the shortlist's answer (which re-measures at 0.79894 there, so the runs agree).

What actually holds: **all six tasks want `normalize=True` and `position_weight=1.0`.** The
clusters differ only on `n`, `idf_exponent` and `dim`. The two product tasks also agree with each
other much better than reported -- ermagellan's winner scores 0.9088 on abtbuy against abtbuy's
own best of 0.9500, while abtbuy's winner scores only 0.7989 on ermagellan.

⚠ **`geo` is not a minor knob.** Paired against `dice` at `normalize=True`: **+0.0991 MAP on
ermagellan, 90/90 comparisons**, +0.0159 on abtbuy (139/144), ~neutral on the four typo tasks
(+0.0043, +0.0060, +0.0010, -0.0033). Helps most where length mismatch is largest, costs nothing
elsewhere.

**Lesson worth keeping: do not draw parameter conclusions from a shortlist selected by other
tasks' rankings.** The shortlist was built from the 4-task aggregate plus per-task winners, so it
inherited their preferences and never sampled the region ermagellan actually wanted.

Best single compromise depends on which field you rank over. Over the five-task **full** grid it
is `n=(1,2) idf=0.5 dim=2 normalize=True position_weight=1.0 denominator='geo'` (best mean rank,
51.2/432, worst 234/432 on abtbuy). Over the 270 unigram-free points all six share it is
`n=(2,4) idf=1.0 dim=2 normalize=True position_weight=1.0 denominator='geo'` (mean normalized MAP
0.881, mean rank 74.8, worst 151/270). **The pre-session default `n=(2,4) idf=0 dim=1
normalize=False` ranks 334 / 161 / 99 / 94 / 98 of 432 across the five full-grid tasks, and
256/270 and 255/270 on the two product tasks**: not a compromise, just unchosen.

### Recommended library defaults (evidence, 2026-09-06)

Scored as "mean fraction of each task's own best MAP" over all six benchmarks:

| candidate default | abtbuy | ermagellan | typo | typo_brutal | typo_hard | typo_ms | mean % |
|---|---|---|---|---|---|---|---|
| pre-session (`normalize=False`, dice) | 0.7238 | 0.3018 | 0.8578 | 0.1557 | 0.7292 | 0.8636 | 69% |
| current (`normalize=True`, dim 1, dice) | 0.7894 | 0.6408 | 0.8690 | 0.1559 | 0.7090 | 0.8562 | 76% |
| **+ `dim=2` + `denominator='geo'`** | **0.7959** | **0.7186** | **0.8843** | **0.1691** | **0.7279** | **0.8740** | **79%** |
| also `idf_exponent=1.0` | 0.8759 | 0.8102 | 0.8791 | 0.1556 | 0.7289 | 0.8464 | 81% |

Row 3 beats the current defaults on **all six benchmarks** -- a Pareto improvement, nothing
traded. Row 4 scores higher on average but loses on `typo_ms` (0.8464 vs 0.8562) and
`typo_brutal`, and would give up `idf_exponent=0.0`'s guarantee that the unweighted path stays
bit-for-bit, so it is *not* recommended as a default even though it ranks best on the mean.

⚠ `n` is deliberately left at `(2,4)` in all of these. `(1,2)` scores better on every typo task
but is a poor index key (see item 14), so it belongs in per-domain advice rather than a default.

**Decided 2026-09-06: NOT shipping `dim=2` / `denominator='geo'` as defaults.** Changing them
breaks 302 tests in `tests/test_parity_with_nmd.py`, and correctly so -- that file pins *"V7's
index score already is the exact nmd similarity"*, i.e. `nmd_per_n_mean`, the arithmetic mean over
n of `ngram_movers_distance` under Dice normalization. `dim=2` makes it a quadratic power mean and
`geo` changes the normalization, so `lookup()` would stop computing n-gram mover's distance by
default. That is categorically different from the `normalize` flip, which stayed inside nmd's own
API. The tuned settings live in the README table and Part 9 as per-domain advice instead; callers
opt in. ⚠ Anyone revisiting this should note the cost is real -- defaults give ~76% of each task's
achievable MAP against ~79% for `dim=2` + `geo`.

### Large vocabularies: what actually costs time

Measured 2026-09-06 on a 300 000-word index drawn from `british-english-insane.txt` (470 351
words survive the `isalpha`, 3-15 char filter):

* lookup is **O(vocabulary), by design**. Every lookup allocates and operates on several
  300k-element arrays -- `np.bincount(minlength=vocab_size)` per n, an `astype`, the normalizing
  division, the power mean, and the `touched` / `candidates` masks -- on top of O(postings)
  scatter-adds.
* **39.6% of a 300k vocabulary gets a nonzero score** at `n=(2,4)`; bigrams are not selective at
  that scale. Only **0.7% (2 086 words) survived the pruning bound** -- which turned out not to
  matter, see the second round below: the bound was removed because computing it cost more than
  it ever saved.
* `dim=2` and `denominator='geo'` are what make the tuned settings slower at scale: both add
  vocabulary-sized math. At 300k, `d1/dice` 25.0 ms/query against `d2/geo` 39.4 ms.

**Landed (bit-identical, 1.13x on the default path at 300k):** `_denominator` was computed
**twice per n per lookup** with identical arguments -- once for the bound, once for the score --
and it is a vocabulary-sized array op. Now computed once. And `_generalized_mean` used
`parts[0] ** dim`; at `dim == 2` a plain multiply is bit-identical and 1.6x faster on a 200k
array (0.51 ms -> 0.32 ms). `** 0.5` was left alone: bit-identical to `np.sqrt` but marginally
faster already.

⚠ **Rejected: precomputing `sqrt(totals)` for the geo denominator.** `sqrt(a*b)` and
`sqrt(a)*sqrt(b)` differ by ~2.7e-16, so it would trade exactness for speed. Not taken.

**Landed: precomputed `sqrt(totals)` for the geo denominator, 1.20-1.24x.**
`sqrt(total * query_total) == sqrt(total) * sqrt(query_total)`, so the per-word half is computed
once in `_freeze` and the lookup does an array-scalar multiply instead of a vocabulary-sized
sqrt. At 300k: `d1/geo` 35.45 -> 29.45 ms, `d2/geo` 38.41 -> 30.98 ms; the dice path is untouched.

⚠ **This is the one deliberate exactness trade in the whole line of work** -- the two forms differ
by ~2.7e-16 relative. It was measured, not argued: across 11 configurations on typo, typo_hard,
abtbuy and ermagellan, **MAP and NDCG@10 are identical to five decimal places in every one**. The
only effect is 9 tie-order flips across 660 ranked lists, none of which move a metric, because
reordering only happens where scores were already tied.

### ⚠ Restricting pass 2 to candidates: built, benchmarked, rejected 2026-09-06

Pass 2 scores every posting although only the `candidates` mask can reach the top k -- 0.7% of a
300k vocabulary. Restricting all four pass-2 branches to candidates is bit-identical (192 lookups,
0 ranking changes, worst score difference 0.0) and **still not worth it**:

| task | docs | ratio |
|---|---|---|
| typo_hard | 30 000 | 1.09x |
| huge300k | 300 000 | **1.05x** |
| ermagellan | 1 035 | 0.96x |
| typo | 8 000 | **0.85x** |
| abtbuy | 1 079 | **0.81x** |

The hypothesis was that scattering into a 2.4 MB score array thrashes cache while scattering into
~16 KB of candidates would not. That was wrong in a specific way worth remembering: the mask
`candidates[posting_idxs]` is *itself* a random-access gather into a vocabulary-sized array, so it
has the same cache behaviour as the scatter it replaces. The cost moved rather than disappeared,
and on small corpora the extra pass is pure loss.

### Two ideas that do not apply here

* **Log space** (turn multiplies/divides into adds). The workload is addition-dominated --
  scatter-adds, `np.bincount`, accumulation across n-grams -- and log space turns *adds* into
  log-sum-exp. It would make the dominant operation several times more expensive to save ~4
  vocabulary-sized divisions per lookup, and scores are legitimately 0 for most words.
* **Caching the dp.** Largely moot since the `reduceat` change: the 1-vs-m branch no longer calls
  the dp, and 98.2% of what remains is equal-mass and closed-formed, so `typo_hard` at `n=(2,4)`
  sends *one* pair to the dp. It would only matter for long documents, and item 4 already records
  a cross-call cache of the same shape measuring 1.2x and being rejected as a hidden global.

### The pattern, now three for three

`banded dp`, `global grouping` and `restricting pass 2` each reduced work by exactly the predicted
amount and each failed to move the wall clock. The two changes that *did* pay -- `reduceat` on the
1-vs-m branch (2.4-2.5x) and the duplicate `_denominator` (1.13x) -- were both found by profiling
and were both in code nobody had looked at, not in the algorithm everyone assumed was hot.
**Profile first; do not reason from operation counts.**

### Large vocabularies, second round: 1M phrases, and the pruning pass is gone (2026-09-06)

The 300k run above was single words. `experiments/large_vocab_bench.py` (new, tracked -- the
300k run had no script behind it) indexes **1 000 000 phrases of 2-5 dictionary words** (mean
34.5 chars) at `n=(2,4)`, queries with 1-3 character edits and, every third query, a dropped
word. This is the regime the owner called "near pathological, where you'd use elasticsearch":
the space character makes `"e "`-type bigrams repeat within a document, so 8% of n=2 postings
(2.62M, 2.10 locations each) land on the `_multi` path the index was never tuned for, and a
query touches **95.4% of the vocabulary**. Build 169 s, RSS 155 → 4035 MB.

| variant (1M phrases, d1/dice, 60 queries, min of 1) | mean | median | max |
|---|---:|---:|---:|
| HEAD `8c03195` | 687 ms | 544 ms | 2493 ms |
| + `_multi_groups` built in `_freeze` | 381 ms | 360 ms | 768 ms |
| + pass 1 removed, one `bincount` per n (landed) | **175 ms** | 161 ms | 357 ms |

`d2/geo` tracks it at 188 ms mean. MRR 0.992 throughout (the task is speed-only; it does not
discriminate quality). Two changes, each found by cProfile, each bit-identical to HEAD
(`.scratch/check_multi_groups.py`: 2016 lookups over 4 n-lists × 2 idf × 4 lookup configs with
interleaved `add_word`, 0 ranking differences, worst score difference 0.0):

1. **80% of the HEAD lookup was python loops over multi-postings** -- the pass-1 count
   correction and the pass-2 `by_count` grouping, 1.95M iterations per 20 lookups. Grouping
   by occurrence count now happens once in `_freeze` (`_multi_groups`, alongside the 4-tuple
   `_multi_flat`) and the lookup iterates groups, not postings. 1.8x.
2. **Pass 1 (the count-based pruning bound) is removed.** Stage timing at 300k phrases put it
   at 21 of 78 ms (27%): concatenate, bincount, astype, a vocabulary-sized divide, a power mean
   and a partition -- a second full pass over the postings whose only effect was to shrink the
   final partial sort, since "restricting pass 2 to candidates" had already been measured at
   0.81-1.09x and rejected. Scoring is now one weighted `np.bincount` per n over the
   concatenated `(word index, contribution)` parts of every query n-gram, with the 1-vs-1
   arithmetic done in place. Bit-identity argument: bincount accumulates sequentially in array
   order, the parts are concatenated in query n-gram order, and `2.0 - w*d == (-(w*d)) + 2.0`
   in IEEE. 300k phrases, separate processes, min of 2: **179 → 43 ms d1/dice, 186 → 46 ms
   d2/geo (4.2x)**; 300k single words 29.5 → 17.4 ms, 36.9 → 21.4 ms (1.7x).

**One behaviour change, deliberate:** the top-k selection is now tie-inclusive at the k-th
score, then sorted `(-score, word)` and truncated, so a boundary tie is always resolved
alphabetically and `lookup(top_k=k)` is a prefix of `lookup(top_k=len(index))` for every k.
HEAD's `argpartition` split boundary ties arbitrarily (38 of 120 top-10 lookups on 300k words
differ, all score-identical; `test_top_k_agrees_with_a_full_scan` in `test_index_v7_idf.py`
documented the old `'hubz'` case and now asserts the words too). New pin:
`tests/test_index_v7.py::test_ties_at_the_top_k_boundary_are_ordered_alphabetically`,
verified red on HEAD's index.

**What is left at 1M** (cProfile, 20 queries, 27k function calls total -- the python loops are
gone): 70% of the time is array arithmetic inside `_similarity_vectors` itself over roughly 5M
postings per query (1.43M measured at 300k phrases, and it scales with the index), `reduceat` 9%,
`_denominator` 5%, `_emd_1d_batch` 4%, `argpartition` 3.5%. The remaining cost is memory
bandwidth over postings and vocabulary-sized arrays, i.e. the O(postings + vocabulary) design.
Getting under ~50 ms at 1M would need a different index (n≥3 keys so bigrams stop touching 95%
of the vocabulary, or WAND-style upper-bound skipping over sorted postings), not more numpy.

The pattern held a fourth and fifth time: both wins were in code found by profiling (python
loops the 300k single-word profile never exercised, because single words have almost no
repeated n-grams), and the "sound but useless" pruning bound was a cost, not a saving.

**The removal helps the small single-word benchmarks too** (2026-09-06, `.scratch/ab_v7.py`,
`HEAD~1` = `8c03195` vs `de621c2`, separate processes, min of 2, `top_k=50` as the eval uses,
d1/dice; d2/geo tracks within 5%). Rankings differ only at boundary ties, verified with
`.scratch/check_task_identity.py` (960 lookups per task, 0 ranking differences, worst score
difference 0):

| task | docs | n=(2,4) | n=(1,2) |
|---|---:|---:|---:|
| typo | 8 000 | 0.30 → 0.19 ms (1.6x) | 1.37 → 0.39 ms (3.5x) |
| typo_hard | 30 000 | 2.2 → 0.97 ms (2.3x) | 17.0 → 2.7 ms (6.3x) |
| abtbuy | 1 079 | 3.1 → 1.2 ms (2.6x) | 36 → 16 ms (2.3x) |
| ermagellan | 1 035 | 97 → 41 ms (2.4x) | not run (unigrams on 138-char docs: 341 s/config) |

The n=(1,2) column is the multi-posting grouping at work: unigrams repeat inside almost every
word, so that configuration always lived on the python-loop path the 300k single-word profile
at n=(2,4) never showed. `index_param_search.py` grids get correspondingly cheaper.

### ⚠ Open: real WAND, since the cheap one did not work

The count-based pruning bound was meant as a cheaper WAND -- a per-word upper bound from
counts alone, no sorted postings, no per-term maxima. It failed for a reason that is worth
stating precisely: **it only ever bounded, it never skipped.** WAND's saving comes from
*not reading* postings below the current threshold; the count bound was computed by reading
every posting once and then reading them all again to score, so its best case was a smaller
final sort. A bound that costs a full pass cannot beat scoring in a full pass.

Real WAND for this index would need, per n-gram: postings sorted by word index (already true
by construction), a per-posting-list upper bound on contribution (`2 * min(c_query, c_max) *
weight / min_denominator` -- cheap to precompute in `_freeze`), and the pivot walk with a
running top-k heap. It pays off exactly where the current design hurts: at 1M phrases the
`"e "` bigram has ~500k postings and an upper bound so low that WAND would skip nearly all of
them once the heap fills. It does *not* pay off on the small benchmarks (the whole vocabulary
fits in cache and the numpy pass is ~1 ms), and the pivot walk is a python loop unless written
in a compiled extension, which the standing constraint on `nmd` staying dependency-free
makes a V8-shaped project rather than a V7 patch. Worth trying only against the 1M-phrase
load test in `experiments/large_vocab_bench.py`, and only with a compiled inner loop; a
python-loop prototype will lose to the numpy pass at every size and prove nothing.

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

### ⚠ "emd is not the bottleneck" is a short-word result and reverses on long documents

The note further down this file says emd was ~7% of a V6 lookup. Profiled 2026-09-06 on
`ermagellan`, `emd_1d_dp` is **78%** of a lookup at `n=(2,4)` and **97%** at `n=(1,)`. It only
looked cheap because 8-character words almost never repeat an n-gram.

**None of the shipped variants beats the plain dp on that workload** (`experiments/emd_variants_bench.py`):

| variant | n=(1,) | n=(2,4) |
|---|---|---|
| `emd_1d_dp` | 127.7 µs | 10.8 µs |
| `emd_1d_fast` | 136.8 µs (0.93x) | 11.2 µs (0.96x) |
| `emd_1d_hybrid` | 144.4 µs (0.88x) | 12.4 µs (0.87x) |

`emd_1d_fast`'s shortcuts are for 1-vs-1 and 1-vs-many, which V7 already closed-forms inline, so
nothing reaching `emd_1d.py` can use them. `emd_1d_hybrid` loses even at 46.2% equal-mass
because it *sorts*, and the common shapes are (3,3) and (2,2) -- two `sorted()` calls cost more
than a nine-cell dp. `emd_1d_old` is hybrid's preprocessing plus brute-force enumeration, so
hybrid strictly dominates it; it stays only as `test_emd_correctness.py`'s oracle.

**What did work: skip the dp when the two sides have equal counts.** With equal masses nothing
goes unmatched, so 1-D optimal transport is the monotone matching and the displacement is the
sum of paired differences -- and no sort is needed, because the index already builds both
location lists in ascending order (verified: 0 of 40 000 workload pairs were unsorted, which is
precisely what `emd_1d_hybrid` was paying for). Exact to 2.2e-16, bit-for-bit identical output,
1.19-1.21x on the emd workload. `TestPresortedLocations` pins the sortedness invariant, since
breaking it would produce silently wrong distances rather than an exception.

**Landed: batching (`_emd_1d_batch`).** Multi-postings are grouped by occurrence count so each
group shares a dp shape and one numpy dp serves the whole group -- `m*n` array ops instead of
`m*n*rows` python ones. Bit-identical on all 40 000 workload pairs. **1.73-1.90x** end to end on
ermagellan at `n=(2,4)`, `(2,3)` and `(1,)`, and a no-op elsewhere.

⚠ It needs the `_EMD_BATCH_MIN_ROWS = 32` gate. Batching *unconditionally* is a wash at
`n=(2,4)` and a **loss** at `n=(1,)` -- 1947 ms/query against 1625 for never batching. Within a
single query n-gram the postings fragment by occurrence count into groups far smaller than the
workload-wide mean suggests; sizing this from pooled group sizes was wrong. The tuning table is
in the source next to the constant.

### ⚠ Banded dp: built, benchmarked, rejected 2026-09-06 -- do not re-derive

The dp only needs the band `i <= j <= i + (len_y - len_x)` when every point of the shorter side
gets matched, which holds whenever the combined spread is at most 2 (below that, matching always
beats dropping both). That is always true for the index's [0, 1] positions. It saves exactly
`len_x * (len_x - 1)` operations, ~1.83x fewer dp cells on ermagellan.

It was implemented, verified bit-identical on 40 000 real pairs plus random stress on both sides
of the spread threshold, and **still not worth keeping**:

| task | n | ratio | banding fires |
|---|---|---|---|
| ermagellan | (1,) | 1.17-1.38x | 69.5% |
| ermagellan | (2, 4) | 1.07x | 21.7% |
| ermagellan | (2, 3) | 0.97x | 20.1% |
| typo_hard | (1, 2) | 1.01x | **0%** |
| typo | (1, 2) | 1.05x | **0%** |
| abtbuy | (2, 4) | 1.05x | **0%** |

The spread check costs a few reductions regardless of dp size, so it has to be gated on
`len_x * (len_x - 1) >= ~12`. Ungated it **cost** 0.77x on typo `n=(1,2)`, where dp shapes are
around (2, 3) and six operations saved cannot pay for the check. But that same gate excludes
every short-string workload outright -- 0% firing -- so the ±5% there is noise on an unchanged
path. The only real win is `n=(1,)` on long documents, which scores **MAP 0.008** and is a
configuration the parameter search ruled out. No threshold rescues it: lowering the gate to fire
on `(2,3)`/`(2,4)` reintroduces the typo regression.

Cost would have been ~45 lines of a second kernel, two tuning constants, and a domain
restriction (valid only for spread <= 2) that a later edit could silently break.

### ⚠ Global grouping across n-grams: built, benchmarked, rejected 2026-09-06 -- do not re-derive

A batch only needs matching dp *shapes*; both sides may vary row to row. So postings could be
grouped by shape across every n-gram in a lookup rather than per n-gram. On paper this looked
strong -- at `n=(2,4)` it cuts groups 961 -> 98, raises the mean group 53 -> 521, drops numpy
calls 4510 -> 1970 (2.3x), and cuts rows falling to the scalar path 4902 -> 382.

Measured interleaved, min-of-N (ratios are sound even though absolute times drift under load):

| task | n | ratio |
|---|---|---|
| ermagellan | (2, 3) | 1.12x |
| abtbuy | (2, 4) | 1.03x |
| ermagellan | (1,) | 1.02x |
| abtbuy | (2, 3) | 0.98x |
| ermagellan | (2, 4) | 0.97x |
| typo | (1, 2) | **0.91x** |
| typo_hard | (1, 2) | **0.84x** |

**The call-count projection was right and still wrong**, because it counted only half the cost.
Per n-gram, every row of a batch shares one n-gram's query locations, so the query side is
`np.broadcast_to(...)` -- zero-copy. Batching across n-grams gives rows *different* query
vectors, so it must materialise `rows * m` floats. That cost scales with batch size, i.e. exactly
where the call savings were meant to come from, and on typo workloads it is pure overhead with
nothing to offset it. `np.repeat` over rows sorted by n-gram would soften it but not change the
verdict: the ceiling is that 1.12x.

⚠ It is also the only change in this line of work that is **not bit-identical** -- worst score
difference 5.7e-14 with 5 ranking changes in 288 lookups, because `bincount` reorders the
additions and scores 1e-14 apart are no longer exact ties, so V7's alphabetical tie-break stops
applying. `lookup`'s docstring promises reproducible tie order, so this would have been a
documented behaviour change on top of a non-win.

### Landed: `reduceat` on the 1-vs-m branch -- the win the dp work was not

Re-profiling after the two dp changes showed the dp is **no longer the bottleneck**: 30% of an
ermagellan lookup at `n=(2,4)` and `(2,3)` (was 78%), and of that the *scalar* `emd_1d_dp` (21%)
now exceeds the batched kernel (9%). What the profile did show, by self-time, was 453 716
`builtins.min` and 315 668 `builtins.abs` calls over 8 queries -- all from the branch where the
**query** n-gram occurs once and the indexed word repeats it:

```python
for other_word_index, other_locs in multi.get(n_gram, ()):
    min_dist = min(abs(location - y) for y in other_locs)
```

That runs for far more (n-gram, document) pairs than the dp ever does. `_freeze` now also stores
each n-gram's multi postings flattened -- `(locations, segment offsets, word indices)` in
`_multi_flat` -- so the branch is `np.minimum.reduceat` over one array instead of a python loop
with a generator-expression `min` per posting.

Bit-identical: worst score difference exactly 0.0 and zero ranking changes over 344 lookups,
including interleaved `add_word`/`lookup` sequences (`_freeze` rebuilds the flat arrays only for
touched n-grams, so a stale cache was the risk). Measured under CPU contention, so read the
ratios rather than the absolute times:

| task | n | lookup | build |
|---|---|---|---|
| typo_hard | (1, 2) | **2.47x** | 0.98x |
| typo | (1, 2) | **2.40x** | 1.03x |
| abtbuy | (2, 3) | **1.69x** | 0.82x |
| abtbuy | (2, 4) | **1.60x** | 0.84x |
| typo | (2, 4) | 1.37x | 0.94x |
| typo_hard | (2, 4) | 1.19x | 0.84x |
| ermagellan | (2, 4) | 1.11x | 0.85x |
| ermagellan | (1,) | 0.98x | 0.57x |
| ermagellan | (2, 3) | 0.89x | 0.70x |

Note the shape: this helps the **short-string** workloads most, the exact opposite of the dp work,
which only ever helped long documents. Build is slower (the flat arrays are constructed in
`_freeze`) and that was accepted as a one-off against a repeated lookup cost. The ermagellan rows
are inconsistent (1.11x vs 0.89x on near-identical configurations) because the 270-config sweep
was saturating the machine; they were not re-measured.

### The pattern behind the two rejections

After the equal-mass shortcut and `_emd_1d_batch` landed, **the dp is no longer where the time
goes**, so further dp micro-optimisation keeps failing to show up end to end. Both banding and
global grouping reduce dp work by the amount predicted and neither moves the wall clock. If this
is picked up again, re-profile first (`experiments/emd_variants_bench.py`, and cProfile on a
lookup) rather than assuming the 78-97% figure from before those two changes still holds.

⚠ Measurement note for anyone benchmarking here: a background sweep on the same machine made
identical baselines vary 2x (`abtbuy n=(2,3)` "before" measured 7.3 ms and 14.1 ms in two runs).
Interleave the two variants and take the **minimum**, not the median -- contention only ever adds
time. Also instrument how often the code path under test actually runs: two apparent regressions
(0.60x, 0.99x) turned out to be on workloads making zero batched calls.

⚠ **Pruning cannot help here.** `_similarity_vectors` computes `candidates` but pass 2 scores
every posting regardless -- the bound is a top-k correctness mechanism, not a work-saving one.
Restricting pass 2 to candidates is a real unexploited optimization on *short* corpora (0.9% of
30k words survive at `n=(2,4)` top_k=5) and worth nothing on long ones (90-100% survive on
ermagellan, because long documents share n-grams with everything). The two levers are inversely
correlated: pruning is selective exactly where the dp is rarely called.
