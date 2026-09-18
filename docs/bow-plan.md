# Plan: indexing and generalizing `bow_ngram_movers_distance`

Written 2026-09-04. Design notes only; nothing here is implemented yet.
Reference implementation being described: `nmd/nmd_bow.py`.

Two separate problems:

1. **Indexing** — retrieve the top-k token sequences from a corpus without
   computing `|Q|·|D|` NMD calls and one Hungarian solve per document.
2. **Split/merge tolerance** — the current metric fails completely when a word
   is split or merged (`["pineapple"]` vs `["pine", "apple"]`), and it is fully
   order-invariant when we usually want in-order matches preferred.

Part 2 changes the metric; part 1's index design survives that change (see
"Index compatibility" at the end).

---

## Part 1 — an index for the current metric

### Why it is indexable at all

`bow_ngram_movers_distance` (`nmd/nmd_bow.py:33-42`) builds a `|Q|×|D|` cost
matrix of `nmd(query_token, doc_token, normalize=True)` and runs
`linear_sum_assignment`. Over N documents that is `N·|Q|·|D|` EMD DPs plus N
Hungarian solves.

The assignment step does not decompose across documents, but the **costs do**:
every entry is `nmd(q, w)` for a query token `q` and a *vocabulary* token `w`.
It does not depend on which document `w` came from. So:

- the same `(q, w)` pair is recomputed once per document containing `w`;
- most pairs share no n-gram, so `sim = 0`, `cost = 1`, and never needed computing.

With `invert=True`, `min(|Q|,|D|) − Σcost` is just `Σ sim` over the assigned
pairs (`sim = 1 − cost`). The problem is a **max-weight bipartite matching on a
sparse similarity matrix**, which is the same shape as Word Mover's Distance
retrieval (Kusner et al. 2015). The standard answer there — index the
vocabulary, bound the matching, prune, solve exactly only for survivors — maps
directly onto what `nmd/nmd_index.py` already has.

### Structure

Three layers.

**1. Vocabulary n-gram index.** An `ApproxWordListV5` over the union of all
tokens in the corpus (deduplicated). One query token → sparse vector of
similarities to every vocab token that shares an n-gram. This is the "lots of
queries" part, done once per distinct query token instead of once per
(query token, document).

The V5 normalization `Σ(len(loc1)+len(loc2)−emd) / (num_grams(q)+num_grams(w))`
(`nmd/nmd_index.py:369`) is identical to
`ngram_movers_distance(q, w, invert=True, normalize=True)` (`nmd/nmd_core.py:64-73`)
for a single `n`, so the values are exact, not approximate.

**2. Document inverted index.** `vocab_idx → [(doc_id, count), ...]`, plus
`doc_id → [vocab_idx, ...]` (the token sequence) and `doc_id → len`. Scatter the
sparse similarity vectors from layer 1 into per-document cost matrices by dict
lookup; no NMD is computed at this layer.

**3. Bound-and-prune cascade** (prefetch-and-prune), cheapest first:

| Stage | Bound on normalized similarity | Needs |
| --- | --- | --- |
| Length | `min(|Q|,|D|) / max(|Q|,|D|)` | lengths only (the `real_quick_ratio` idea, now HANDOFF item 12) |
| Count | `sim(q,w) ≤ 2·Σ_g min(c_q(g), c_w(g))`, already used by V5 at `nmd_index.py:340-345` | n-gram count index, no EMD DP |
| Relaxed (RWMD) | `min( Σ_q max_{w∈D} sim(q,w), Σ_{w∈D} max_q sim(q,w) )` | real NMD values from layer 1 |
| Exact | `linear_sum_assignment` | only on docs whose bound beats the current k-th best exact score |

Order candidates by bound descending so the k-th-best threshold tightens early.
Documents sharing no n-gram with any query token never enter the candidate set;
their score is exactly 0.

Exactness shortcut: build a greedy matching (pairs in descending `sim`, skip
used rows/cols). If its total equals the relaxed bound, the bound is tight and
the Hungarian solve can be skipped. Near-duplicates hit this constantly.

### Sketch

```python
class BowIndex:
    def __init__(self, n=2):
        self.vocab = ApproxWordListV5(n=n, filter_n=0)  # needs an "all nonzero sims" mode
        self.postings = {}   # vocab_idx -> [(doc_id, count)]
        self.docs = []       # doc_id -> [vocab_idx, ...]

    def add(self, tokens): ...

    def lookup(self, query_tokens, top_k=10):
        # 1. one vocab query per distinct query token -> {vocab_idx: sim}
        sims = {q: self.vocab.all_similarities(q) for q in set(query_tokens)}
        # 2. scatter to docs -> doc_id -> upper bound (relaxed, both directions, min)
        bounds = ...
        # 3. exact assignment on candidates in bound order until bound < k-th best
        heap = []
        for doc_id, ub in sorted(bounds.items(), key=lambda x: -x[1]):
            if len(heap) == top_k and ub <= heap[0][0]:
                break
            cost = [[1 - sims[q].get(w, 0.0) for w in self.docs[doc_id]] for q in query_tokens]
            score = exact_from(cost)  # same arithmetic as bow_ngram_movers_distance
            ...
        return sorted(heap, reverse=True)
```

### Required changes to existing code

- `ApproxWordListV5.__lookup_similarity` is top-k oriented and prunes on
  `min_acceptable_score`. The BOW index needs *every* nonzero similarity (or
  everything above a `min_similarity`). The second half of that method
  (`nmd/nmd_index.py:347-376`) computes the right thing if the top-k filter is
  skipped — this is the "add a min_similarity filter" idea, now HANDOFF item 12.
- Normalization must match: `bow_ngram_movers_distance` uses a single `n`; the
  index default is `(2, 4)` averaged via `mean(..., dim)`. Use a single `n`, or
  accept that the indexed metric differs from `nmd_bow.py`.
- The `ZeroDivisionError` fallback `int(w1 != w2)` (`nmd/nmd_bow.py:38-39`)
  fires when `num_grams_1 + num_grams_2 == 0`, i.e. tokens shorter than `n−2`
  with a large `n`. The index must reproduce it to stay bit-exact.
- Duplicate tokens in a bag: postings carry counts; expand to repeated
  rows/columns when building the cost matrix, as brute force does.

### Expected gain (reasoning, not measured)

Layer 1 replaces `N·|Q|·|D|` EMD DPs with `|Q| · avg_posting_length`. The
Hungarian solves drop from N to the number of candidates whose bound beats the
k-th best, which for a distinctive query is tens, not thousands. Memory is the
vocabulary index plus one postings entry per (doc, distinct token).

### Test

Bit-exact parity against `bow_ngram_movers_distance` over random bags is the
whole correctness test — the index must return the same top-k with the same
scores. Sabotage check: break the relaxed bound (e.g. use only one direction ×
0.5) and confirm the parity test goes red.

---

## Part 2 — split/merge tolerance with in-order preference

### The problem

Token-level assignment treats `["pineapple"]` vs `["pine", "apple"]` as one
weak match plus one unmatched token. Character-level NMD on the joined string
handles that but is order-sensitive in the wrong way (no cheap reordering of
whole words). The current BOW metric is the opposite extreme: fully
order-invariant.

Wanted: a matcher that can **jump** to another position at word boundaries
(reorder), but is **not forced** to stop at a boundary (a match may run across
one, i.e. merge), and that **prefers** in-order matches.

### Formulation: segment mover's distance

> Superseded in one respect by Part 4: mass is measured in n-grams of the
> tokens, not in tokens (`mass='ngrams'` in `nmd/nmd_segments.py`). The
> token-mass version below is kept as `mass='tokens'`; it absorbs junk tokens
> into segments.

Generalize the assignment from tokens to **segments**: a segment is the
concatenation (no separator) of `p` consecutive tokens, `1 ≤ p ≤ L`. Both
sequences are covered by segments; segments are matched across sides; every
token is used at most once.

Let A have `k` tokens, B have `m` tokens. For a segment `s ⊂ A` of `p` tokens
and `t ⊂ B` of `q` tokens:

- `sim(s, t)` = `ngram_movers_distance(concat(s), concat(t), invert=True, normalize=True)`
- `pos(s)` = normalized center of the segment: mean token index divided by
  `k − 1` (0 when `k == 1`), the same normalized-location convention NMD uses
  for n-grams (`nmd/nmd_core.py:52`)
- `Δ = |pos(s) − pos(t)|`

Gain of matching `s` to `t`:

```
gain(s, t) = (p + q) · sim(s, t)          # mass on both sides, as in nmd_core
           − λ · (p + q) / 2 · Δ          # positional shift, "in-order" preference
           − μ · (p + q − 2)              # merge penalty per boundary crossed
```

```
similarity(A, B) = max over valid matchings of Σ gain(s, t)
distance(A, B)   = (k + m) − similarity(A, B)
normalized       = ÷ (k + m)
```

Normalizing by `k + m` rather than `max(k, m)` mirrors `nmd_core` (which
normalizes by `num_grams_1 + num_grams_2`) and is what makes `["pineapple"]` vs
`["pine", "apple"]` score `1.0` at `μ = 0`: one match, `p + q = 3` = total mass.

Why displacement rather than inversion count for the order term: displacement
is linear in the matching variables (stays an assignment / ILP), inversions
are quadratic. It is also exactly what NMD already does at the character level,
so the token level and character level use the same notion of "moved".

### Reductions (sanity properties, also the test cases)

| Setting | Reduces to |
| --- | --- |
| `L = 1, λ = 0, μ = 0` | current `bow_ngram_movers_distance` (up to the `k+m` vs `max(k,m)` normalization) |
| `k = m = 1` | `ngram_movers_distance` |
| `L = 1, λ > 0` | Hungarian with a positional cost term — cheapest possible "prefer in-order" upgrade, no other change |
| `λ = ∞` (no reordering) | monotone block alignment DP, `O(k · m · L²)` |
| full merge, `μ = 0` | character-level NMD of the joined strings is a feasible matching, so `similarity ≥` that |

### Solving

Instances are tiny (addresses, names, titles: `k, m ≲ 10`). Segments per side
`≤ L·k`, candidate pairs `≤ L²·k·m` ≈ 900 at `L = 3, k = m = 10`.

1. **General case (reordering + segments):** 0/1 ILP. Variable `x_{s,t}` per
   candidate pair; one packing constraint per token
   (`Σ x over pairs whose segments cover the token ≤ 1`); maximize `Σ gain·x`.
   `scipy.optimize.milp` (HiGHS, scipy ≥ 1.9) — scipy is already the
   dependency for `linear_sum_assignment`. Prune pairs with `gain ≤ 0` first;
   that removes most of them, because most segment pairs share no n-gram.
2. **`L = 1`:** plain `linear_sum_assignment` on `1 − sim + λ·Δ/2`. Ship this
   first; it is a one-line change to `nmd_bow.py`.
3. **`λ = ∞`:** DP over `(i, j)` prefix positions with transitions
   `(i−p, j−q) → (i, j)`, `0 ≤ p, q ≤ L`, not both zero (`p = 0` or `q = 0` is
   a deletion at full mass). Useful on its own for aligned data (ASR reference
   vs hypothesis) and as a fast lower bound on similarity for the ILP.

Caching: `sim(s, t)` depends only on the two concatenated strings, so memoize
on the string pair; segments recur across documents.

### Parameters and known trade-offs

- `L`: 3 covers split (`1↔2`), merge (`2↔1`) and boundary shift
  (`"sports hub", "clementi"` ↔ `"sports", "hubclementi"`, a `3↔2`). Larger `L`
  rarely helps and grows the ILP quadratically.
- `λ = 1` mirrors `nmd_core` (a token-pair match displaced by the full length is
  worth nothing). `λ = 0` gives the current order-invariant behaviour.
- `μ`: with `μ = 0` and heavily reordered input the solver may prefer one big
  merge (character-level NMD of everything, no positional penalty) over
  token-level matches — that is a real trade-off between "same words, reordered"
  and "same characters", and `μ > 0` biases toward keeping word structure.
  Start with `μ = 0` and only add it if this shows up on real data.
- Concatenation is separator-free on the assumption that the source error
  dropped a space. A boundary-crossing 2-gram (`"sh"` in `"sportshub"`) is one
  unmatched n-gram of noise; acceptable.

### Index compatibility

Part 1 survives this change:

- add every 1..`L`-token concatenation of each document to the vocabulary
  index, with postings `(doc_id, start, length)`;
- the relaxed bound becomes `Σ_s max_t gain(s, t)` over query segments with the
  packing constraints dropped, still an upper bound;
- the exact stage runs the ILP instead of the Hungarian solve.

Postings grow by roughly `L×`; the vocabulary of concatenations is larger than
the token vocabulary but still deduplicated across documents.

---

## Part 3 — wrapper vs. inside NMD

Part 2 is a wrapper *because* it adds coupling: all n-grams of a segment move
together. NMD itself is a sum of independent per-n-gram EMDs, which is what
makes it fast and indexable, and also why it has no notion of a word moving as
a unit. The alternative is to keep the independence and change the ground cost.

### Option B — change the ground cost inside NMD

Each n-gram occurrence gets three coordinates instead of one:
`(global_loc, word_id, within_word_offset)`. Tokenize first, n-gram each token
with its own STX/ETX, global position = cumulative gram index normalized. Cost
of moving an occurrence:

```
cost = min( |Δglobal|,                     # slide: split/merge, small edits
            γ + |Δwithin_word_offset| )    # jump: moved to another word
```

- split/merge takes the slide path (global positions barely change);
- reordering takes the jump path (within-word offsets are preserved, pay `γ`
  per gram);
- `γ` is the in-order knob: `0` = order-invariant at word level, `∞` = current NMD.

The min-of-two-costs breaks the monotone-matching property `emd_1d_dp` relies
on, but per-n-gram multiplicities are tiny: one occurrence per side (the
overwhelming case) is just the formula; 2–3 per side is a brute-force
permutation. No solver.

Index impact: `ApproxWordListV5` already stores `(word_index, positions)` per
n-gram; the position becomes a triple. Count bound unchanged; only the
second-pass per-n-gram EMD changes. Easier to index than Part 2.

What breaks:

- the 1-D monotone structure `emd_1d_dp` relies on (only for multiplicities
  > 1; brute-force those);
- the triangle inequality. Min of two metrics is not a metric:
  `a=(g 0.0, off 0.0)`, `b=(g 0.05, off 0.9)`, `c=(g 0.9, off 0.9)` gives
  `d(a,b)=0.05`, `d(b,c)=γ`, `d(a,c)=0.9`. EMD inherits this, so the string
  distance is not a metric either. Irrelevant for the inverted index; relevant
  if a metric tree is ever wanted.

**The real problem — free re-indexing.** n-grams jump independently, so a
common bigram in unrelated words on both sides is matched for `γ` instead of a
large displacement. In a decoupled model this cannot be prevented: any global
control (jump count budget, jumped-mass budget) Lagrangian-relaxes back to a
constant per-jump price, i.e. `γ` again, and `γ` is a weak lever for bigrams.
The fix is *coupling* — the decision "word i ↔ word j" made once per word
pair — which is exactly what NMD's sum-of-independent-EMDs cannot express.

**What Option B is good for:** it is a relaxation of the coupled model (each
gram gets at least as good an option as the coupled solution gives it, for the
same `γ`), so its similarity is an upper bound on the coupled score. That is
the relaxed stage of the Part 1 cascade, and it lives in the existing V5
position lists.

### Run-length cost inside linear sum assignment (considered, rejected)

Idea: a single assignment over all n-gram occurrences with a bonus for keeping
contiguous runs (`x[i,j]` and `x[i+1,j+1]` both chosen). The run term is
pairwise between assignment variables — a quadratic assignment problem, which
LSA cannot represent. Injecting it (sequential discounts, or iterate
solve → detect runs → discount → re-solve) is a heuristic with no optimality
guarantee and can oscillate.

Exact versions of the same idea:

- monotone: affine-gap alignment (Gotoh). Gap-open = per-jump price,
  gap-extend ≈ 0. Exact `O(n·m)` DP, no reordering.
- with reordering: edit distance with block moves, NP-hard; greedy
  longest-matching-block (difflib / Ratcliff-Obershelp) is the usual heuristic;
  the tractable exact restriction is jumps only at word boundaries = Part 2.

A contiguous run of matched n-grams is a common substring, i.e. a word or part
of one, so "prefer runs" and "stop n-grams re-indexing independently" are the
same coupling constraint. `n=(2,4)` inside the per-pair `sim` is the cheap
separable approximation (a matched 4-gram is three consecutive matched
bigrams).

### Decision (revised 2026-09-04)

- Exact score: Part 2 (coupled segment matching; Hungarian at `L=1`, ILP with
  splits/merges; `sim` = NMD with `n=(2,4)` on concatenations).
- Index bound: Option B as the relaxed stage of the Part 1 cascade.
- Not pursued: run-length terms in LSA; jump budgets; Option B as the final score.

---

## Part 4 — prototype log

### 2026-09-04: reference implementation + Abt-Buy evaluation

Implemented Part 2 as `nmd/nmd_segments.py` (`align_segments`,
`segment_movers_distance`, `explain_alignment`), solved with
`scipy.optimize.milp`. `tests/test_segments.py` checks the ILP against
brute-force enumeration (40 random cases, both mass modes), the reductions
table, split/merge = 1.0, symmetry, and pins the token-mass defect below as a
strict xfail. Sabotage check done: perturbing the gain formula turned 9 tests
red.

**Finding 1 — token mass absorbs junk.** With `gain = (p+q)·sim`, a short junk
token (`in`, `a`, `the`) adds a full unit of token mass but only 2–3 n-grams, so
`(p+q)/grams` rises and merging it in *increases* the gain:

```
[theater in a] <-> [theater]   3:1  sim=0.65  gain=2.39    # [theater]<->[theater] alone: gain=2.00
[handset for the] <-> [expandable]   3:1  sim=0.11  gain=0.25
```

Fix, now the default (`mass='ngrams'`): measure mass in n-grams of the
*tokens* and define

```
gain(s, t) = mass(s) + mass(t) − nmd_distance(concat(s), concat(t))
distance   = Σ_matched nmd_distance(concat_s, concat_t) + Σ_unmatched token n-grams
```

i.e. character-level NMD with jumps allowed only at word boundaries — which is
the original ask. Merging junk adds zero matched n-grams so it cannot help.
Subtlety: concatenating `p` tokens changes the gram count by `(p−1)(n−3)` per
side (fewer for n=2, more for n=4), so mass must be counted on the tokens, not
on the concatenation, for a perfect split/merge to be exactly distance 0.
Side effect: long tokens (model numbers) weigh more than short ones, which
suits product names; `mass='tokens'` is kept for the bow-like alternative.

**Finding 2 — noise-floor pairs.** Any positive gain is accepted, so bigram
coincidences produce matches like `[920] <-> [voyager 510] 1:2 sim=0.07`.
Added `min_sim` (normalized-sim cutoff); 0.2 removes them, improves ranking
slightly and halves runtime (fewer ILP variables).

**Finding 3 — position convention.** Tokens must be treated as unit intervals
(`pos = (start + length/2) / k`) so a whole-sequence segment sits at 0.5 on both
sides; with point positions `["pineapple"]` vs `["pine","apple"]` got a
spurious shift of 0.5.

**Evaluation.** Abt-Buy product-name benchmark
(https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip, names only, casefold +
whitespace split, drop non-alphanumeric tokens). 100 random Abt queries
(seed 0), each prefiltered to the top-20 of 1092 Buy names by char-NMD(2); the
true match was inside the candidates for 99. Regenerate with
`PYTHONPATH=. python experiments/segment_eval.py 100 20`.

| metric | hit@1 | MRR | sec (all 99) |
| --- | --- | --- | --- |
| char-nmd(2) on joined string | 0.697 | 0.792 | 0.3 |
| bow(2) | 0.727 | 0.801 | 1.5 |
| seg, tokens, L=1, λ=0 | 0.737 | 0.824 | 27.9 |
| seg, ngrams, L=1, λ=0 | 0.758 | 0.838 | 26.3 |
| seg, ngrams, L=3, λ=0 | 0.768 | 0.844 | 36.9 |
| seg, ngrams, L=3, λ=0.5 | 0.737 | 0.827 | 33.1 |
| seg, ngrams, L=3, λ=1 | 0.758 | 0.830 | 28.6 |
| **seg, ngrams, L=3, λ=0, min_sim=0.2** | **0.768** | **0.847** | 13.5 |
| seg, ngrams, L=3, λ=0.5, min_sim=0.2 | 0.768 | 0.847 | 13.4 |

Small sample; treat as direction, not a measurement. Reading the alignments
matters more than the table:

- splits / merges are found in the wild: `[mdrex55bk] <-> [mdr ex55/blk] 1:2`,
  `[5.8ghz] <-> [5.8 ghz] 1:2`, `[genesis s320] <-> [genesis s-320] 2:2`;
- reordering works: `[iphone] <-> [iphone] shift=0.49` in
  `Griffin iPhone 3G ... Case` vs `Griffin Elan Form for iPhone 3G`;
- λ does not help on this data — product names reorder freely (model number
  moves from end to front), so the in-order preference is a domain choice, not
  a free win;
- exact ties are broken toward multi-token merges (`[camera case] 2:2` instead
  of two 1:1s); harmless, a tiny `mu` makes alignments read cleaner;
- misses are dominated by (a) the true Buy name carrying a long spec suffix,
  which symmetric normalization punishes — an asymmetric "fraction of the query
  explained" normalization is the retrieval fix (README's "prefix lookup"
  todo), and (b) model-number near-misses (`ln52a750` vs `ln55a950`), which no
  string metric resolves.

Runtime is the reference implementation's: ~0.1 s per pair with all segment
sims computed from scratch; Part 1's vocabulary index is what makes it usable
at corpus scale.

## Part 5 — how it compares to standard methods

Measured 2026-09-04. Scripts: `experiments/baseline_eval.py`,
`experiments/er_magellan_eval.py`. Every baseline is implemented inside
`baseline_eval.py` (no new dependencies). Names only, in both experiments.

### Retrieval: rank the Buy catalogue given an Abt name

100 queries sampled with `seed=0`, all 1092 Buy names, no prefilter.

| metric                     | hit@1 |   MRR | recall@20 |  sec |
|----------------------------|------:|------:|----------:|-----:|
| tf-idf cosine, char 3-gram | 0.890 | 0.930 |     1.000 |  0.4 |
| soft tf-idf (JW, θ=0.9)    | 0.840 | 0.899 |     1.000 | 15.7 |
| jaccard, char 3-gram       | 0.770 | 0.835 |     0.990 |  0.3 |
| monge-elkan (JW)           | 0.750 | 0.829 |     1.000 | 36.7 |
| tf-idf cosine, token       | 0.700 | 0.776 |     0.960 |  0.1 |
| `char-nmd(2)`              | 0.690 | 0.785 |     0.990 |  9.8 |
| jaccard, token             | 0.630 | 0.738 |     0.960 |  0.1 |
| difflib ratio              | 0.500 | 0.606 |     0.820 | 17.9 |

Reranking the top-20 of a *neutral* tf-idf char-3gram prefilter (true match
present for 100/100), so every metric sees identical candidates:

| metric                     | hit@1 |   MRR |  sec |
|----------------------------|------:|------:|-----:|
| tf-idf cosine, char 3-gram | 0.890 | 0.930 |  0.0 |
| soft tf-idf (JW, θ=0.9)    | 0.830 | 0.895 |  0.1 |
| `segment L3 λ=0 min_sim.2` | 0.760 | 0.840 | 14.1 |
| monge-elkan (JW)           | 0.760 | 0.841 |  0.6 |
| `bow-nmd(2)`               | 0.720 | 0.795 |  1.5 |
| `char-nmd(2)`              | 0.690 | 0.785 |  0.3 |

### Classification: the published ER-Magellan Abt-Buy split

5743 train / 1916 test blocked pairs (616 / 206 positive), from the Ditto repo.
Each metric becomes one global threshold tuned on train, applied to test — a far
weaker learner than the published systems, which are supervised over every
attribute. Test F1 ×100:

| metric                     | F1    | prec  | rec   |
|----------------------------|------:|------:|------:|
| tf-idf cosine, char 3-gram | 64.88 | 72.46 | 58.74 |
| soft tf-idf (JW, θ=0.9)    | 62.39 | 55.73 | 70.87 |
| jaccard, char 3-gram       | 46.82 | 40.57 | 55.34 |
| `segment L3 λ=0 min_sim.2` | 46.68 | 44.16 | 49.51 |
| `segment L3 λ=0`           | 46.53 | 43.15 | 50.49 |
| monge-elkan (JW)           | 44.29 | 34.12 | 63.11 |
| `char-nmd(2,4)`            | 44.25 | 40.65 | 48.54 |
| `char-nmd(2)`              | 43.68 | 41.48 | 46.12 |
| `bow-nmd(2)`               | 42.73 | 40.17 | 45.63 |
| jaccard, token             | 41.86 | 34.84 | 52.43 |

Published on this split (supervised, all attributes): Magellan 43.6,
DeepMatcher+ 62.8, Ditto 89.33.

### What this says

- **The segment work does what it was built to do.** Within the nmd family the
  ordering is consistent across both experiments: `char-nmd` → `bow-nmd` →
  `segment`, +4 to +7 points of hit@1 and +4 F1 over `bow-nmd`. The alignment
  machinery is not the weak part.
- **The representation is the weak part: no IDF.** Every method above the
  segment metric is idf-weighted; every method below it is not. Jaccard vs
  tf-idf over the *same* char-3gram feature set is +12 hit@1 and +18 F1 — the
  largest single effect in either table. NMD gives every n-gram equal mass, so
  it spends most of its budget matching `digital`, `camera`, `black`, `series`.
  `_pair_gain` already carries a per-token mass, so `mass *= idf(gram)` is the
  obvious next experiment, and it is orthogonal to Parts 1–4.
- **A plain tf-idf char-3gram cosine beats all of this**, at 0.890 hit@1 and
  ~400× faster than the segment ILP; at 64.88 F1 it also beats DeepMatcher+
  using one threshold and one attribute. Anything built here has to clear that
  bar to be worth its cost.
- Caveats: 100 queries is ±0.04 on each hit@1; names only, no description or
  price; the ER-Magellan threshold is tuned on train, so it is not unsupervised.

## Part 6 — hyperparameter search, and why nmd trails jaccard

Measured 2026-09-04, ER-Magellan Abt-Buy split, `name` only. Scripts:
`experiments/position_penalty.py`, `experiments/hyperparam_search.py`,
`experiments/idf_variants.py`.

### nmd is jaccard *minus* an order allowance, not plus one

A set has no order, so jaccard is already fully order-invariant; nmd charges for
displacement on top of the same n-gram overlap. Dropping nmd's position term
leaves `2·|A n B|_multiset / (|A| + |B|)` — Dice over n-gram multisets — and Dice
is a monotone transform of Jaccard (`J = D / (2 - D)`), so the two are
rank-equivalent. `position_penalty.py` confirms this: `jaccard set n=k` and
`dice set n=k` produce identical AP and F1 at every n, and `nmd <= dice-multiset`
in 15/15 checks. Hence

    normalized nmd  =  jaccard-family overlap  -  positional discount  -  (set -> multiset)

Train AP at n=2 splits the gap almost evenly: jaccard/dice set 0.484 ->
dice multiset 0.464 (counting occurrences rather than types) -> nmd 0.441 (the
positional discount). The discount removes 7.2% of the score on true matches but
only 6.7% on non-matches, so it shaves the wrong side harder: separation drops
from 0.166 to 0.152. `sony camera dsc123` vs `dsc123 sony camera` — identical
content, reordered — loses a quarter of its score (dice 0.722, nmd 0.541).

At λ=0 with merges allowed, the segment metric scores 46.68 against
dice-multiset-n=2's 46.56: it buys back almost exactly what the position penalty
took, landing on the unweighted-overlap ceiling. The design works; the ceiling is
the problem.

### The search buys ~3 F1

`hyperparam_search.py`, ranked by train AP (threshold-free, so the test set is
untouched), then threshold-tuned test F1. char-nmd and bow-nmd are exhaustive
over n; segment is a coordinate search on a stratified subsample (616 positives,
3x negatives), which is not exhaustive and can miss parameter interactions.

| family   | tuned config                                    | before | after |
|----------|-------------------------------------------------|-------:|------:|
| char-nmd | `n=(2,)` — already the default, 14 combos tried  |  43.68 | 43.68 |
| bow-nmd  | `n=4` (train AP 0.4224 vs 0.3847 for n=2)        |  42.73 | 42.61 |
| segment  | `n=(3,4) L=4 λ=0 μ=0.5 mass=ngrams min_sim=0.4`   |  46.68 | 49.90 |

- **n does not want tuning at the character level.** char-nmd is monotone
  decreasing in n (AP 0.437 / 0.434 / 0.421 / 0.397 for n = 2/3/4/5) and no
  multi-n combination beats plain n=2. (`n=1` was unavailable when this ran;
  it was added afterwards — see Part 7, where it matters a great deal.)
- **bow-nmd's tuning did not transfer**: +0.038 train AP, −0.12 test F1.
- **λ hurts monotonically** on this data (subsample AP 0.691 / 0.684 / 0.677 /
  0.660 for λ = 0 / 0.25 / 0.5 / 1.0), confirming Part 4.
- ⚠ subsample AP is at 25% prevalence and is **not** comparable to the
  full-train AP figures elsewhere in this doc (10.7%).
- `max_len` and `min_sim` both selected the edge of their grids, so both axes
  were extended. Neither was hiding anything: `min_sim` peaks at 0.4-0.5 and
  falls away after (subsample AP 0.7008 / 0.7014 / 0.6944 / 0.6602 / 0.6442 for
  0.4 / 0.5 / 0.6 / 0.7 / 0.8), and `max_len` 4, 5 and 6 give **identical** AP
  to four decimal places — segments longer than four tokens are never selected.
  `min_sim=0.5` wins the search criterion by 0.0006 AP but scores 49.38 test F1
  against 0.4's 49.90, i.e. the two are indistinguishable and the search has
  saturated at ~50 F1.

### idf is worth 5-15x what the hyperparameters are worth

Weighting is structurally free: `nmd_core.py:64-68` accumulates similarity per
n-gram *type* independently and idf is constant within a type, so `w(g)` is a
scalar multiplier on that type's contribution. `emd_1d_dp` is untouched, no
matching decision changes, non-negative weights keep it a metric, and the
`similarity + distance == total` identity holds with `total = Σ w(g)·count(g)`.
`df(g)` is already in the index as `len(postings[g])`.

Char 2-grams, train AP / test F1:

| variant                        | train AP | test F1 |
|--------------------------------|---------:|--------:|
| nmd                            |    0.441 |   43.46 |
| nmd + idf                      |    0.588 |   53.62 |
| nmd + geometric normalization  |    0.450 |   43.75 |
| nmd + idf + geometric norm     |    0.596 |   54.01 |
| nmd + idf² + geometric norm    |    0.685 |   60.72 |
| cosine log-tf + idf (baseline) |    0.754 |   65.19 |

Note that tf-idf **cosine** puts `w(g)²` in the numerator for a shared gram, so
"idf in a cosine" and "idf in a Dice-shaped measure" are different weightings —
most of what looks like a normalization advantage is the squared weight.
Normalization alone is worth +0.012 AP; the exponent is worth +0.10.

### ⚠ but the exponent does not peak, and that is a finding about the dataset

Sweeping `w(g) = idf(g)^p` at n=3, train AP climbs monotonically to p≈15-25
(AP 0.827, test F1 74.76) — past every baseline including tf-idf cosine. As
p → ∞ the score is decided by the single rarest shared n-gram, and that limit
measured on its own ("max idf among shared 3-grams, over the rarest gram
available") scores **AP 0.746 / F1 73.66 with no mover's distance at all**.
46% of 3-gram types in this corpus contain a digit.

So Abt-Buy is substantially a model-number matching benchmark, and a large idf
exponent is a roundabout rare-substring detector. nmd at p=15 (AP 0.827) does
beat the pure rare-gram rule (0.746), so the overlap structure adds real signal —
but tuning p here optimizes for model numbers and should not be expected to
transfer to company names or addresses.

**Recommendation: ship `weights` with a default of `p=2`** (captures +0.24 AP /
+17 F1, stays a smooth general-purpose metric) rather than a benchmark-tuned p.
Watch punctuation: the rarest 3-grams here are `') ,'`, `'v ,'`, `'t :'` —
formatting noise, which p=15 amplifies about a millionfold. Normalize
punctuation before gramming, floor the idf, or take `df` from a corpus larger
than the thing being matched.

For the segment metric, `mass(token) = Σ_g w(g)` over the token's n-grams; every
Part 2/4 invariant is linear in mass so segmentation-invariance and
`similarity + distance == total` carry over, but `test_split_merge_is_perfect`
must be re-run because merge-boundary n-grams now carry unequal weights.

## Part 7 — the dictionary lookup task reverses almost all of Part 6

Measured 2026-09-04. Script: `experiments/word_lookup_eval.py`. 8000 words from
`experiments/words_en.txt`, 241 queries corrupted with 1-2 random edits
(insert / delete / substitute / transpose), rank the whole dictionary.

Everything in Parts 5-6 was product names. This is the task nmd was built for,
and the conclusions do not carry over.

| metric                        | hit@1 |   MRR | hit@10 |
|-------------------------------|------:|------:|-------:|
| `nmd n=(1,2)`                 | 0.938 | 0.957 |  0.988 |
| `nmd n=1`                     | 0.934 | 0.957 |  0.992 |
| difflib ratio                 | 0.905 | 0.937 |  0.983 |
| `nmd n=(1,2,4)`               | 0.900 | 0.928 |  0.979 |
| `nmd n=2 + idf¹`              | 0.863 | 0.904 |  0.971 |
| `nmd n=2`                     | 0.846 | 0.894 |  0.971 |
| `nmd n=(2,3)`                 | 0.842 | 0.879 |  0.954 |
| cosine n=2, **no** idf        | 0.838 | 0.886 |  0.967 |
| jaccard n=2                   | 0.834 | 0.886 |  0.971 |
| `nmd n=(2,4)` — README default | 0.826 | 0.869 |  0.954 |
| `nmd n=2 + idf²`              | 0.763 | 0.836 |  0.946 |
| jaccard n=3                   | 0.743 | 0.803 |  0.921 |
| tf-idf cosine n=2             | 0.734 | 0.810 |  0.938 |
| tf-idf cosine n=3             | 0.676 | 0.760 |  0.900 |
| tf-idf cosine n=2, idf²       | 0.440 | 0.542 |  0.739 |
| `nmd n=2 + idf⁶`              | 0.344 | 0.445 |  0.627 |

### Three reversals

1. **nmd wins here.** 0.938 against tf-idf cosine's 0.734 and jaccard's 0.834.
   On product names nmd lost to tf-idf cosine by 21 F1; here it wins by 20 points
   of hit@1. The metric is not weak, it was being measured on the wrong task.
2. **idf hurts.** Cosine *without* idf scores 0.838, *with* idf 0.734. For nmd,
   idf¹ is worth +0.017 and everything above that is destructive (idf² 0.763,
   idf⁶ 0.344). The reason is that the two tasks have opposite relationships
   between rarity and signal: in a product catalogue a rare n-gram is the model
   number, i.e. the answer; in a misspelled word a rare n-gram is usually **the
   typo itself**, i.e. the noise. Weighting by rarity amplifies whichever it is.
3. **Low n wins, monotonically**: 0.934 / 0.846 / 0.751 / 0.639 for n = 1/2/3/4.
   The README's `(2,4)` default is 11 points worse than `(1,2)`.

### What n=1 does and does not discard

`n=1` keeps *position* — it is still a 1-D EMD over character positions — and
discards only *adjacency*. So the finding is not "order does not matter" but
"positional information carries the signal, and adjacency costs more in
brittleness under edits than it pays back". A single substitution destroys two
bigrams and three trigrams but only one unigram, which is exactly the observed
monotone decline in n.

⚠ **`n=1` is a good scorer but a bad index key.** A unigram posting list over a
26-letter alphabet has almost no selectivity, so Part 1's n-gram filter still
needs n >= 2. The natural combination is to prune on 2- or 3-grams and rescore
with `n=(1,2)`, which is also why the README's index default is left alone.

### Consequence for the idf plan in Part 6

Part 6 recommended shipping `weights` with a default of `p=2`. **That default is
wrong for this library.** idf should be opt-in with no weighting by default: the
documented primary use case is dictionary lookup, where p=1 is marginal and p=2
costs 8 points of hit@1. Corpus-appropriate weighting is a caller's decision,
not a default. The structural change (a `weights` parameter, free because nmd
accumulates per n-gram type) is still worth making.

### Caveats

- 241 queries, so differences below ~0.03 hit@1 are not resolvable; `n=1` and
  `n=(1,2)` are tied within noise, but both clearly beat `(2,4)`.
- Corruptions are uniform random edits. Real typos are keyboard-adjacent or
  phonetic, and the README's examples at the time (`asalamalaikum` ->
  `assalamualaikum`) were transliteration variants with a different edit profile.
  The README's examples were changed to phonetic English misspellings on
  2026-09-17 (`fotografer` -> `photographer`); this caveat is about the *corpus*
  measured here, which is unchanged, so it still stands.
- English only; `experiments/words_ms.txt` is untested.

## Part 8 — every metric, every dataset, one ranking

Measured 2026-09-05. Script: `experiments/benchmark_all.py`. Three tasks:

- **A** product-name retrieval — 100 Abt names ranked against 1092 Buy names
- **B** product-name matching — ER-Magellan Abt-Buy pairs, one tuned threshold, test F1
- **C** typo correction — 241 corrupted words ranked against an 8000-word dictionary

`idf` is fitted per task on the corpus being searched, as an index would have it.

### Rank within each task (1 = best), sorted by worst showing

| metric | A retrieval | B match F1 | C typos | worst |
|---|---:|---:|---:|---:|
| `nmd n=2 idf¹` | 7 | 7 | 5 | **7** |
| `nmd n=2 idf²` | 4 | 5 | 9 | 9 |
| monge-elkan (JW) | 10 | 10 | 4 | 10 |
| jaccard char-3gram | 9 | 11 | 11 | 11 |
| tf-idf cos char-2gram | 1 | 2 | 12 | 12 |
| jaccard char-2gram | 8 | 12 | 7 | 12 |
| `nmd n=3 idf²` | 5 | 6 | 13 | 13 |
| `nmd n=(1,2)` | 13 | 13 | **1** | 13 |
| tf-idf cos char-3gram | 3 | 3 | 14 | 14 |
| tf-idf cos char-3gram idf² | 2 | **1** | 15 | 15 |
| `nmd n=2` | 15 | 14 | 6 | 15 |
| `nmd n=(2,4)` | 11 | 16 | 8 | 16 |
| soft tf-idf (JW) | 6 | 4 | 17 | 17 |
| `nmd n=1` | 16 | 17 | 2 | 17 |
| difflib ratio | 17 | 19 | 3 | 19 |

Headline numbers: A `tf-idf cos char-2gram` 0.890 hit@1; B `tf-idf cos char-3gram idf²`
**70.10 F1** (clearing DeepMatcher+ 62.8 with one threshold and one attribute);
C `nmd n=(1,2)` 0.938 hit@1.

### What it says

- **No metric wins everywhere, and the spread is enormous.** The task-B winner
  (`tf-idf char-3gram idf²`) places 15th of 17 on typos, 0.560 against the
  leader's 0.938. The task-C winner (`nmd n=(1,2)`) places 13th on both product
  tasks. Picking by a single benchmark would be a mistake in either direction.
- **`nmd n=2 idf¹` is the best all-rounder** — never first, never worse than
  7th. Exponent 1 is where the two domains stop disagreeing: it is worth
  +0.15 hit@1 on A and +10 F1 on B, and is still slightly *positive* on C
  (0.846 -> 0.867), whereas exponent 2 buys another +6 F1 on B but costs 7
  points of hit@1 on C.
- **The rarity/signal split from Part 7 explains the whole table.** Everything
  that ranks well on A and B is idf-weighted or char-tf-idf; everything that
  ranks well on C is low-n and unweighted. A rare n-gram is the model number in
  a catalogue and the typo itself in a misspelling.
- **`difflib` is a genuinely good typo matcher** (3rd, 0.905) and the worst
  product matcher in the table (17th / 19th).

### Caveats

- ⚠ The `sec` columns are **not** a speed benchmark. Metrics are called pairwise
  with no caching, so tf-idf rebuilds both vectors on every comparison; with
  precomputed vectors the same task-A work takes 0.4 s rather than 17 s. Use
  `experiments/baseline_eval.py` for timing, or the V7 index.
- ⚠ `tf-idf cos token` and `soft tf-idf` score ~0 on task C because they are
  **structurally inapplicable**, not merely bad: a misspelled word is an unseen
  token, so its token tf-idf vector is empty and its cosine against every
  candidate is 0. Their task-C rows should be read as N/A.
- 100 queries on A, 241 on C, so differences below ~0.04 hit@1 are noise.
- Names only throughout; no description or price field.

### Prior art to check before implementing

- Word Mover's Distance retrieval (Kusner et al. 2015): WCD/RWMD bounds,
  prefetch-and-prune — Part 1 is this with NMD as the ground distance.
- Gale–Church / block sentence alignment: the `λ = ∞` DP is the same shape.
- Edit distance with block moves (Shapira & Storer; Cormode & Muthukrishnan) is
  the general "jumps anywhere" problem and is NP-hard; restricting jumps to
  token boundaries with a small `L` is what keeps this tractable.
- "Extended edit distance" (previously evaluated; the reference implementation
  was found buggy) — not used here.

## Part 9 — the index's own parameters, across six retrieval benchmarks

Parts 5-8 compared *scoring formulas*, pairwise, over every candidate. None of
them touched an index: `baseline_eval`, `word_lookup_eval`, `segment_eval`,
`benchmark_all` and `hyperparam_search` all call `metric(a, b)` per candidate, so
`ApproxWordListV7`'s own parameters had never been measured. Part 9 is that
search. Code: `experiments/index_param_search.py`; results as tracked CSVs in
`experiments/results/`.

### What was missing before this

- **MAP and NDCG did not exist anywhere in the tree.** Every eval script inlined
  its own `rank = next(...)` loop taking only the *first* relevant item's rank,
  which silently collapses multi-answer ground truth. They now live in
  `experiments/ranking_metrics.py` alongside `precision_at_k` and `r_precision`.
- **Two of the largest effects measured in Parts 6-7 were unreachable from the
  library.** Every winning row in Parts 6 and 8 uses *geometric* normalization,
  but V7 hardcoded the additive Dice denominator; and nothing in `nmd/` could
  down-weight the position term that `position_penalty.py` found harmful. Both
  are now `lookup` parameters (`denominator`, `position_weight`), defaulting to
  the previous behaviour bit-for-bit.

### The benchmarks

| task | corpus | queries | regime |
|---|---|---|---|
| `typo` | 8 000 words, words_en | 241 | 1-2 edits |
| `typo_ms` | 8 000 words, words_ms | 237 | the only non-English task |
| `typo_hard` | 30 000 words, british-english-insane | 248 | 2-3 edits, denser dictionary |
| `typo_brutal` | 30 000 of 518k merged words | 250 | 2-5 edits, worst-ranked only |
| `abtbuy` | 1 079 Buy product names | 100 | short product names |
| `ermagellan` | 1 035 entities, name + description | 206 | long documents (~138 chars) |

`typo_brutal` ranks corruptions by **Damerau**-Levenshtein (not plain
Levenshtein: `corrupt` draws uniformly from insert/delete/substitute/*transpose*,
and plain Levenshtein charges a transposition 2 where Damerau charges 1, which
would have packed the benchmark with transposition-heavy cases) normalized by
answer length, and requires answers of 6+ characters. ⚠ that length floor
is not cosmetic — normalizing by length *inverts* the length bias rather than
removing it, and the unfiltered top of the pool was `('iv', 'daily')`,
`('ot', 'eten')`, `('xtc', 'nati')`: unrecoverable, not hard. Note also that the
edit budget is an upper bound, not the achieved distance, since operations
overlap and cancel — the realized distance has to be measured, which is the other
reason to rank rather than just raise the budget.

### The result: two clusters, differing on `n`, `idf_exponent` and `dim`

| task | best MAP | `n` | idf | dim | normalize | pos_w | den |
|---|---|---|---|---|---|---|---|
| `typo` | 0.9646 | (1, 2) | 0.5 | 2 | True | 1.0 | dice |
| `typo_ms` | 0.9554 | (1, 2) | 0.0 | 2 | True | 1.0 | dice |
| `typo_hard` | 0.8562 | (1, 2) | 0.5 | 2 | True | 1.0 | dice |
| `typo_brutal` | 0.4129 | (1, 2) | 0.0 | 2 | True | 1.0 | dice |
| `abtbuy` | 0.9500 | (2, 3) | **3.0** | 1 | False | 0.0 | dice |
| `ermagellan` | 0.9039 | (2, 4) | **3.0** | 1 | True | 1.0 | **geo** |

⚠ **Corrected 2026-09-06.** The `ermagellan` row first read 0.7989 with `normalize=False`,
`position_weight=0.0`, `dice` -- measured on a 19-configuration shortlist that did not contain
the winning combinations. Re-run over the full 270-point grid its best is `normalize=True`,
`position_weight=1.0`, `geo` at **0.9039**; the shortlist configuration re-measures at 0.79894
there, so the runs agree and the shortlist was simply too narrow to rank on. **Do not draw
parameter conclusions from a shortlist drawn from other tasks' rankings.**

The clusters differ by `n`, `idf_exponent` and `dim` -- not by every knob. With the full grid in,
**all six tasks want `normalize=True` and `position_weight=1.0`**, and the two product tasks
agree far better than the shortlist implied: `ermagellan`'s winner scores 0.9088 on abtbuy
(against abtbuy's own best of 0.9500), while abtbuy's winner manages only 0.7989 on ermagellan.
So `n=(2,3)/(2,4) idf=3.0 dim=1 normalize=True position_weight=1.0 denominator='geo'` is a better
*product* setting than either task's individual optimum.

It still reproduces Part 7 at a larger scale: Part 7 found the dictionary task reversing Part 6's
product findings, and here four typo tasks reverse two product tasks on `n` and `idf_exponent`.

**Best single compromise:** on the five-task full grid, `n=(1,2) idf_exponent=0.5 dim=2
normalize=True position_weight=1.0 denominator='geo'` (best mean rank, 51.2 of 432). Restricted
to the unigram-free field all six benchmarks share, it is `n=(2,4) idf_exponent=1.0 dim=2
normalize=True position_weight=1.0 denominator='geo'` (mean normalized MAP 0.881, mean rank 74.8,
worst rank 151 of 270).
`geo` is what makes it a compromise: softening the length mismatch buys +0.022 MAP
on abtbuy for -0.004 on typo. ⚠ its worst rank is **234 of 432**, on
abtbuy, where it scores 0.834 against 0.950 achievable. There is no configuration
that is good everywhere.

**The old default is not a compromise, it is unchosen.** `n=(2,4) idf=0 dim=1
normalize=False` ranks 334 / 161 / 99 / 94 / 98 of 432 on the five full-grid
tasks, and 255 of 270 on ermagellan (MAP 0.3018 against 0.9039).

### Per-knob findings

- **`normalize` is the largest single knob, and it defaulted to off.** At the
  default `idf_exponent=0.0`, turning it on gains +0.3390 MAP on ermagellan
  (0.3018 -> 0.6408), +0.1954 on abtbuy (24/24 paired cells), +0.0733 on typo
  (24/24), +0.0696 on typo_brutal (22/24), +0.0580 on typo_hard (21/24) and
  +0.0502 on typo_ms (17/24). **V7's default is now `True`** — deliberately
  diverging from V6 and `ngram_movers_distance`, pinned by
  `tests/test_index_v7_knobs.py::TestDefaults`.
- ⚠ **`normalize` and `idf_exponent` substitute for each other.** Both
  counteract length bias, so applying both over-corrects. On abtbuy
  `normalize=True` is worth **+0.145** MAP at `idf_exponent=0` and **-0.079** at
  `idf_exponent=3`. Any future default change to one must re-check the other.
- **`position_weight` splits exactly as `position_penalty.py` predicted.** On
  abtbuy `position_weight=0.0` scores 0.9550 against 0.9462 at 1.0, and 0.0
  appears in both product tasks' winners. On typo it loses monotonically —
  0.75 -> -0.002, 0.5 -> -0.006, 0.25 -> -0.008, 0.0 -> -0.021, helping **0 of 22**
  paired comparisons. Position is the whole signal for a transposed character and
  noise for a reordered product title.
- **`idf_exponent` disagrees by an order of magnitude between clusters**: 0-0.5 on
  typo (collapsing above 1.0 — 2.0 -> 0.900, 3.0 -> 0.847), 3.0 on products, still
  near the grid edge there. Part 7's reversal of Part 6 holds.
- **`n=(1,2)` wins all four typo tasks**, confirming Part 7 through real candidate
  generation and MAP rather than pairwise rescoring alone.
- ⚠ **`denominator='geo'` is NOT minor** -- an earlier draft of this part said so, from a grid
  where `ermagellan` had only a shortlist. Paired against `dice` at `normalize=True` it is worth
  **+0.0991 MAP on ermagellan, helping 90 of 90 comparisons**, and +0.0159 on abtbuy (139/144),
  while being ~neutral on the typo tasks (+0.0043, +0.0060, +0.0010, -0.0033). It helps most
  where length mismatch is largest and costs nothing elsewhere, which makes it a good default
  rather than a tuning knob; 8 of the overall top 10 use it.
- **`dim` is minor**: `dim=2` beats `dim=1` on the typo tasks by ~0.002-0.007 MAP.

### Caveats

- ⚠ **`ermagellan` is measured on 270 of the 432 grid points**, not all of them: the 162
  configurations containing `n=1` cost **341 s each** there and score MAP **0.008**, because
  every 138-character document contains every letter, so they were excluded rather than spend
  ~15 hours measuring noise. The six-task aggregate therefore ranks those 270 unigram-free
  configurations, and its per-task "best" values are **lower than the five-task numbers above**
  wherever the true optimum used unigrams -- `typo` peaks at 0.9058 in that restricted field
  against 0.9646 with `n=(1,2)` available. They are different fields; do not mix rows between
  them.
- ⚠ **`n` is doing two jobs** — selecting candidates *and* scoring them — so
  a grid over a single `n` conflates "good index key" with "good scorer". That is
  exactly what Part 7's unimplemented prune-then-rescore design would separate, and
  it means any recommended `n` here is a compromise between the two roles.
- ⚠ **V7's "~98% of postings are single-occurrence" is short-word-specific.**
  Measured 2026-09-06: 99.5% on typo (8 chars), 92.3% on abtbuy (53 chars), **78.8%
  on ermagellan (138 chars)**. On long documents 21% of postings fall to the python
  dp fallback, which is where that 67 s per configuration goes. If long documents
  matter, that fallback is the thing to vectorise.
- The four typo tasks have exactly one right answer, so MAP there is numerically
  identical to MRR. abtbuy and ermagellan are set-valued but close to 1:1, so MAP
  and MRR track closely everywhere in this part; NDCG@10 is what adds information
  over hit@1.
- `top_k=50` throughout, so every metric is a cutoff metric at that depth. Changing
  it makes runs incomparable.
- The easy typo tasks are near saturation and rank poorly as discriminators: their
  top ten configurations span 0.9646-0.9566 (0.8% relative), against 0.4129-0.3579
  (13%) on `typo_brutal`. Prefer `typo_brutal` when comparing close candidates.

## Part 10 — V7 against the systems anyone would use instead

Parts 5-9 compared V7 with itself and with pairwise scorers. None of them put it next to
what a practitioner would reach for: BM25 over the *same character n-grams* (which is what
Elasticsearch's `ngram` tokenizer is), tf-idf cosine over them, token BM25 for documents, and
brute-force Damerau-Levenshtein for typos. Part 10 is that comparison, on 17 tasks, under one
protocol, with the position term ablated and V7 also tried as a reranker of, fusion with, and
interpolant into the BM25 list. Code: `experiments/search_benchmark.py`; every number is in
`experiments/results/search_benchmark_<task>.csv`, the tables in
`experiments/results/search_benchmark_summary.md`. All measured 2026-09-06.

### Protocol

- Each task's queries are split in half by seed (at most 300 per task). **Every system picks
  its configuration on the tune half by MAP and is reported on the test half.** Per-task tuning
  is deliberate — typo correction and entity search are different jobs — and symmetric: BM25
  gets `n`, `k1`, `b` and (see below) an idf exponent swept the same way V7 gets its knobs.
  The untuned V7 default is reported alongside, so the cost of not tuning is visible.
- Ranked lists cut at `top_k=50`, alphabetical tie-break, zero-score documents never returned:
  the V7 rules, applied to every system.
- Grids. V7: `n` from a tier by mean document length (`(1,2),(2,),(2,3),(2,4),(1,2,3)` for
  short strings; `(2,),(3,),(2,3),(2,4),(3,4)` above 60 chars; `(3,),(4,),(3,4)` above 600 —
  see the scifact caveat), `idf ∈ {0,1,2,3}`, `dim ∈ {1,2}`, `normalize`, `position_weight ∈
  {0,0.5,1}`, `denominator`. `bm25_char`: six n-lists × `k1 ∈ {0.5,1.2,2}` × `b ∈
  {0.3,0.75,1}` × `idf^{1,2,3}`. `tfidf_char`: log-tf, Lucene idf, L2, six n-lists.
  `bm25_word`: textbook, `k1 × b`. Damerau-Levenshtein: band 3, typo tasks only.
- Combinations: `X>Y` reranks X's top 50 by Y (Y's ties fall back to X's order);
  `rrf(v7,X)` is reciprocal-rank fusion at k=60; `mix(X,v7)` is `(1-λ)·X/max + λ·v7/max`
  with λ ∈ {0.1,0.25,0.5,0.75,0.9} tuned like everything else.
- ⚠ Timing: every ms/query below was measured with two or three benchmark processes sharing
  the laptop. Ratios between systems on the same task are meaningful; absolute values are not.

### The tasks

| regime | task | corpus | mean chars | test queries |
|---|---|---|---|---|
| short strings | `typo`, `typo_ms`, `typo_hard`, `typo_brutal` | Part 9's four | 8-9 | 119-125 |
| | `norvig` | 41 433 words; real misspellings from Norvig's `spell-testset1.txt` | 8 | 150 |
| records | `abtbuy`, `ermagellan` | Part 9's two | 53, 138 | 50, 103 |
| | `amazon_google`, `walmart_amazon`, `dirty_walmart_amazon`, `dblp_acm`, `dirty_dblp_acm`, `dblp_scholar`, `fodors_zagats`, `itunes_amazon` | the rest of Magellan in ditto's format, 238-10 693 records | 111-310 | 22-150 |
| documents | `scifact`, `nfcorpus` | BEIR title+abstract, 5 183 / 3 593 docs, graded qrels | ~1 500 | 150 |

The document tasks are the negative boundary: BM25 is the published baseline there
(BEIR reports token-BM25 nDCG@10 0.665 on SciFact, 0.325 on NFCorpus), and a character
n-gram matcher is expected to lose. They are here so that "should this be used for search" has
a number attached at the far end.

### Results (test MAP; nDCG@10 and recall@50 in the summary)

| task | V7 | best baseline | Δ | position (pw1 − pw0) | best combination | Δ vs best single | V7 ms | baseline ms |
|---|---:|---|---:|---:|---|---:|---:|---:|
| typo | 0.966 | bm25_char 0.873 | **+0.092** | +0.011 | mix(bm25,v7) 0.955 | −0.011 | 1 | 0.5 |
| typo_ms | 0.953 | bm25_char 0.828 | **+0.125** | +0.008 | bm25>v7 0.954 | +0.002 | 2 | 0.7 |
| typo_hard | 0.856 | bm25_char 0.722 | **+0.135** | +0.001 | mix(tfidf,v7) 0.852 | −0.004 | 7 | 1.2 |
| typo_brutal | 0.417 | bm25_char 0.138 | **+0.278** | **+0.091** | mix(bm25,v7) 0.393 | −0.024 | 3 | 2.5 |
| norvig | 0.870 | bm25_char 0.797 | **+0.073** | +0.022 | mix(bm25,v7) 0.872 | +0.001 | 6 | 1.4 |
| abtbuy | 0.923 | tfidf_char 0.904 | +0.019 | **−0.033** | bm25>v7 0.923 | 0 | 4 | 0.9 |
| ermagellan | 0.878 | tfidf_char 0.935 | **−0.057** | +0.008 | mix(bm25,v7) 0.936 | +0.001 | 75 | 9.4 |
| dblp_acm | 0.993 | bm25_char 1.000 | −0.007 | 0 | v7>bm25 1.000 | 0 | 49 | 7.7 |
| dblp_scholar | 0.765 | tfidf_char 0.780 | −0.014 | −0.003 | mix(tfidf,v7) 0.781 | +0.002 | 52 | 7.6 |
| amazon_google | 0.865 | tfidf_char 0.872 | −0.007 | +0.019 | mix(tfidf,v7) 0.877 | +0.005 | 17 | 7.3 |
| walmart_amazon | 0.898 | bm25_char 0.886 | +0.013 | +0.011 | mix(tfidf,v7) 0.903 | +0.004 | 32 | 7.9 |
| fodors_zagats | 1.000 | 1.000 | 0 | 0 | 1.000 | 0 | 15 | 2.8 |
| itunes_amazon | 0.981 | tfidf_char 0.981 | 0 | 0 | 0.981 | 0 | 67 | 3.3 |
| dirty_dblp_acm | 0.993 | bm25_word 1.000 | −0.007 | 0 | rrf(v7,tfidf) 0.997 | −0.003 | 53 | 1.0 |
| dirty_walmart_amazon | 0.872 | bm25_char 0.879 | −0.007 | +0.014 | v7>bm25 0.879 | 0 | 72 | 16.0 |
| scifact | 0.618 | bm25_char 0.637 | −0.019 | +0.018 | v7>bm25 0.638 | +0.001 | 20 | 23.8 |
| nfcorpus | 0.112 | bm25_char 0.121 | −0.009 | +0.003 | rrf(v7,tfidf) 0.122 | +0.001 | 3 | 1.6 |

The noise floor: with 150 test queries, one query moving between rank 1 and 2 is 0.003 MAP;
on `abtbuy` (50 queries) it is 0.010. Anything inside ±0.01 above is one or two queries.

**Three regimes, and the answer differs in each.**

1. **Short strings: V7 is the best system by a wide margin.** +0.07 to +0.28 MAP over the
   best baseline on all five tasks, at 1-7 ms/query; it also beats brute-force
   Damerau-Levenshtein (0.921 / 0.943 / 0.805 / 0.017 / 0.779 on the five) at 200-1400 ms.
   That includes `norvig`'s real human misspellings, not just `corrupt`'s random edits.
   Token BM25 scores 0 everywhere here (a misspelled word is not a token in the index); char
   BM25 at its best is 0.87 / 0.83 / 0.72 / 0.14 / 0.80.
2. **Records: a wash, at 5-10x the cost.** V7 is within ±0.02 of the best baseline on 9 of
   10 tasks and never clearly ahead (+0.019 on abtbuy is two queries of 50); it loses **0.057
   on ermagellan**, the one long-record task where the gap is real (0.878 vs 0.935; nDCG@10
   0.906 vs 0.949). Four tasks are saturated (≥ 0.98 for everything) and say nothing.
3. **Documents: V7 loses narrowly and should not be used.** nDCG@10 0.650 vs 0.679
   (bm25_char) / 0.665 (bm25_word) on scifact, 0.273 vs 0.298 / 0.294 on nfcorpus. The
   token BM25 reproduces BEIR's 0.665 on scifact exactly and lands under its 0.325 on
   nfcorpus (no stemming, no stopwords, our 150-query half). V7 only gets this close with
   `normalize=False` (containment rather than similarity — a 10-word query against a 250-word
   abstract) and `n=(4,)`; the library default scores **0.029** here at 500 ms/query.

### Does positional information help? — yes on short strings, noise elsewhere

`position_weight=0` is order-blind dice over n-gram multisets, so `v7_pw1.0 − v7_pw0.0`, each
tuned over everything else, is the value of the mover's-distance term per task. It is
**+0.091 on typo_brutal**, +0.022 on norvig, +0.011 / +0.008 / +0.001 on the other typo
tasks; on the records +0.011 to +0.019 on amazon_google, walmart_amazon and
dirty_walmart_amazon, +0.008 on ermagellan, zero on the four saturated tasks, **and never below
−0.003 except abtbuy (−0.033, three queries of 50)**; +0.018 / +0.003 on the documents. Position
is the whole signal when the error is a transposition inside a short string; on a 150-character
record it is one more slightly-noisy feature, worth about the noise floor. Part 9's "position is
harmful on products" came from abtbuy alone and does not generalize — but neither does it help
enough on records to matter.

### Does it help as a tie-breaker, reranker or fusion partner? — no

This was the weak-form hypothesis: order information "should not make me worse, at least for
breaking ties or reranking the top few". Measured three ways against the best baseline on each
task:

- **Reranking the top 50 is just the reranker.** On all twelve record and document tasks
  `bm25_char>v7` equals V7 alone and `v7>bm25_char` equals BM25 alone, to within 0.003:
  recall@50 is 1.000 for both on every record task, so the first stage admits everything and
  only the second stage's ordering remains. On the short-string tasks the first stage *does*
  bound recall (BM25 R@50 0.456 on typo_brutal) and reranking by V7 gets 0.313 against V7's
  own 0.417; the other way round, V7's better candidate set lifts BM25 by 0.01-0.06 but never
  to V7's level.
- **RRF is never best.** At most +0.003 over its better input (nfcorpus), usually between the
  two, three times below both (dblp_scholar, walmart_amazon, dirty_walmart_amazon).
- **Tuned interpolation (`mix`) recovers the better input at best.** The tuned λ sits at the
  endpoints — 0.9 (nearly all V7) on every short-string task, 0.1 on most record and
  document tasks. The mixture beats the best single system on a task by at most **+0.005**
  (amazon_google, walmart_amazon: one query), and is *below* its own better input by
  0.01-0.03 on the five short-string tasks and on abtbuy, itunes_amazon and walmart_amazon,
  where the weaker system's normalized score leaks in.

So on the tasks where BM25 wins, V7's ordering information is not complementary to it: the
queries BM25 gets wrong are not the ones V7 gets right. ⚠ This is a result about *this
score*, not about order in general: V7's position term is a coarse per-n-gram displacement
inside a Dice frame, and the next section says the Dice frame is where the loss is.

### If BM25 does well, what is V7 missing? — the numerator, not the position term

Answered on ermagellan, the one record task with a real gap, in `.scratch/probe_ermagellan_gap.py`
(2026-09-06, numbers transcribed here because that directory is gitignored). Starting from tf-idf
cosine at `n=(2,3)` — log-tf, idf¹, L2 — test MAP **0.935**, and replacing one piece at a time
with V7's choice:

| change | test MAP | cost |
|---|---:|---:|
| tf-idf cosine, log-tf, idf¹, L2 (the winning baseline) | 0.935 | — |
| … idf² instead of idf¹ | 0.939 | +0.004 |
| … L1 totals under a geometric mean instead of L2 (V7's `geo` denominator) | 0.896 (idf¹), 0.930 (idf²) | −0.039 / −0.009 |
| **`Σ idf³·min(c_q, c_d)` instead of `Σ idf²·c_q·c_d`** — V7's numerator, pooled over n, geo | 0.899 | **−0.036** |
| … the same, averaged per n instead of pooled (V7's `_similarity_vectors` does this) | 0.869 (tune 0.914) | −0.030, noisy |
| V7 proper at `pw=0`, `n=(2,3)`, idf 3, geo | 0.875 | ≈ the row above |
| V7 proper at `pw=1` (position term on) | 0.883 | +0.008 |

Three things, in order of size:

1. **The `min` numerator.** V7 scores the *transported mass* — `min(c_q, c_d)` per n-gram,
   because that is what a mover's distance moves — where cosine and BM25 score the *product*
   `c_q · c_d` (BM25: `tf` saturated). The product rewards a document for containing a rare
   n-gram *many* times; `min` caps that at the query's count, which is almost always 1. On
   records with repeated fields (brand in title and description) the product is the better
   evidence. This is −0.036 by itself and is structural: it is what makes the score an EMD.
2. **L2, idf-weighted length normalization.** Cosine divides by the L2 norm of the
   idf-weighted counts, so a long tail of common n-grams costs a document little; V7's `geo`
   divides by `sqrt(|q|₁·|d|₁)` on raw counts. Worth 0.01-0.04 depending on the idf exponent
   (Part 9 found `normalize` and `idf` substitute for each other; this is why).
3. **Per-n averaging.** V7 computes a similarity per n and averages; the baselines pool all
   n-grams into one vector. The tune/test disagreement (−0.030 test, +0.010 tune) says this one
   is within noise on 103 queries.

The position term is **+0.008** on this task — it is not where the gap is, and turning it off
does not close the gap either.

⚠ **The BM25 baseline was under-tuned until 2026-09-06, and that would have flattered V7.**
Textbook BM25 uses idf¹; the observation that tf-idf *cosine* beat it on every record task was
explained by the cosine dot product carrying idf on both sides — idf². Adding `idf_exp ∈ {1,2,3}`
to the BM25 grid (`.scratch/probe_bm25_vs_tfidf.py`, `n=(2,3)`, best of k1/b at each exponent):
ermagellan 0.862 → 0.903 → **0.917**, abtbuy 0.872 → 0.896 → **0.932**. The selected `bm25_char` uses idf³ on abtbuy,
ermagellan and itunes, idf² on the walmart and dirty tasks, idf¹ on typo and documents. This is
the same finding as Part 6's "idf is worth 5-15x what the hyperparameters are worth", now on
the baseline side; a comparison against textbook BM25 alone would have reported V7 *ahead* of
BM25 on ermagellan by 0.016 instead of behind by 0.054 — it was the cosine baseline that exposed it. **V7's own idf exponent is not grid-limited**:
past the grid edge on ermagellan it goes 3 → 0.883, 4 → 0.895, 5 → 0.871, 6 → 0.851, 8 → 0.768.

### Should it be used for text search?

- **Fuzzy lookup of short strings** (spelling correction, dictionary / name / code lookup,
  anything under ~20 characters where the error model is edits): **yes, and it is the best
  thing measured here**, with the position term on. Nothing in the baseline set comes close,
  and it is 100x faster than the edit-distance scan that would be the honest alternative.
- **Record / entity search** (product titles, citations, 50-300 characters): **not on
  quality grounds.** It matches char-n-gram BM25 or tf-idf cosine to within noise on nine of
  ten tasks and loses on the tenth, at 5-10x the query cost, and combining it with BM25 buys
  nothing. Use it only if one index has to serve both regimes.
- **Documents**: no. It is behind BM25 with a containment normalization and behind
  everything at its defaults.

### Caveats

- ⚠ **`n=2` was excluded from the V7 grid on the document tasks.** With 1 500-character
  documents every bigram is in every document; on scifact `n=(2,)` measured nDCG@10 0.296 at
  550 ms/query against 0.523 at 90 ms for `(3,4)` in the preview (before the full grid, so not
  the numbers above), and the full grid with it would have taken ~8 hours. The document tier
  is `(3,),(4,),(3,4)`; both tasks chose `(4,)`.
- The four saturated Magellan tasks (`dblp_acm`, `dirty_dblp_acm`, `fodors_zagats`,
  `itunes_amazon`) distinguish nothing at ≥ 0.98 and are kept only for completeness.
- `dblp_scholar`'s V7 timing rows predate the baseline refresh (its V7 rows were kept and only
  the baselines re-run, via `--baselines`), so its ms/query column mixes two runs.
- Selected configurations are per task and at most 150 tune queries chose them; the
  `selected configurations` table in the summary shows how unstable `n` is across the record
  tasks (`(2,)`, `(3,)`, `(2,3)`, `(2,4)`, `(3,4)` all appear). A single "records" setting
  would score lower than the per-task numbers above.
- The interpolation normalizes each system by its per-query max; a z-score or min-max over
  the top 50 might behave differently at λ near 0.5. Not tried; the λ=0.1/0.9 endpoints
  winning everywhere says the two scores are not complementary at any scale.
