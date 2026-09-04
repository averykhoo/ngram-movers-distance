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
| Length | `min(|Q|,|D|) / max(|Q|,|D|)` | lengths only (the README's `real_quick_ratio` todo) |
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
  skipped — this is the README's "add a min_similarity filter" todo.
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
  multi-n combination beats plain n=2. `n=1` is unavailable anyway
  (`nmd/nmd_core.py:34`).
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

### Prior art to check before implementing

- Word Mover's Distance retrieval (Kusner et al. 2015): WCD/RWMD bounds,
  prefetch-and-prune — Part 1 is this with NMD as the ground distance.
- Gale–Church / block sentence alignment: the `λ = ∞` DP is the same shape.
- Edit distance with block moves (Shapira & Storer; Cormode & Muthukrishnan) is
  the general "jumps anywhere" problem and is NP-hard; restricting jumps to
  token boundaries with a small `L` is what keeps this tractable.
- "Extended edit distance" (previously evaluated; the reference implementation
  was found buggy) — not used here.
