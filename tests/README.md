# Tests for N-gram Mover's Distance

Counts below were measured 2026-09-05 (`992 passed, 1 xfailed`). They will drift — re-run
`pytest --collect-only -q` rather than trusting them.

## Running

```bash
pytest                                    # everything
pytest tests/test_nmd_core.py             # one file
pytest tests/test_index_v6.py -k ZeroDiv  # one pattern
```

Some files need optional dependencies, since `nmd` itself has none (see the dependency table
in the top-level README):

| file | needs |
|---|---|
| `test_bow.py`, `test_segments.py` | `scipy`, `numpy` |
| `test_index_v7.py`, `test_index_v7_idf.py` | `numpy` |
| `test_word_set.py`, `test_word_set_idf.py` | `pyroaring`, `regex` |
| `test_parity_with_nmd.py` | `numpy`, `pyroaring`, `regex` for some classes; skips per fixture |

`test_parity_with_nmd.py` also reads `experiments/words_en.txt` and skips the whole module if
it is missing — it is the only test file that needs a corpus.

## Test files

| file | count | covers |
|---|---:|---|
| `test_nmd_core.py` | 15 | `ngram_movers_distance` — the metric itself, `invert` / `normalize`, `n=1` |
| `test_emd_correctness.py` | 4 | the 1-D EMD kernels against each other: symmetry, edge cases, `emd_1d_old` as an oracle |
| `test_index_v6.py` | 79 | `ApproxWordListV6`, i.e. the shipped `WordList` — the two bugs fixed 2026-09-05, and guards that the frozen V3/V5 behaviour and the `n=(2, 4)` default did **not** move |
| `test_index_v7.py` | 124 | `ApproxWordListV7`: container protocol, and that pruning never drops a true top-k |
| `test_index_v7_idf.py` | 139 | V7's `idf_exponent` path, including `test_exponent_zero_matches_no_exponent_exactly` |
| `test_word_set.py` | 42 | `WordSet` outside idf: `min_similarity`, defaults, set protocol, unicode, edge cases |
| `test_word_set_idf.py` | 21 | `WordSet`'s incrementally-maintained idf, against a brute-force document frequency |
| `test_bow.py` | 59 | `bow_ngram_movers_distance`, plus `emd_1d_fast ≡ emd_1d_dp` (`TestEmd1dFast`) |
| `test_segments.py` | 50 | `segment_movers_distance` and `align_segments`: split/merge tolerance, `lam` / `mu` |
| `test_find_replace_trie.py` | 5 | `experiments/find_replace_trie.py` — note this is the only file testing code **outside** the `nmd` package |
| `test_parity_with_nmd.py` | 455 | every index's score against `ngram_movers_distance` recomputed from the strings, over 2000 words of `words_en.txt` |

## Things worth knowing

* **`TestEmd1dFast` in `test_bow.py` is what licenses the `emd_1d_fast` substitution.** Both
  `nmd_core` and `ApproxWordListV6` call `emd_1d_fast` rather than `emd_1d_dp`; that class
  pins them equal exhaustively over a quantized grid, on random input, and outside the unit
  interval. If it goes red, two shipped code paths are wrong, not one.
* **`test_segments.py` has one strict `xfail`** — `mass='tokens'` absorbs junk tokens. It is a
  known limitation of that mode, not a bug, and `strict=True` means it fails if it ever passes.
* **A test that passes before *and* after a fix guards nothing.** Both fix-driven files
  (`test_index_v6.py`, `test_word_set.py`) were re-run against the pre-fix tree to confirm
  which tests actually go red; each names them in its module docstring. Do the same when
  adding a regression test here — and where the bug is only reachable *after* a fix (the
  denominator clamps in V6), delete the guard and watch it raise instead.
* **`find_similar`'s scoring is now verified** (HANDOFF #10, closed 2026-09-05 by
  `test_parity_with_nmd.py`). It equals `ngram_movers_distance` exactly — but *pooled*:
  `sum_n similarity / sum_n n-gram count`, one ratio over every n, where V6 and V7 return the
  mean of the per-n ratios. Both are exactly nmd; they are different aggregations of it, and
  they differ by ~0.09 on short words. Do not expect `WordSet` and `WordList` to agree.
* **Two known, deliberate divergences from nmd are pinned rather than fixed**, both in
  `test_parity_with_nmd.py`: `WordSet` keeps the START/END markers at `n == 1` (nmd drops
  them), and V6/V7 score a `0 / 0` normalization as 0.0 where nmd falls back to comparing the
  strings and returns 1.0 (reachable only for identical words of length `<= n - 3`).
* **The parity file was sabotage-checked, not just run green** (2026-09-05). Adding `1e-6` to
  the similarity accumulator in each of V5, V6 (`nmd_index.py`), V7 (`nmd_index_v7.py`) and
  `WordSet` (`nmd_word_set.py`) turned exactly that class's tests red — 164 and 232 failures
  respectively, none of them leaking into the other classes' pins.
