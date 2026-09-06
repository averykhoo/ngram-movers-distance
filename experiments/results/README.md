# Parameter-search results

Tracked output of `experiments/index_param_search.py`. Measured 2026-09-05 / 2026-09-06 against
`ApproxWordListV7`; the analysis is written up in `docs/bow-plan.md` Part 9.

**These are the durable copy.** The run logs under `.scratch/` are gitignored and disposable --
every number quoted in Part 9, HANDOFF.md and the README comes from the CSVs here.

## Files

| file | what it is |
|---|---|
| `index_param_search_<task>_unified.csv` | the **unified grid**: one identical 432-point grid per task. This is what cross-task comparison uses. |
| `index_param_search_overall.csv` | cross-task aggregate written by `--aggregate`: per-config mean rank, worst rank, mean per-task-normalized MAP, and each task's MAP |
| `index_param_search_typo.csv`, `..._abtbuy.csv` | **stage 1**, a different 400-point grid that predates `position_weight` and `denominator`. Kept for the record; not comparable to the unified files. |
| `..._stage2.csv` | stage 2: the two newer knobs swept over stage 1's best configurations, plus a spread of the rest as a regression check |
| `..._merged.csv` | stage 1 + stage 2 deduplicated |

## Columns

`n`, `idf_exponent`, `dim`, `normalize`, `position_weight`, `denominator` are the configuration.
`map`, `ndcg@10`, `ndcg@5`, `mrr`, `hit@1`, `hit@10`, `recall@10`, `p@10`, `rprec` are the
metrics, each averaged over the task's queries. `build_sec` / `eval_sec` are timings.

## ⚠ Two things that will mislead you if unnoticed

1. **`ermagellan` covers 270 of the 432 grid points, not all of them.** The 162 configurations
   containing `n=1` were excluded: each costs ~341 s there and scores MAP ~0.008, because every
   138-character document contains every letter. So any aggregate *including* `ermagellan` ranks
   only those 270 unigram-free configurations, and its per-task bests are lower than the
   five-task numbers wherever the true optimum used unigrams (`typo` peaks at 0.9058 in the
   restricted field against 0.9646 with `n=(1,2)` available). Do not mix rows between the two.

2. **Every metric is a cutoff metric at `top_k=50`.** A run with a different `top_k` is not
   comparable to these.

Also note the four typo tasks have exactly one right answer per query, so on them MAP is
numerically identical to MRR.

## Reproducing

```bash
python experiments/index_param_search.py all --unified          # full grid, every task
python experiments/index_param_search.py ermagellan --unified --exclude-n=1
python experiments/index_param_search.py typo,typo_ms,typo_hard,typo_brutal,abtbuy,ermagellan --aggregate
```

`abtbuy` and `ermagellan` need the gitignored datasets in `.scratch/data/`; the four typo tasks
are self-contained. See the module docstring for where to download the other two.
