# Histogram membership-inference attack / privacy audit for tabular PE

This is the Aug-PE nearest-neighbor histogram MIA
(`aug-pe-baseline/attacks/`) ported to the tabular Private Evolution pipeline in
this repo. **The attack algorithm is unchanged** — same NN Voronoi assignment,
same in-distribution Poisson occupancy null, same per-round pure / noised
log-likelihood ratios, and same multi-round aggregation. Only the data plumbing
(text → tabular) and one observation-model detail (the DP mechanism's noise model)
were adapted.

## Why it leaks (unchanged)

For each PE iteration and class, every private record votes for the Voronoi cell
of its nearest *synthetic* sample (in embedding space). The released synthetic
embeddings + their vote counts are public, so an attacker recomputes that same
assignment for any candidate `x`, reads the count in the cell `x` lands in, and
forms a per-round likelihood ratio `log P(count | x member) − log P(count | x non)`
with the null per-cell occupancy estimated from non-private reference data
(`Poisson(N·q_j)`; membership adds `x`'s own +1). Aggregating over the `T`
iterations amplifies the signal and averages out the independent DP noise.

## What changed vs. `aug-pe-baseline/attacks/`

| file | status | change |
| --- | --- | --- |
| `evaluate.py` | **verbatim** | AUC / TPR@FPR / certificate metrics — byte-for-byte copy. |
| `histogram_mia.py` | **core unchanged** | `nearest_cell`, `cell_occupancy`, `llr_pure`, `llr_noised`, `score_records`, multi-round aggregation are identical. **One adaptation:** added a `censored` flag to the noised observation model. The text pipeline stored `max(k+noise, threshold) − threshold` (left-censored Gaussian); this repo's mechanism (`pe/dp/gaussian.py`) releases `clean + N(0, σ²)` with **no clipping**, so the tabular driver passes `censored=False` (plain Gaussian density at every `y`, including negative). |
| `tabular_embedding.py` | **new (replaces text encoder)** | The text attack embedded records with a learned sentence encoder. Tabular PE uses the deterministic `pe.embedding.TabularEmbedding` (min-max numerics + weighted one-hot). We reuse it with `info` from `TabularCSV.get_tab_info()` — the same embedding the histogram was built in (verified: recomputed cell embeddings match the saved ones exactly, max abs diff = 0). |
| `audit_set.py` | **adapted** | Same member/non-member construction; records are now feature rows keyed by integer label-id instead of text strings keyed by `label1\tlabel2`. Members = private train rows; non-members = test split (deduplicated against members). |
| `reconstruct.py` | **rewritten (same output contract)** | Text version read counts from `{t}/count_class/*.csv` and candidate texts from `{t-1}_all/samples.csv`, then re-embedded. Tabular PE persists everything in one place: each checkpoint `{checkpoint}/{t:09d}/data_frame.pkl` already holds, for every voted cell, its embedding (`PE.EMBEDDING.TabularEmbedding`) **and** its released counts (`PE.CLEAN_HISTOGRAM`, noised `PE.DP_HISTOGRAM`). So we just load each checkpoint, keep the rows carrying a histogram value (the cells voted on that round, kept by the `keep_selected` population), and group by label-id. Emits the identical `{class: [iteration_dict, …]}` structure `score_records` consumes. |
| `run_mia.py` | **adapted driver** | Same 4-step pipeline and same CLI shape. Takes `--checkpoint_folder` + the dataset CSVs instead of `--result_folder` + a feature-extractor name. **Default regime = `noised`** (reads the released `PE.DP_HISTOGRAM`). `n_private` per class is taken from the public class sizes rather than `round(sum(clean_count))`, because only the *selected* cells are persisted (see caveat). |

## Caveats specific to this pipeline

- **Released cells are the kept/selected subset.** The tabular run only persists
  the cells the `keep_selected` population carries forward (≈150/round on
  breast-cancer), not the full candidate set. Consequences: (a) `n_private` is
  taken from known class sizes, not the (truncated) count sum; (b) the *pure*
  regime's exact non-membership certificate is no longer one-sided — a member
  whose true vote went to a non-selected cell can land in a count-0 selected cell
  and be wrongly certified. **The noised regime (the default) has no certificate
  path and is unaffected.** Usable rounds are the `keep_selected` iterations
  (t = 5..19 on the default breast-cancer config → 15 rounds).
- Requires the run to have used `lookahead_degree = 0` (the tabular examples do),
  so each cell's indexed embedding equals its per-sample embedding.

## Run

```bash
python -m attacks.run_mia \
  --checkpoint_folder results/tabular/breast-cancer_composite_population/checkpoint \
  --noise_multiplier 1.853642779702104 \
  --regime noised
```

`--noise_multiplier` is the value printed in the run's `log.txt`
(`DP epsilon=…, noise_multiplier=…`). CSV paths default to the hosted
breast-cancer files used by `example/tabular/breast_cancer.py`. `--regime both`
also runs the `raw` baseline and the `pure` (clean-histogram) upper bound.

## Modules

- `histogram_mia.py` — NN assignment, reference-occupancy null, per-round LLRs
  (pure + Gaussian noised), multi-round aggregation. *(core identical to original)*
- `reconstruct.py` — load checkpoints, align cells ↔ released counts per class.
- `audit_set.py` — labelled members (private) vs non-members (holdout).
- `tabular_embedding.py` — deterministic `TabularEmbedding` embed function.
- `evaluate.py` — AUC, TPR@FPR, certificate metrics. *(verbatim)*
- `run_mia.py` — end-to-end driver.
