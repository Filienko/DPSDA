# Running summary — histogram MIA against tabular Private Evolution

**Handoff doc.** What each attack signal is, which experiments used it and how, what we
learned, and where every result file lives. All experiments are on **tabular Aug-PE /
DPSDA** synthetic-data runs; the headline setting is **ε = 10** on four datasets
(artificial-characters, breast-cancer, adult, person-activity).

---

## 0. Setup / threat model (read first)

- **What PE releases each round:** a nearest-neighbor **Voronoi histogram** — synthetic
  "cells" and a **DP-noised vote count** per cell (`clean_count + N(0, σ²)`, uncensored;
  `pe/dp/gaussian.py`). The attacker sees only these noisy releases.
- **Audit set** (`attacks/audit_set.py`): **members** = the private training records used
  for generation; **non-members** = an in-distribution holdout split of the test set. A
  second disjoint half of the test set is the **reference set D_ref** used to build the
  attack's null model (`ref_holdout_frac=0.5`).
- **Embedding** (`attacks/tabular_embedding.py`): min-max-normalized numerics + weighted
  one-hot categoricals. Every distance/nearest-cell computation is in this space.
- **σ (noise multiplier) at ε=10:** artificial-characters 1.839, breast-cancer 1.854,
  adult 2.814, person-activity 2.044. All attack numbers below are on these **DP-noisy**
  releases.

---

## 1. Signals and attacks — what we used and how

The attack framework is **`attacks/histogram_mia.py`**. It exposes three membership
signals; every experiment picks one (or combines them).

| Signal | Function (`attacks/histogram_mia.py`) | What it reads | How it scores |
|---|---|---|---|
| **count** (= "the histogram attack"; baseline) | `score_records(regime="noised")` | the **DP-noised vote count** in the Voronoi cell a record lands in | per-cell noised log-likelihood ratio vs a Poisson null μ0=N·q_j (non-member) vs μ1=(N−1)·q_j (member); q_j = per-cell occupancy from D_ref (`cell_occupancy`); summed over rounds. Models σ explicitly via a Gaussian observation model (`llr_noised`). |
| **lineage** (geometry) | `lineage_density_records` | **positions of surviving synthetic cells** over rounds (NOT counts) | time-averaged Parzen density of survivor cells at the record, calibrated to a D_ref density ratio. Untouched by count noise — DP noise enters it only via which cells were selected during generation. |
| **combined / tuned** | `attacks/improved.py` → `score_config` | count + optional add-ons | count with toggleable components: k-NN multi-vote (`pool_m`), SNR/inv-var round weighting, per-class LiRA calibration, selection-bias model, and z-scored fusion with lineage (`density`). All-off ≡ count baseline. |

**Where the algorithm lives / what you run:**
- Core algorithm: `attacks/histogram_mia.py`.
- Baseline attack, end-to-end: **`attacks/run_mia.py`** (batch: `attacks/run_attack_all.sh`).
- Best/tuned attack + ablation: **`attacks/improved.py`**, driven by **`attacks/ablate.py`**.

---

## 2. Experiments — what, key result, and where the results are

All result files are under `results/tabular/`. The analysis JSON+figures live in
`results/tabular/dist_shift/analysis/` (abbreviated **ANALYSIS/** below).

### A. Best baseline/tuned attack AUC per dataset (ε=10, all members)
- **Script:** `attacks/ablate.py` (uses `attacks/improved.py`); baseline via `attacks/run_mia.py`.
- **Result files:** `results/tabular/_mia_logs/ablation_summary.json`,
  `results/tabular/_mia_logs/mia_summary.json`, per-run `mia_report_{ref,noref}.json`
  next to each `results/tabular/<dataset>_composite_population/`.
- **Result (ε=10 overall AUC, baseline → best config):** artificial-characters 0.633→0.639
  (selection), breast-cancer 0.575→0.580 (lira), adult 0.520→0.521, person-activity
  0.497→0.510 (lira). **Tuning helps only marginally (≤0.01).**
- **Learned:** at ε=10 the *overall* leakage is modest (0.50–0.64). DP is working; the
  clever components (pooling, SNR weighting, LiRA, selection) barely move overall AUC.

### B. Distribution-shift sweep (the confound)
- **Scripts:** `example/tabular/dist_shift/make_natural_splits.py` (natural PCA-1 band
  sampling), `run_pe_shift.py` + `run_natural_all.sh` (PE + MIA per shift level),
  `attacks/dist_analysis.py`, plots in `example/tabular/dist_shift/plot_natural.py`.
- **Result files:** `results/tabular/dist_shift/natural_{iid,q50,q35,q25,q18}/` (runs +
  `mia_report_*.json`); `ANALYSIS/summary.json`; figures `ANALYSIS/fig_marginals.png`,
  `fig_votehist.png`, `fig_auc.png`, `fig_utility.png`.
- **Result:** as the **member↔aux gap** grows (mean-embedding gap 0.015→0.77), apparent
  MIA AUC climbs toward ~1.0 — but a **model-blind distribution-only detector matches it**.
- **Learned:** high AUCs in shifted settings are a **distribution confound, not leakage**.
  The genuine matched-distribution leak at ε=10 is ~0.55–0.60.

### C. Causal isolation of the two distribution axes
- **Scripts:** `attacks/fixed_generator_test.py` (member↔non-member gap),
  `attacks/aux_shift_test.py` (private↔aux reference gap).
- **Result files:** `ANALYSIS/fixed_generator_test.json`, `ANALYSIS/aux_shift_test.json`,
  figure `ANALYSIS/fig_mmd_vs_auc.png`.
- **Result:** (a) widening the **member↔non-member** gap causally inflates AUC 0.55→0.84
  (it's distribution detection). (b) widening the **private↔aux** reference gap (priv-aux
  MMD 0.02→0.50, members & challenges fixed) does **not** inflate — it mildly **degrades**
  AUC 0.60→0.55 while miscalibrating the null.
- **Learned:** the gap that Aug-PE breaks (private≠aux) does not help the attacker; only a
  member/non-member distribution mismatch does, and that's not real membership leakage.

### D. Per-iteration convergence vs leakage
- **Script:** `attacks/per_iteration.py` (artificial-characters, matched, ε=10).
- **Result files:** `ANALYSIS/per_iteration.json`, figure `ANALYSIS/fig_per_iteration.png`.
- **Result:** as PE refines, synth↔private MMD halves (0.093→0.034) and utility rises
  37%→61%, but per-round AUC stays flat ~0.58 and cumulative AUC only 0.61→0.63.
- **Learned:** **fidelity and per-round leakage are decoupled** — DP noise per round bounds
  leakage regardless of how well the synthetic data approximates private. (Tractable
  stand-in for the FM-prior-mismatch scenario; the true FM case needs an algorithm change.)

### E. Outlier disparate impact (the main finding)
- **Script:** `attacks/outlier_disparity.py` (count attack, all 4 datasets, matched, ε=10).
- **Result files:** `ANALYSIS/outlier_disparity_multidataset.json` (and single-dataset
  `outlier_disparity.json`); figures `ANALYSIS/fig_outlier_multidataset.png`,
  `fig_outlier_disparity.png`.
- **Result (count-attack member-vs-nonmember AUC by within-class kNN-outlierness tertile):**
  Spearman(outlierness, score) negative on every dataset (−0.28 to −0.44); **inlier AUC
  0.60–0.73, outlier AUC 0.41–0.52** (outliers at/below chance).
- **Learned:** **inverse disparate impact** — the histogram MIA leaks *typical* records and
  *hides* outliers. This is the OPPOSITE of memorization attacks (DP-SGD/LLM), because the
  attack reads votes out of the synthetic histogram and PE doesn't cover sparse regions, so
  an outlier's cell has count≈0 whether or not it's a member.

### F. Lineage check — is "outliers safe" attack-specific?
- **Script:** `attacks/lineage_check.py` (count vs lineage vs combined, per outlier group).
- **Result files:** `ANALYSIS/lineage_check.json`, figure `ANALYSIS/fig_lineage_check.png`.
- **Result (inlier / outlier AUC, ε=10, matched):**

  | dataset | count | lineage | combined |
  |---|---|---|---|
  | artificial-characters | 0.732 / 0.522 | 0.469 / 0.542 | 0.637 / 0.524 |
  | breast-cancer | 0.668 / 0.408 | 0.459 / **0.581** | 0.576 / 0.496 |
  | adult | 0.617 / 0.439 | 0.552 / 0.417 | 0.606 / 0.408 |
  | person-activity | 0.602 / 0.424 | 0.460 / **0.558** | 0.563 / 0.474 |

- **Learned:** the "outliers are hidden" result is a property of the **count** signal, not
  of PE. A geometry-aware **lineage** attack recovers outlier leakage on 2/4 datasets
  (breast-cancer 0.41→0.58, person-activity 0.42→0.56). Count and lineage are
  complementary — count owns inliers, lineage owns outliers; **no single attack exposes
  both**, and naive z-sum fusion doesn't recover outliers (count's near-chance outlier
  scores add variance).

---

## 3. Key takeaways (one screen)

1. **Genuine ε=10 leakage is low** (overall AUC 0.50–0.64); tuning the attack barely helps.
2. **High AUCs in shifted setups are a distribution confound**, not membership leakage —
   proven causally (Exp C), and matched by a model-blind detector (Exp B).
3. **Fidelity ≠ leakage per round** — DP noise bounds per-round leakage regardless of how
   well synthetic approximates private (Exp D).
4. **Inverse disparate impact** — the standard (count) histogram MIA leaks *typical*
   records and *hides outliers*, general across 4 datasets (Exp E).
5. **But outlier protection is attack-specific** — a geometry-aware (lineage) attacker
   recovers outliers count misses on some datasets (Exp F). Report against the strongest
   attack, not the convenient one.

---

## 4. Reproduce

```bash
cd /home/daniilf/privacy/DPSDA_tab
PY=.venv/bin/python                       # env has generalimport etc.; base conda does NOT

# Baseline + tuned attack, all datasets x eps:
$PY -m attacks.ablate --eps all           # -> results/tabular/_mia_logs/ablation_summary.json
bash attacks/run_attack_all.sh            # -> mia_report_*.json next to each run

# The five analysis experiments (all write to results/tabular/dist_shift/analysis/):
$PY -m attacks.per_iteration              # Exp D
$PY -m attacks.outlier_disparity          # Exp E
$PY -m attacks.lineage_check              # Exp F
$PY -m attacks.fixed_generator_test       # Exp C (member<->nonmember)
$PY -m attacks.aux_shift_test             # Exp C (private<->aux)

# Distribution-shift sweep (Exp B) — regenerate splits/runs then plot:
python example/tabular/dist_shift/make_natural_splits.py
bash   example/tabular/dist_shift/run_natural_all.sh
python example/tabular/dist_shift/plot_natural.py
```

**Env note:** use `.venv/bin/python`, not the active `base` conda env (missing
`generalimport`).

---

## 5. Open items / caveats

- **adult & person-activity** have weak *overall* base signal (~0.50–0.52), so their
  per-group numbers are noisy — **seed-average** before leaning on them.
- **combined fusion** (Exp F) is naive z-sum; a smarter router (count for inliers, lineage
  for outliers) would be the real adaptive attack.
- **True FM-mismatch scenario** (private = data the foundation model never saw; initial
  generator from a different distribution) needs an **algorithm change**, not just
  analysis — Exp D is the tractable stand-in only.

---

## 6. The count attack reads the wrong cell set (measured on real runs)

**Every number in §2 is a lower bound, and the reason is mechanical.** Round `t`
votes on **all** rows of checkpoint `t-1` (`pe/runner/pe.py:203-207`), but top-k
selection keeps ~1 in 4 (`pe/population/pe_population.py:129-134`) and only
survivors retain the histogram and embedding columns, so `reconstruct.py:91`
recovers survivors only. A record whose cell did not survive is scored against an
unrelated, more popular, more distant cell, and `cell_occupancy` renormalises `q`
over the survivors, inflating `mu0 = N*q_j` about fourfold. The `dispersion=1.8` /
`soft_tau=0.02` / `ref_alpha=0.05` tuning is compensating for that, not for a
property of the data.

**How wrong.** Reassign the private records to their nearest cell over each pool
and compare against the released `PE.CLEAN_HISTOGRAM` (`--self_test`). Fraction of
private votes landing in the wrong cell:

| dataset | full pool | survivor-only (what §2 used) |
|---|---|---|
| artificial-characters (4 seeds) | 0 – 1.2e-4 | **0.33 – 0.36** |
| breast-cancer | 0 | **0.13** |
| adult | 1.4e-5 | **0.18** |
| person-activity | 1.2e-5 | **0.35** |

The full-pool residual is nearest-neighbour tie-breaking (on adult, 0.1% of rows
have a NN gap below 1e-5), not a structural error. The baseline misplaces **13-36%
of the entire vote mass**.

Three further channels are released and unread: the **censoring** fact `y_j < tau`
for non-survivors; the **child multiplicity** in the `selection_mode="sample"`
rounds 1-4, a multinomial readout of a histogram those checkpoints never persist
(~36% of the budget, currently zero signal); and the **ancestry graph**
(`PE.PARENT_SYN_DATA_INDEX` is an exact row index into the previous checkpoint).

### What to run

```bash
python -m attacks.run_pe_for_attack --dataset artificial-characters --inout_split
python -m attacks.strong_mia --run_dir <run> --train_csv <run>/audit_members.csv \
       --test_csv <run>/audit_nonmembers.csv --metadata <meta> --self_test
python -m attacks.strong_mia_sweep --runs <run> ... --metadata <meta> ...
```

`attacks/strong_mia.py` is the new attack (nothing existing modified);
`attacks/run_pe_for_attack.py` generates runs; `attacks/strong_mia_sweep.py`
regenerates `attacks/strong_mia_results.json`, from which every number below comes.
`python -m attacks.strong_mia_simtest` checks the machinery with no data at all.

**Note: the dataset store moved.** `tabular/<slug>_train.csv` now 404s; the live
path is `tabular/real/<slug>/<slug>_train.csv`. Every script under
`example/tabular/` (including all 32 in `variants/`) still points at the dead one
and needs a one-line fix; `make_shift_data.py` is the only one already correct.

### Results (eps=10, in/out split -- see below)

AUC, and the delta over the §2 attack scored on the identical audit set:

| dataset | baseline | strong | delta |
|---|---|---|---|
| artificial-characters (4 seeds) | 0.578 +- 0.013 | **0.672 +- 0.002** | **+0.095 +- 0.014** |
| breast-cancer | 0.588 | 0.603 | +0.015 |
| adult | 0.546 | 0.567 | +0.021 |
| person-activity | 0.518 | 0.532 | +0.014 |

TPR@1%FPR on artificial-characters rises 0.021 -> 0.031. Channel attribution
(artificial-characters, 4 seeds):

| channels | AUC |
|---|---|
| baseline (survivor-only pool) | 0.5775 |
| count — **the pool fix alone** | 0.6642 |
| + censored | **0.6723** |
| + multiplicity | 0.6719 |
| geometry alone | 0.5443 |
| all four | 0.6639 |

**The pool fix is ~90% of the gain.** Censoring adds a small but consistent
+0.008 (all 4 seeds). Multiplicity is neutral on mean AUC though it gives the
tightest spread (sd 0.0006 vs 0.0021). The geometry/trajectory channel helped in
simulation but **does not transfer** -- 0.544 alone here, below chance on adult
(0.496), and it costs ~0.008 when fused. It is off by default.

### Protocol change: in/out split

These runs use `--inout_split`: deduplicate the private CSV, train PE on a random
half, use the held-out half as non-members. Two reasons. It removes the train/test
confound of Exp B/C by construction. And on artificial-characters **72% of test
rows exactly duplicate a training feature row**; `audit_set.py:67-68` correctly
drops them, which leaves only 226 usable non-members out of 1533 under the old
protocol. Absolute AUCs are therefore not comparable to §2A (the §2A-style
train/test run gives baseline 0.554 -> strong 0.687, +0.133, on those 226).

### Finding E survives -- and my prediction about it was wrong

I predicted that fixing the pool would *raise* outlier AUC, since the per-round SNR
of the `+1` vote is `1/sqrt(sigma^2 + phi*N*q_j)`, largest where `q_j` is small.
That is wrong on 3 of 4 datasets. Inlier/outlier AUC (member tertile vs all
non-members):

| dataset | baseline in / out | strong in / out |
|---|---|---|
| artificial-characters (mean of 4) | 0.605 / 0.563 | **0.786 / 0.555** |
| breast-cancer | 0.515 / 0.624 | 0.748 / 0.432 |
| adult | 0.604 / 0.527 | 0.530 / **0.594** |
| person-activity | 0.577 / 0.473 | 0.625 / 0.435 |

The corrected attack lifts *inliers* by ~0.18 and leaves outliers flat, so the gap
**widens** from 0.04 to 0.23 on artificial-characters. §2E is not an artifact of
the cell-set truncation; it was understated. adult is the one inversion, and its
overall signal is the weakest (0.55), exactly the noise §5 warns about.

### Caveats

- Fresh PE runs, not the author's, so absolute AUCs differ from §2 by more than
  attack seed noise. The **delta on one run** is the trustworthy quantity.
- Only artificial-characters has 4 seeds; the other three are single runs.
- adult and person-activity audit sets are capped at 4000/4000 by the sweep.
