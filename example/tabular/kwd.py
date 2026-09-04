"""k-way Wasserstein Distance (k-WD), reproduced per the definition:

    k-WD(D, D') = (1/|C_k|) * sum_{S in C_k} WD( D(X_S), D'(X_S) )

where C_k is the set of all k-sized subsets of features (including the label),
D(X_S)/D'(X_S) are the real/synthetic data projected onto those columns, and
WD is the Wasserstein distance, computed with the POT package (as specified).

No implementation of this metric was found anywhere in this repo (grepped for
"wasserstein"/"_wd"/"POT"/"ot.emd" repo-wide -- only per-column 1-D WD and a
single flattened Sinkhorn WD exist elsewhere, neither of which is this k-way
marginal average), so this reproduces it directly from the definition, using
`ot.emd2` (exact OT) as POT's canonical Wasserstein-distance call.

Numeric encoding (a real judgment call the definition doesn't specify):
  - continuous (int/float) columns: min-max scaled to [0,1], scaler fit on the
    REAL data (never on synthetic), matching the same convention used by
    DOMIAS's own WD implementation (src/domias/metrics/wd.py) for consistency
    with a codebase you've already pointed me at for this class of metric.
  - categorical columns: label-encoded (categories fixed from the real data).
  Ground cost = Euclidean distance in the resulting encoded k-dim space;
  WD = ot.emd2(uniform weights, uniform weights, that cost matrix) -- this is
  the standard order-1 empirical Wasserstein distance under the given metric.

Subsampling (an unavoidable practical necessity, not in the definition):
  Exact OT (ot.emd2 / network simplex) is impractical above a few thousand
  points -- person-activity has 115,402 real training rows, cubically
  infeasible. Both sides are subsampled to N_SUB rows (seeded) before each
  pairwise WD call. This is standard practice for empirical WD estimation but
  DOES mean absolute values are not directly comparable to a source that used
  the full dataset or a different N_SUB -- reported honestly, not hidden.
"""

import glob
import json
import math
import os
import time

import numpy as np
import ot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from itertools import combinations

from pe.data import Data, TabularCSV
from pe.constant.data import TABULAR_DATA_COLUMN_NAME, LABEL_ID_COLUMN_NAME, VARIATION_API_FOLD_ID_COLUMN_NAME

ROOT = "results/tabular"
OUT = f"{ROOT}/_utility_logs/utility_summary.json"
BASE = "https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/heads/main/tabular/real"
DATASETS = ["adult", "breast-cancer"]
DEGREE = 2
N_SUB = 1000
SEED = 0

BUDGET_FROM_SUFFIX = {"": "eps10", "_nonoise": "inf"}


def log(msg):
    print(msg, flush=True)


def budget_of(run):
    suffix = run.split("_composite_population", 1)[1]
    return BUDGET_FROM_SUFFIX.get(suffix, suffix.lstrip("_") or "eps10")


def _last_checkpoint_dir(run_dir):
    ckpt_root = os.path.join(run_dir, "checkpoint")
    if not os.path.isdir(ckpt_root):
        return None
    numbered = sorted(
        (d for d in glob.glob(os.path.join(ckpt_root, "*"))
         if os.path.isdir(d) and os.path.basename(d).isdigit()),
        key=lambda d: int(os.path.basename(d)),
    )
    return numbered[-1] if numbered else None


def _load_syn_final(run_dir):
    ckpt_dir = _last_checkpoint_dir(run_dir)
    if ckpt_dir is None:
        return None
    data = Data()
    return data if data.load_checkpoint(ckpt_dir) else None


def _features_df(data, feature_cols, label_cols):
    import pandas as pd
    label_ids = data.data_frame[LABEL_ID_COLUMN_NAME].tolist()
    df = pd.DataFrame(data.data_frame[TABULAR_DATA_COLUMN_NAME].tolist(), columns=feature_cols)
    for col in label_cols:
        df[col] = [data.metadata.label_info[lid].column_values[col] for lid in label_ids]
    return df


def _encode(df_real, df_syn, cat_cols, cont_cols):
    """Fit encoders on REAL data only; return numeric arrays (n_real,k)/(n_syn,k)."""
    enc_real, enc_syn = [], []
    for col in cont_cols:
        scaler = MinMaxScaler().fit(df_real[[col]].astype(float))
        enc_real.append(scaler.transform(df_real[[col]].astype(float)))
        enc_syn.append(scaler.transform(df_syn[[col]].astype(float).clip(
            lower=df_real[col].min(), upper=df_real[col].max())))
    for col in cat_cols:
        le = LabelEncoder().fit(df_real[col].astype(str))
        classes = set(le.classes_)
        enc_real.append(le.transform(df_real[col].astype(str)).reshape(-1, 1).astype(float))
        syn_vals = df_syn[col].astype(str).map(lambda v: v if v in classes else le.classes_[0])
        enc_syn.append(le.transform(syn_vals).reshape(-1, 1).astype(float))
    return np.hstack(enc_real), np.hstack(enc_syn)


def _wd(X, Y, rng):
    nx = min(len(X), N_SUB)
    ny = min(len(Y), N_SUB)
    xi = rng.choice(len(X), size=nx, replace=False)
    yi = rng.choice(len(Y), size=ny, replace=False)
    Xs, Ys = X[xi], Y[yi]
    a = np.full(nx, 1.0 / nx)
    b = np.full(ny, 1.0 / ny)
    M = ot.dist(Xs, Ys, metric="euclidean")
    return float(ot.emd2(a, b, M))


def compute_dataset(slug):
    priv_data = TabularCSV(csv_path=f"{BASE}/{slug}/{slug}_train.csv",
                           metadata_path=f"{BASE}/{slug}/{slug}_metadata.json")
    test_data = TabularCSV(csv_path=f"{BASE}/{slug}/{slug}_test.csv",
                           metadata_path=f"{BASE}/{slug}/{slug}_metadata.json")

    cat_cols = priv_data.metadata["cat_columns"]
    cont_cols = priv_data.metadata["int_columns"] + priv_data.metadata["float_columns"]
    label_cols = priv_data.metadata["label_columns"]
    feature_cols = cat_cols + cont_cols
    all_cols = feature_cols + label_cols
    total_cols = len(all_cols)
    log(f"  {slug}: {len(feature_cols)} feature + {len(label_cols)} label = {total_cols} total columns "
       f"-> C({total_cols},{DEGREE})={math.comb(total_cols, DEGREE)} subsets")

    priv_df = _features_df(priv_data, feature_cols, label_cols)
    test_df = _features_df(test_data, feature_cols, label_cols)
    # labels are categorical for this metric regardless of underlying dtype
    cat_set = set(cat_cols) | set(label_cols)
    cont_set = set(cont_cols)

    subsets = list(combinations(all_cols, DEGREE))
    rng = np.random.default_rng(SEED)

    def kwd_against(syn_df):
        vals = []
        for S in subsets:
            cat_S = [c for c in S if c in cat_set]
            cont_S = [c for c in S if c in cont_set]
            Xr, Xs = _encode(priv_df[list(S)], syn_df[list(S)], cat_S, cont_S)
            vals.append(_wd(Xr, Xs, rng))
        return float(np.mean(vals)), float(np.std(vals))

    per_run = {}
    for run_dir in sorted(glob.glob(f"{ROOT}/{slug}_composite_population*")):
        if not os.path.isdir(os.path.join(run_dir, "checkpoint")):
            continue
        run = os.path.basename(run_dir)
        if run.split("_composite_population", 1)[0] != slug:
            continue
        budget = budget_of(run)
        syn_data = _load_syn_final(run_dir)
        if syn_data is None:
            log(f"    {budget}: no checkpoint found, skipping")
            continue
        syn_data = syn_data.filter({VARIATION_API_FOLD_ID_COLUMN_NAME: -1})
        syn_df = _features_df(syn_data, feature_cols, label_cols)
        t0 = time.time()
        mean_wd, std_wd = kwd_against(syn_df)
        log(f"    {budget}: {DEGREE}-WD = {mean_wd:.6f} +/- {std_wd:.6f}  ({time.time()-t0:.1f}s, "
           f"n_syn={len(syn_df)})")
        per_run[budget] = {f"{DEGREE}way_wd_mean": mean_wd, f"{DEGREE}way_wd_std": std_wd}

    t0 = time.time()
    real_mean, real_std = kwd_against(test_df)
    log(f"    real-reference: {DEGREE}-WD = {real_mean:.6f} +/- {real_std:.6f}  ({time.time()-t0:.1f}s)")
    real_vals = {f"{DEGREE}way_wd_mean": real_mean, f"{DEGREE}way_wd_std": real_std,
                "n_sub": N_SUB, "n_subsets": len(subsets)}

    return per_run, real_vals


def save(summary):
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"  -- saved {OUT}")


def main():
    with open(OUT) as f:
        summary = json.load(f)

    for slug in DATASETS:
        log(f"=== {slug} ===")
        t_ds = time.time()
        per_run, real_vals = compute_dataset(slug)

        for row in summary["rows"]:
            if row["dataset"] != slug:
                continue
            vals = per_run.get(row["budget"])
            if vals is None:
                row[f"{DEGREE}way_wd_mean"] = None
                row[f"{DEGREE}way_wd_std"] = None
                continue
            row.update(vals)

        summary.setdefault("real_reference", {}).setdefault(slug, {}).update(real_vals)
        save(summary)
        log(f"  ({slug} done in {time.time()-t_ds:.1f}s)")

    log("\nALL DATASETS DONE")


if __name__ == "__main__":
    main()
