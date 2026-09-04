"""5-/6-/7-way TVD for the tabular fidelity tables.

Reuses ComputeTVD (pe/callback/tabular/compute_tvd.py) exactly as the existing
1-/2-way TVD in utility_summary.json were produced -- just at higher degree.
Those weren't logged during the original PE runs (only degree=1,2 were wired
into the generation scripts), so this recomputes them retroactively from each
run's saved final-iteration checkpoint (the same "filtered to fold=-1
released set" convention as everything else in utility_summary.json) plus the
real private train data, and separately for the real-vs-real reference point
(real train vs real test, matching real_reference.py).

Two distinct "can't compute" cases, reported differently (never silently 0.0):
  1. Too FEW columns: ComputeTVD's combinations(feature+label_columns, degree)
     is empty when degree > total column count, and the existing code returns
     0.0 for that -- which would misreport "not computable" as "perfect
     fidelity". Checked via total_cols < degree.
  2. Too MANY columns: e.g. breast-cancer has 31 columns, so degree=6/7 mean
     C(31,6)=736281 / C(31,7)=2629575 combinations, each a full value_counts
     groupby over the reference dataframe -- computationally intractable
     (hours-to-days), not a modeling degeneracy. Checked via a hard combo-count
     cap (MAX_COMBOS); anything over is marked infeasible with the exact count
     rather than run indefinitely.

Adds 5way_tvd / 6way_tvd / 7way_tvd fields to every row in
results/tabular/_utility_logs/utility_summary.json and to real_reference, plus
a "*_status" field ("ok" / "too_few_columns" / "combinatorially_infeasible")
so a null value's reason is never ambiguous.
"""

import glob
import json
import math
import os
import sys
import time

from pe.data import Data
from pe.data import TabularCSV
from pe.callback import ComputeTVD
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

ROOT = "results/tabular"
OUT = f"{ROOT}/_utility_logs/utility_summary.json"
BASE = "https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/heads/main/tabular/real"
# Fast datasets first so partial progress is useful even if a slow one stalls.
DATASETS = ["adult", "breast-cancer"]
DEGREES = [5]
MAX_COMBOS = 10_000_000_000  # cutoff removed per explicit request -- compute regardless of cost

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


def _tvd_value(items):
    return float(items[0].value) if items else None


def _status_and_cbs(total_cols, priv_data, filter_criterion):
    """Return {degree: (status, callback_or_None)}."""
    out = {}
    for d in DEGREES:
        if total_cols < d:
            out[d] = ("too_few_columns", None)
            continue
        n_combos = math.comb(total_cols, d)
        if n_combos > MAX_COMBOS:
            out[d] = (f"combinatorially_infeasible ({n_combos} combinations > cap {MAX_COMBOS})", None)
            continue
        out[d] = ("ok", ComputeTVD(priv_data=priv_data, degree=d, filter_criterion=filter_criterion))
    return out


def compute_dataset(slug):
    priv_data = TabularCSV(csv_path=f"{BASE}/{slug}/{slug}_train.csv",
                           metadata_path=f"{BASE}/{slug}/{slug}_metadata.json")
    test_data = TabularCSV(csv_path=f"{BASE}/{slug}/{slug}_test.csv",
                           metadata_path=f"{BASE}/{slug}/{slug}_metadata.json")

    feature_cols = (priv_data.metadata["cat_columns"] + priv_data.metadata["int_columns"]
                    + priv_data.metadata["float_columns"])
    label_cols = priv_data.metadata["label_columns"]
    total_cols = len(feature_cols) + len(label_cols)
    log(f"  {slug}: {len(feature_cols)} feature + {len(label_cols)} label = {total_cols} total columns")

    per_run_status = _status_and_cbs(total_cols, priv_data, {VARIATION_API_FOLD_ID_COLUMN_NAME: -1})
    real_status = _status_and_cbs(total_cols, priv_data, None)
    for d in DEGREES:
        s, _ = per_run_status[d]
        if s != "ok":
            log(f"    degree={d}: SKIPPED -- {s}")

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
        vals = {}
        for d in DEGREES:
            status, cb = per_run_status[d]
            vals[f"{d}way_tvd_status"] = status
            if status != "ok":
                vals[f"{d}way_tvd"] = None
                continue
            t0 = time.time()
            vals[f"{d}way_tvd"] = _tvd_value(cb(syn_data))
            log(f"    {budget} degree={d}: {vals[f'{d}way_tvd']:.6f}  ({time.time()-t0:.1f}s)")
        per_run[budget] = vals

    real_vals = {"total_columns_available": total_cols}
    for d in DEGREES:
        status, cb = real_status[d]
        real_vals[f"{d}way_tvd_status"] = status
        if status != "ok":
            real_vals[f"{d}way_tvd"] = None
            continue
        t0 = time.time()
        real_vals[f"{d}way_tvd"] = _tvd_value(cb(test_data))
        log(f"    real-reference degree={d}: {real_vals[f'{d}way_tvd']:.6f}  ({time.time()-t0:.1f}s)")

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
                for d in DEGREES:
                    row[f"{d}way_tvd"] = None
                    row[f"{d}way_tvd_status"] = "no_checkpoint_found"
                continue
            row.update(vals)

        summary.setdefault("real_reference", {}).setdefault(slug, {}).update(real_vals)
        save(summary)
        log(f"  ({slug} done in {time.time()-t_ds:.1f}s)")

    log("\nALL DATASETS DONE")


if __name__ == "__main__":
    main()
