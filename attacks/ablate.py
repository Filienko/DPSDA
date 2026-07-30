"""Phase-2 ablation: score every run under each attack config and tabulate AUC.

For each dataset x epsilon it loads the data once, reconstructs the noised
iterations once (disjoint D_ref, ref_holdout_frac=0.5), then scores every config in
CONFIGS via attacks.improved.score_config. C0_baseline must equal the existing
attack. Writes results/tabular/_mia_logs/ablation_summary.json and prints a table.

Usage:
  python -m attacks.ablate                       # all datasets, eps {inf,10,1}
  python -m attacks.ablate --datasets person-activity --eps inf
  python -m attacks.ablate --eps all
"""

import argparse
import json
import os

import numpy as np

from attacks.audit_set import build_audit_set, reference_rows_by_class
from attacks.reconstruct import reconstruct
from attacks.evaluate import evaluate
from attacks.tabular_embedding import load_private, make_embed_fn
from attacks.improved import score_config, Config

BASE = ("https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/"
        "heads/main/tabular")
ROOT = "results/tabular"
OUT = f"{ROOT}/_mia_logs/ablation_summary.json"

DATASETS = ["adult", "artificial-characters", "breast-cancer", "person-activity"]
# eps tag -> results folder suffix
EPS_FOLDER = {"inf": "_nonoise", "100": "_eps100", "10": "", "5": "_eps5",
              "1": "_eps1", "0.5": "_eps0p5", "0.25": "_eps0p25"}

# All components default OFF in Config(); C0 == current attack.
CONFIGS = {
    "C0_baseline":  Config(),
    "pool3":        Config(pool_m=3),
    "w_snr":        Config(round_weight="snr"),
    "w_invvar":     Config(round_weight="invvar"),
    "lira":         Config(calibrate="lira"),
    "density":      Config(density=True),
    "selection":    Config(selection_model=True),
    "ALL":          Config(pool_m=3, round_weight="snr", calibrate="lira",
                           selection_model=True, density=True),
    # leave-one-out from ALL
    "ALL-pool":     Config(pool_m=1, round_weight="snr", calibrate="lira",
                           selection_model=True, density=True),
    "ALL-wsnr":     Config(pool_m=3, round_weight="uniform", calibrate="lira",
                           selection_model=True, density=True),
    "ALL-lira":     Config(pool_m=3, round_weight="snr", calibrate="none",
                           selection_model=True, density=True),
    "ALL-sel":      Config(pool_m=3, round_weight="snr", calibrate="lira",
                           selection_model=False, density=True),
    "ALL-density":  Config(pool_m=3, round_weight="snr", calibrate="lira",
                           selection_model=True, density=False),
}


def noise_multiplier(folder):
    log = os.path.join(folder, "log.txt")
    if not os.path.isfile(log):
        return 0.0
    import re
    vals = re.findall(r"noise_multiplier=([0-9.eE+-]+)", open(log).read())
    return float(vals[-1]) if vals else 0.0


def run_cell(slug, eps_tag):
    folder = f"{ROOT}/{slug}_composite_population{EPS_FOLDER[eps_tag]}"
    if not os.path.isdir(os.path.join(folder, "checkpoint")):
        return None
    nm = noise_multiplier(folder)
    md = f"{BASE}/{slug}_metadata.json"
    tr, te = f"{BASE}/{slug}_train.csv", f"{BASE}/{slug}_test.csv"

    priv, info = load_private(tr, md)
    embed = make_embed_fn(priv, info)
    from pe.constant.data import LABEL_ID_COLUMN_NAME as L
    npc = {int(k): int(v) for k, v in priv.data_frame[L].value_counts().items()}

    rows, classes, labels, _ = build_audit_set(tr, te, md, ref_holdout_frac=0.5)
    ref_rows = reference_rows_by_class(te, md, ref_holdout_frac=0.5)
    ref_by = {c: embed(r) for c, r in ref_rows.items()}
    q = embed(rows)
    by_class = reconstruct(f"{folder}/checkpoint", embed, ref_by, npc,
                           use_clean=False)
    sigma = nm * 1.0

    cell = {"noise_multiplier": nm}
    for name, cfg in CONFIGS.items():
        s = score_config(by_class, q, classes, ref_by, cfg, sigma)
        m = evaluate(labels, s)
        cell[name] = {"auc": m["auc"], "tpr@1": m["tpr@fpr=0.01"]}
    return cell


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*", default=DATASETS)
    p.add_argument("--eps", nargs="*", default=["inf", "10", "1"],
                   help="eps tags, or 'all'")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()
    eps_list = list(EPS_FOLDER) if args.eps == ["all"] else args.eps

    results = {}
    if os.path.isfile(args.out):
        results = json.load(open(args.out)).get("cells", {})

    names = list(CONFIGS)
    print(f"{'dataset':22} {'eps':4} | " + " ".join(f"{n[:9]:>9}" for n in names))
    print("-" * (29 + 10 * len(names)))
    for slug in args.datasets:
        for eps in eps_list:
            cell = run_cell(slug, eps)
            if cell is None:
                continue
            results[f"{slug}@{eps}"] = cell
            base = cell["C0_baseline"]["auc"]
            row = []
            for n in names:
                a = cell[n]["auc"]
                row.append(f"{a:>9.4f}" if n == "C0_baseline" else f"{a-base:>+9.4f}")
            print(f"{slug:22} {eps:4} | " + " ".join(row))
            with open(args.out, "w") as f:
                json.dump({"configs": {k: vars(v) for k, v in CONFIGS.items()},
                           "cells": results}, f, indent=2, default=str)
    print(f"\nTable shows C0 AUC then each config's ΔAUC vs C0. Wrote {args.out}")


if __name__ == "__main__":
    main()
