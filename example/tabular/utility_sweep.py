"""Collect downstream UTILITY metrics across every tabular-PE run and plot them
against epsilon, plus a privacy-utility tradeoff against the MIA AUC.

Each run already logs, per iteration, a TabICL classifier trained on the synthetic
data and evaluated on the real test split (train-on-synthetic / test-on-real), as
well as FID and 1-/2-way TVD. We take the FINAL iteration's value (filtered to the
released selected set, PE.VARIATION_API_FOLD_ID = -1) as the utility of the
released synthetic data, and tabulate it by dataset x privacy budget.

Outputs:
  results/tabular/_utility_logs/utility_summary.json
  results/tabular/_utility_logs/plots/utility_vs_eps_<dataset>.png  (+ _all.png)
  results/tabular/_utility_logs/plots/privacy_utility_<dataset>.png (+ _all.png)
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pe.data import TabularCSV
from pe.constant.data import LABEL_ID_COLUMN_NAME

ROOT = "results/tabular"
OUTDIR = f"{ROOT}/_utility_logs"
PLOTS = f"{OUTDIR}/plots"
MIA_SUMMARY = f"{ROOT}/_mia_logs/mia_summary.json"
BASE = ("https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/"
        "heads/main/tabular")

BUDGET_FROM_SUFFIX = {"": "eps10", "_nonoise": "inf"}


def baselines(datasets):
    """Trivial-classifier baselines from the real test labels, per dataset:
    majority-class accuracy (predict the most frequent class) and uniform random
    guessing (1 / num_classes). These are the floors a useful synthetic set beats.
    """
    out = {}
    for slug in datasets:
        te = TabularCSV(csv_path=f"{BASE}/{slug}_test.csv",
                        metadata_path=f"{BASE}/{slug}_metadata.json")
        vc = te.data_frame[LABEL_ID_COLUMN_NAME].value_counts()
        n, k = int(vc.sum()), int(len(vc))
        out[slug] = {"majority_acc": 100.0 * int(vc.max()) / n,
                     "uniform_acc": 100.0 / k, "num_classes": k}
    return out


def budget_of(run):
    suffix = run.split("_composite_population", 1)[1]
    return BUDGET_FROM_SUFFIX.get(suffix, suffix.lstrip("_") or "eps10")


def eps_of(budget):
    if budget in ("inf", "nonoise"):
        return float("inf")
    return float(budget.replace("eps", "").replace("p", "."))


def _last_value(folder, pattern):
    """Final-iteration value from a '<iter>,<value>' CSV matched by glob pattern."""
    hits = glob.glob(os.path.join(folder, pattern))
    if not hits:
        return None
    df = pd.read_csv(hits[0], header=None, names=["it", "val"])
    if df.empty:
        return None
    return float(df.sort_values("it")["val"].iloc[-1])


def collect():
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/*_composite_population*")):
        if not os.path.isdir(os.path.join(d, "checkpoint")):
            continue
        run = os.path.basename(d)
        slug = run.split("_composite_population", 1)[0]
        budget = budget_of(run)
        # Filtered to the released selected set (fold -1).
        rows.append({
            "dataset": slug,
            "budget": budget,
            "epsilon": eps_of(budget),
            "clf_acc": _last_value(d, "tabular_classifier_tabicl_filter_*_test_acc.csv"),
            "clf_auc": _last_value(d, "tabular_classifier_tabicl_filter_*_test_auc.csv"),
            "clf_f1":  _last_value(d, "tabular_classifier_tabicl_filter_*_test_f1.csv"),
            "fid":     _last_value(d, "fid_*_{*}.csv"),
            "tvd_1way": _last_value(d, "1way-tvd_*_{*}.csv"),
            "tvd_2way": _last_value(d, "2way-tvd_*_{*}.csv"),
        })
    rows.sort(key=lambda r: (r["dataset"], -r["epsilon"]))
    return rows


def _xticks(ax, finite, inf_x):
    ax.set_xscale("log")
    ax.set_xticks(finite + [inf_x])
    ax.set_xticklabels([f"{e:g}" for e in finite] + [r"$\infty$"])
    ax.set_xlabel(r"privacy budget $\varepsilon$  (right = no privacy)")
    ax.grid(True, which="both", alpha=0.3)


def plot_utility(rows, datasets, finite, inf_x, base):
    def series(ds):
        rs = sorted((r for r in rows if r["dataset"] == ds), key=lambda r: r["epsilon"])
        xs = [inf_x if r["epsilon"] == float("inf") else r["epsilon"] for r in rs]
        return xs, [r["clf_acc"] for r in rs], [r["clf_auc"] for r in rs]

    def draw(ax, ds):
        xs, acc, auc = series(ds)
        ax.plot(xs, acc, "o-", color="C2", label="test accuracy")
        ax.plot(xs, auc, "s--", color="C3", label="test AUC")
        b = base[ds]
        ax.axhline(b["majority_acc"], color="dimgray", ls="--", lw=1.2,
                   label=f"majority-class baseline ({b['majority_acc']:.1f}%)")
        ax.axhline(b["uniform_acc"], color="silver", ls=":", lw=1.2,
                   label=f"random 1/{b['num_classes']} ({b['uniform_acc']:.1f}%)")
        # Keep the y-range wide enough that both random baselines stay visible.
        vals = [v for v in acc + auc if v is not None] + [b["majority_acc"], b["uniform_acc"]]
        ax.set_ylim(min(vals) - 4, max(vals) + 4)
        _xticks(ax, finite, inf_x)
        ax.set_title(ds)
        ax.legend(loc="lower right", fontsize=7)

    for ds in datasets:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        draw(ax, ds)
        ax.set_ylabel("downstream TabICL utility (%)  (train-synth / test-real)")
        fig.tight_layout()
        fig.savefig(f"{PLOTS}/utility_vs_eps_{ds}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[i // 2][i % 2]
        draw(ax, ds)
        ax.set_ylabel("TabICL utility (%)")
    for j in range(len(datasets), 4):
        axes[j // 2][j % 2].axis("off")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/utility_vs_eps_all.png", dpi=150)
    plt.close(fig)


def plot_tradeoff(rows, datasets):
    """Privacy-utility: MIA AUC (x) vs classifier accuracy (y), one point per eps."""
    if not os.path.exists(MIA_SUMMARY):
        print("  (no mia_summary.json -- skipping privacy-utility tradeoff)")
        return
    mia = {(r["dataset"], r["budget"]): r["auc_ref"]
           for r in json.load(open(MIA_SUMMARY))["rows"]}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[i // 2][i % 2]
        rs = sorted((r for r in rows if r["dataset"] == ds), key=lambda r: r["epsilon"])
        xs = [mia.get((ds, r["budget"])) for r in rs]
        ys = [r["clf_acc"] for r in rs]
        labels = [("∞" if r["epsilon"] == float("inf") else f"{r['epsilon']:g}") for r in rs]
        ax.plot(xs, ys, "o-", color="C0")
        for x, y, lb in zip(xs, ys, labels):
            if x is not None and y is not None:
                ax.annotate(lb, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axvline(0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("MIA ROC AUC (privacy leak; 0.5 = none)")
        ax.set_ylabel("TabICL test accuracy (%)  (utility)")
        ax.set_title(ds); ax.grid(True, alpha=0.3)
    for j in range(len(datasets), 4):
        axes[j // 2][j % 2].axis("off")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/privacy_utility_all.png", dpi=150)
    plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    rows = collect()
    datasets = sorted({r["dataset"] for r in rows})
    finite = sorted({r["epsilon"] for r in rows if r["epsilon"] != float("inf")})
    inf_x = finite[-1] * 4.0

    print("Computing trivial-classifier baselines from real test labels...")
    base = baselines(datasets)

    # Console table.
    hdr = (f"{'dataset':22} {'budget':8} | {'clf_acc':>8} {'clf_auc':>8} "
           f"{'clf_f1':>8} {'fid':>7} {'tvd1':>6} {'tvd2':>6}")
    print(hdr); print("-" * len(hdr))
    def g(x, p=2): return "   n/a" if x is None else f"{x:.{p}f}"
    for r in rows:
        print(f"{r['dataset']:22} {r['budget']:8} | {g(r['clf_acc']):>8} {g(r['clf_auc']):>8} "
              f"{g(r['clf_f1']):>8} {g(r['fid'],3):>7} {g(r['tvd_1way'],3):>6} {g(r['tvd_2way'],3):>6}")

    print("\nrandom baselines (real test set):")
    print(f"{'dataset':22} {'classes':>7} {'majority_acc':>13} {'uniform_acc':>12}")
    for ds in datasets:
        b = base[ds]
        print(f"{ds:22} {b['num_classes']:>7} {b['majority_acc']:>12.2f}% {b['uniform_acc']:>11.2f}%")

    with open(f"{OUTDIR}/utility_summary.json", "w") as f:
        json.dump({"metric": "final-iteration, released set (fold -1)",
                   "baselines": base, "rows": rows}, f, indent=2, default=str)
    print(f"\nWrote {OUTDIR}/utility_summary.json")

    plot_utility(rows, datasets, finite, inf_x, base)
    plot_tradeoff(rows, datasets)
    print(f"Wrote plots under {PLOTS}/")


if __name__ == "__main__":
    main()
