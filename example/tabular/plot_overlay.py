"""Overlay privacy and utility on one twin-axis plot per dataset.

x  = privacy budget epsilon (log; inf = no privacy at the right)
left  y = MIA ROC AUC (with D_ref, noised DP histogram)   -- privacy leak
right y = TabICL downstream test accuracy (train-synth/test-real) -- utility

Reads results/tabular/_mia_logs/mia_summary.json and
results/tabular/_utility_logs/utility_summary.json. Writes
results/tabular/_utility_logs/plots/overlay_<dataset>.png (+ _all.png).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "results/tabular"
MIA = f"{ROOT}/_mia_logs/mia_summary.json"
UTIL = f"{ROOT}/_utility_logs/utility_summary.json"
PLOTS = f"{ROOT}/_utility_logs/plots"


def eps_of(b):
    return float("inf") if b in ("inf", "nonoise") else float(b.replace("eps", "").replace("p", "."))


def main():
    os.makedirs(PLOTS, exist_ok=True)
    mia = json.load(open(MIA))["rows"]
    util = json.load(open(UTIL))
    base = util["baselines"]
    auc = {(r["dataset"], r["budget"]): r["auc_ref"] for r in mia}
    acc = {(r["dataset"], r["budget"]): r["clf_acc"] for r in util["rows"]}

    datasets = sorted({r["dataset"] for r in mia})
    budgets = sorted({r["budget"] for r in mia}, key=eps_of)  # strong->weak
    finite = [eps_of(b) for b in budgets if eps_of(b) != float("inf")]
    inf_x = max(finite) * 4.0

    def xpos(b):
        e = eps_of(b)
        return inf_x if e == float("inf") else e

    def draw(ax, ds):
        bs = sorted((b for b in budgets if (ds, b) in auc), key=eps_of)
        xs = [xpos(b) for b in bs]
        a_priv = [auc[(ds, b)] for b in bs]
        a_util = [acc[(ds, b)] for b in bs]

        ax.set_xscale("log")
        l1, = ax.plot(xs, a_priv, "o-", color="C0", label="MIA AUC (privacy leak)")
        ax.axhline(0.5, color="C0", ls=":", lw=1, alpha=0.6)
        ax.set_ylabel("MIA ROC AUC  (0.5 = no leak)", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax.set_ylim(0.45, 1.0)
        ax.set_xticks(finite + [inf_x])
        ax.set_xticklabels([f"{e:g}" for e in finite] + [r"$\infty$"])
        ax.set_xlabel(r"privacy budget $\varepsilon$  (right = no privacy)")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(ds)

        ax2 = ax.twinx()
        l2, = ax2.plot(xs, a_util, "s--", color="C2", label="TabICL accuracy (utility)")
        l3 = ax2.axhline(base[ds]["majority_acc"], color="dimgray", ls="--", lw=1.1,
                         label=f"majority baseline ({base[ds]['majority_acc']:.1f}%)")
        ax2.set_ylabel("TabICL test accuracy (%)", color="C2")
        ax2.tick_params(axis="y", labelcolor="C2")
        lo = min(a_util + [base[ds]["majority_acc"]]) - 4
        hi = max(a_util + [base[ds]["majority_acc"]]) + 4
        ax2.set_ylim(lo, hi)
        return [l1, l2, l3]

    # One figure per dataset.
    for ds in datasets:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        handles = draw(ax, ds)
        ax.legend(handles=handles, loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{PLOTS}/overlay_{ds}.png", dpi=150)
        plt.close(fig)

    # Combined 2x2.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[i // 2][i % 2]
        handles = draw(ax, ds)
        ax.legend(handles=handles, loc="upper left", fontsize=7)
    for j in range(len(datasets), 4):
        axes[j // 2][j % 2].axis("off")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/overlay_all.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {PLOTS}/overlay_*.png")


if __name__ == "__main__":
    main()
