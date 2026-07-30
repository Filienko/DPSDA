"""Plot the Phase-2 ablation: per-dataset ΔAUC of each attack config vs the C0
baseline, at each epsilon. Reads results/tabular/_mia_logs/ablation_summary.json,
writes plots into results/tabular/_mia_logs/plots/ablation_*.png.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUMMARY = "results/tabular/_mia_logs/ablation_summary.json"
PLOTS = "results/tabular/_mia_logs/plots"

# Single-component configs to show (skip the leave-one-out variants for clarity).
SHOW = ["pool3", "w_snr", "lira", "density", "selection", "ALL"]


def eps_val(e):
    return float("inf") if e == "inf" else float(e)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    data = json.load(open(SUMMARY))
    cells = data["cells"]

    keys = sorted(cells, key=lambda k: (k.split("@")[0], -eps_val(k.split("@")[1])))
    datasets = sorted({k.split("@")[0] for k in keys})
    eps_tags = sorted({k.split("@")[1] for k in keys}, key=lambda e: -eps_val(e))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), squeeze=False)
    x = np.arange(len(eps_tags))
    width = 0.8 / len(SHOW)
    for i, ds in enumerate(datasets):
        ax = axes[i // 2][i % 2]
        for j, cfg in enumerate(SHOW):
            deltas = []
            for e in eps_tags:
                cell = cells.get(f"{ds}@{e}")
                if cell and cfg in cell and "C0_baseline" in cell:
                    deltas.append(cell[cfg]["auc"] - cell["C0_baseline"]["auc"])
                else:
                    deltas.append(np.nan)
            ax.bar(x + j * width, deltas, width, label=cfg)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels([f"ε={e}" for e in eps_tags])
        base = [cells.get(f"{ds}@{e}", {}).get("C0_baseline", {}).get("auc") for e in eps_tags]
        base_s = ", ".join(f"{e}:{b:.3f}" for e, b in zip(eps_tags, base) if b is not None)
        ax.set_title(f"{ds}\nC0 AUC  {base_s}", fontsize=9)
        ax.set_ylabel("ΔAUC vs baseline")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
    for k in range(len(datasets), 4):
        axes[k // 2][k % 2].axis("off")
    fig.suptitle("Phase-2 ablation: AUC gain of each attack component over the "
                 "count-LLR baseline (>0 = helps)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/ablation_delta_auc.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {PLOTS}/ablation_delta_auc.png")


if __name__ == "__main__":
    main()
