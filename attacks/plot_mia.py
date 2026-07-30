"""Plot MIA AUC vs epsilon, one graph per dataset.

Reads results/tabular/_mia_logs/mia_summary.json and plots, for each dataset, the
membership-inference ROC AUC WITH the auxiliary reference set D_ref on the NOISED
(released DP-histogram) regime --- i.e. the `auc_ref` field --- against the privacy
budget epsilon. The no-privacy run (epsilon = inf) is drawn at the right of the
log x-axis under an "inf" tick. Writes one PNG per dataset plus a combined figure
into results/tabular/_mia_logs/plots/.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMMARY = "results/tabular/_mia_logs/mia_summary.json"
OUTDIR = "results/tabular/_mia_logs/plots"


def eps_of(budget):
    if budget in ("inf", "nonoise"):
        return float("inf")
    return float(budget.replace("eps", "").replace("p", "."))


def main():
    with open(SUMMARY) as f:
        rows = json.load(f)["rows"]

    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)

    os.makedirs(OUTDIR, exist_ok=True)
    datasets = sorted(by_ds)

    # Place inf at the right of the finite epsilons on the log axis.
    finite = sorted({eps_of(r["budget"]) for r in rows if eps_of(r["budget"]) != float("inf")})
    inf_x = finite[-1] * 4.0  # position for the "no privacy" point

    def series(recs):
        pts = sorted(((eps_of(r["budget"]), r["auc_ref"]) for r in recs), key=lambda t: t[0])
        xs = [inf_x if e == float("inf") else e for e, _ in pts]
        ys = [a for _, a in pts]
        return xs, ys

    def style_axis(ax, title):
        ax.set_xscale("log")
        ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance (0.5)")
        ax.axvline(inf_x / 2.0, color="lightgray", ls=":", lw=1)
        ticks = finite + [inf_x]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{e:g}" for e in finite] + [r"$\infty$"])
        ax.set_xlabel(r"privacy budget $\varepsilon$  (right = no privacy)")
        ax.set_ylabel("MIA ROC AUC (with $D_{ref}$, noised DP histogram)")
        ax.set_ylim(0.45, 1.0)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    # One figure per dataset.
    for ds in datasets:
        xs, ys = series(by_ds[ds])
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(xs, ys, "o-", color="C0", lw=2, ms=6, label="AUC")
        style_axis(ax, ds)
        fig.tight_layout()
        out = os.path.join(OUTDIR, f"auc_vs_eps_{ds}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")

    # Combined 2x2 grid.
    n = len(datasets)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.5 * nrow), squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[i // ncol][i % ncol]
        xs, ys = series(by_ds[ds])
        ax.plot(xs, ys, "o-", color="C0", lw=2, ms=6, label="AUC")
        style_axis(ax, ds)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    out = os.path.join(OUTDIR, "auc_vs_eps_all.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
