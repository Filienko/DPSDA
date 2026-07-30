"""Figures for the natural distribution-shift study.

Reads the sweep outputs and renders (single-axis panels only, Okabe-Ito
colorblind-safe palette):

  fig_marginals.png  -- PCA-1 score density, members vs aux, per variant. Visual
                        answer to "how close are the private and aux distributions".
  fig_votehist.png   -- released vote occupancy (private) vs aux-induced occupancy
                        (the null), iid vs q18. "How close is the vote histogram."
  fig_auc.png        -- AUC vs member<->aux gap: real MIA (with/without D_ref) and
                        a model-blind distribution-only detector, all one axis.
  fig_utility.png    -- utility on aux and fidelity-to-private vs the gap.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attacks.reconstruct import reconstruct
from attacks.histogram_mia import cell_occupancy
from attacks.audit_set import reference_rows_by_class
from attacks.tabular_embedding import load_private, make_embed_fn
from pe.constant.data import LABEL_ID_COLUMN_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NAT = os.path.join(DATA, "natural")
META = os.path.join(DATA, "metadata.json")
ROOT = os.path.join(HERE, "..", "..", "..", "results", "tabular", "dist_shift")
OUT = os.path.join(ROOT, "analysis")
TAGS = ["iid", "q50", "q35", "q25", "q18"]
LABELS = {"iid": "iid (matched)", "q50": "q50", "q35": "q35", "q25": "q25", "q18": "q18"}

C = {"member": "#0072B2", "aux": "#E69F00", "mia_ref": "#0072B2",
     "mia_noref": "#009E73", "distonly": "#E69F00", "util": "#D55E00",
     "fid": "#CC79A7"}
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def pca1_axis():
    meta = json.load(open(META))
    fc = meta["int_columns"] + meta["float_columns"]
    pool = pd.concat([pd.read_csv(os.path.join(DATA, "train.csv")),
                      pd.read_csv(os.path.join(DATA, "test.csv"))], ignore_index=True)
    X = pool[fc].to_numpy(float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    return fc, mu, sd, Vt[0]


def proj(df, fc, mu, sd, v1):
    return ((df[fc].to_numpy(float) - mu) / sd) @ v1


def fig_marginals(axis):
    fc, mu, sd, v1 = axis
    fig, axes = plt.subplots(1, len(TAGS), figsize=(3.0 * len(TAGS), 3.0),
                             sharex=True, sharey=True)
    for ax, tag in zip(axes, TAGS):
        m = pd.read_csv(os.path.join(NAT, f"members_{tag}.csv"))
        a = pd.read_csv(os.path.join(NAT, f"aux_{tag}.csv"))
        sm, sa = proj(m, fc, mu, sd, v1), proj(a, fc, mu, sd, v1)
        bins = np.linspace(-5, 6, 40)
        ax.hist(sm, bins=bins, density=True, color=C["member"], alpha=0.55,
                label="private (members)")
        ax.hist(sa, bins=bins, density=True, color=C["aux"], alpha=0.55,
                label="aux (non-members + D_ref)")
        ax.set_title(LABELS[tag]); ax.set_xlabel("PCA-1 score")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Natural member/aux separation along the data's principal axis")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_marginals.png"), dpi=150)
    plt.close(fig)


def occupancy(tag):
    """Released private occupancy p and aux-induced occupancy q for the last round."""
    priv, info = load_private(os.path.join(NAT, f"members_{tag}.csv"), META)
    embed_fn = make_embed_fn(priv, info)
    n_priv = {int(k): int(v) for k, v in
              priv.data_frame[LABEL_ID_COLUMN_NAME].value_counts().items()}
    ref_rows = reference_rows_by_class(os.path.join(NAT, f"aux_{tag}.csv"), META,
                                       max_per_class=2000, seed=0, ref_holdout_frac=0.5)
    ref_by_class = {c: embed_fn(r) for c, r in ref_rows.items()}
    by_class = reconstruct(os.path.join(ROOT, f"natural_{tag}", "checkpoint"),
                           embed_fn, ref_by_class, n_priv, start_t=1, use_clean=False)
    ps, qs = [], []
    for cls, iters in by_class.items():
        it = iters[-1]
        c = np.clip(np.asarray(it["counts"], float), 0, None)
        if c.sum() <= 0:
            continue
        q = cell_occupancy(it["reference_features"], it["cell_features"], mode="L2")
        ps.append(c / c.sum()); qs.append(q / q.sum())
    return np.concatenate(ps), np.concatenate(qs)


def fig_votehist():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, tag in zip(axes, ["iid", "q18"]):
        p, q = occupancy(tag)
        order = np.argsort(-p)
        x = np.arange(len(p))
        ax.plot(x, p[order], color=C["member"], lw=1.6, label="private (released votes)")
        ax.plot(x, q[order], color=C["aux"], lw=1.6, label="aux (assumed null)")
        ax.set_title(f"vote occupancy — {LABELS[tag]}")
        ax.set_xlabel("synthetic cell (sorted by private occupancy)")
        ax.set_ylabel("occupancy fraction")
    axes[0].legend(fontsize=8)
    fig.suptitle("Released vote histogram vs the aux-induced null the attack assumes")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_votehist.png"), dpi=150)
    plt.close(fig)


def _report_auc(tag, key):
    r = json.load(open(os.path.join(ROOT, f"natural_{tag}",
                                    f"mia_report_{key}.json")))
    return r["regimes"]["noised"]["all_rounds"]["auc"]


def fig_auc_utility():
    summ = {r["tag"]: r for r in json.load(open(os.path.join(OUT, "summary.json")))["rows"]}
    gap = [summ[t]["mean_embedding_gap"] for t in TAGS]
    mia_ref = [_report_auc(t, "ref") for t in TAGS]
    mia_noref = [_report_auc(t, "noref") for t in TAGS]
    distonly = [summ[t]["auc_distribution_only"] for t in TAGS]
    util = [summ[t]["utility_acc_on_aux"] for t in TAGS]
    vmmd = [summ[t]["vote_hist_mmd_priv_vs_aux"] for t in TAGS]

    # AUC panel (single axis).
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(gap, mia_ref, "o-", color=C["mia_ref"], lw=2, label="MIA (histogram, with D_ref)")
    ax.plot(gap, mia_noref, "s-", color=C["mia_noref"], lw=2, label="MIA (histogram, no D_ref)")
    ax.plot(gap, distonly, "^--", color=C["distonly"], lw=2,
            label="distribution-only detector (no model access)")
    ax.axhline(mia_ref[0], color="gray", ls=":", lw=1.2)
    ax.annotate("genuine leak at matched dist.", (gap[0], mia_ref[0]),
                xytext=(0.14, 0.555), fontsize=8, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    ax.axhline(0.5, color="silver", ls="-", lw=1, alpha=0.7)
    # variant tags along the clear bottom strip
    for x, t in zip(gap, TAGS):
        ax.annotate(LABELS[t].split()[0], (x, 0.47), fontsize=7, color="gray",
                    ha="center", va="center")
    ax.set_xlabel("member ↔ aux distance (mean embedding gap)")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.44, 1.02)
    ax.set_xlim(-0.03, 0.82)
    ax.set_title("At fixed ε=10, apparent MIA leakage is driven by the\n"
                 "distribution gap — and a model-blind detector matches it")
    ax.legend(fontsize=8, loc="center right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_auc.png"), dpi=150)
    plt.close(fig)

    # Utility + fidelity + vote-cosine (single axis each).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(gap, util, "o-", color=C["util"], lw=2)
    axes[0].set_xlabel("member ↔ aux distance (mean embedding gap)")
    axes[0].set_ylabel("TabICL accuracy on aux (%)")
    axes[0].set_title("Utility on the aux distribution collapses with the gap")
    axes[0].axhline(10.0, color="silver", ls=":", lw=1.2)
    axes[0].annotate("10-class random floor", (gap[-2], 10.5), fontsize=8, color="gray")

    axes[1].plot(gap, vmmd, "D-", color=C["fid"], lw=2)
    axes[1].set_xlabel("member ↔ aux distance (mean embedding gap)")
    axes[1].set_ylabel("kernel MMD(private vote hist, aux null)")
    axes[1].set_title("Vote histogram and aux null diverge with the gap")
    axes[1].set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_utility.png"), dpi=150)
    plt.close(fig)


def fig_mmd_vs_auc():
    """Private<->aux distribution MMD vs MIA AUC, with the generator, members, and
    non-member challenges all FIXED and only the aux / D_ref sample varying (drawn
    from increasingly shifted PCA-1 windows of the real pool). Isolates the effect
    of the private<->aux gap -- the assumption Aug-PE breaks -- on the attack."""
    aux = json.load(open(os.path.join(OUT, "aux_shift_test.json")))["aux"]
    keys = list(aux.keys())
    mmd = [aux[k]["priv_aux_mmd"] for k in keys]
    auc = [aux[k]["auc"] for k in keys]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(mmd, auc, "o-", color=C["mia_ref"], lw=2)
    for x, y, k in zip(mmd, auc, keys):
        lab = "matched" if k == "matched" else k.replace("shift", "")
        ax.annotate(lab, (x, y), fontsize=7, color=C["mia_ref"],
                    xytext=(4, 5), textcoords="offset points")
    ax.axhline(auc[0], color="gray", ls=":", lw=1.2)
    ax.annotate("genuine leak (matched aux)", (mmd[0], auc[0]),
                xytext=(mmd[1], auc[0] + 0.012), fontsize=8, color="gray")
    ax.axhline(0.5, color="silver", lw=1, alpha=0.7)
    ax.set_xlabel("private ↔ aux distribution distance  (two-sample RBF-MMD on records)")
    ax.set_ylabel("MIA ROC AUC")
    ax.set_ylim(0.49, 0.63)
    ax.set_title("A private↔aux (reference) gap does NOT inflate the attack —\n"
                 "it mildly degrades it (members & challenges fixed; only aux moves)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_mmd_vs_auc.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    axis = pca1_axis()
    fig_marginals(axis)
    fig_votehist()
    fig_auc_utility()
    fig_mmd_vs_auc()
    print("wrote fig_marginals.png fig_votehist.png fig_auc.png fig_utility.png "
          f"fig_mmd_vs_auc.png under {os.path.relpath(OUT, HERE)}/")


if __name__ == "__main__":
    main()
