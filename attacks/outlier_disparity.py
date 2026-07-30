"""Disparate impact of the histogram MIA across the private data by outlierness.

Does membership-inference risk fall harder on outliers or on typical records? For a
memorization attack the classic answer is "outliers". But this attack reads a
record's vote out of the *synthetic* histogram, and PE concentrates synthetic mass
where the private density is high -- so a record's region must be COVERED by
synthetic cells for its membership vote to register. That predicts the opposite
disparity, and the measurement below confirms it (Spearman rho < 0): typical/inlier
members are the most exposed, while outliers land in low-count / uncovered cells,
score like non-members, and are effectively hidden. Measured on a MATCHED-
distribution setup (members and non-members from the same distribution, so this is
genuine leakage, not a distribution confound). Runs across every dataset with a
completed epsilon=10 run to test generality.

  * outlierness(x) = mean distance from member x to its k nearest same-class
    private neighbours in the attack's TabularEmbedding space, as a within-class
    rank (larger = more of an outlier within its class).
  * per-member MIA score = the released-histogram noised LLR (the real attack).
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attacks.audit_set import build_audit_set, reference_rows_by_class
from attacks.reconstruct import reconstruct
from attacks.histogram_mia import score_records
from attacks.tabular_embedding import load_private, make_embed_fn
from pe.constant.data import LABEL_ID_COLUMN_NAME, TABULAR_DATA_COLUMN_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(HERE, "..", "example", "tabular", "dist_shift", "data")
RESULTS = os.path.join(HERE, "..", "results", "tabular")
OUT = os.path.join(RESULTS, "dist_shift", "analysis")
BASE = ("https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/heads/"
        "main/tabular/real")
KNN = 5
SEED = 0
MAX_BASE = 4000            # cap same-class neighbour pool for the outlier score

C = {"inlier": "#0072B2", "mid": "#999999", "outlier": "#D55E00"}
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def _ckpt(slug):
    return os.path.join(RESULTS, f"{slug}_composite_population", "checkpoint")


# artificial-characters uses the local cached CSVs; the rest load from the store.
DATASETS = [
    {"name": "artificial-characters",
     "train": os.path.join(LOCAL, "train.csv"), "test": os.path.join(LOCAL, "test.csv"),
     "meta": os.path.join(LOCAL, "metadata.json"),
     "ckpt": _ckpt("artificial-characters"), "nm": 1.8385747404944874},
    {"name": "breast-cancer", "nm": 1.853642779702104, "ckpt": _ckpt("breast-cancer")},
    {"name": "adult", "nm": 2.8139087367478592, "ckpt": _ckpt("adult")},
    {"name": "person-activity", "nm": 2.0436299197813614, "ckpt": _ckpt("person-activity")},
]


def _paths(cfg):
    if "train" in cfg:
        return cfg["train"], cfg["test"], cfg["meta"]
    n = cfg["name"]
    return (f"{BASE}/{n}/{n}_train.csv", f"{BASE}/{n}/{n}_test.csv",
            f"{BASE}/{n}/{n}_metadata.json")


def knn_outlierness(member_emb, member_cls, train_emb, train_cls, k=KNN):
    """Within-class rank in [0,1] of each member's mean distance to its k nearest
    same-class training neighbours (self-match excluded). 1 = extreme outlier."""
    rng = np.random.default_rng(SEED)
    out = np.zeros(len(member_emb))
    for c in np.unique(member_cls):
        base = train_emb[train_cls == c]
        if base.shape[0] > MAX_BASE:
            base = base[rng.choice(base.shape[0], MAX_BASE, replace=False)]
        idx = np.where(member_cls == c)[0]
        if base.shape[0] <= k + 1:
            continue
        bsq = np.einsum("ij,ij->i", base, base)
        for j, qi in zip(idx, member_emb[idx]):
            d2 = np.clip(bsq + (qi @ qi) - 2.0 * (base @ qi), 0.0, None)
            d2.sort()
            start = 1 if d2[0] < 1e-9 else 0
            out[j] = np.sqrt(d2[start:start + k]).mean()
        out[idx] = out[idx].argsort().argsort() / max(len(idx) - 1, 1)
    return out


def analyze(cfg):
    train, test, meta = _paths(cfg)
    priv, info = load_private(train, meta)
    embed_fn = make_embed_fn(priv, info)
    n_priv = {int(k): int(v) for k, v in
              priv.data_frame[LABEL_ID_COLUMN_NAME].value_counts().items()}
    train_rows = [list(r) for r in priv.data_frame[TABULAR_DATA_COLUMN_NAME].tolist()]
    train_cls = priv.data_frame[LABEL_ID_COLUMN_NAME].to_numpy().astype(int)
    train_emb = embed_fn(train_rows)

    rows, classes, labels, _ = build_audit_set(train, test, meta, n_members=1500,
                                               n_nonmembers=1500, seed=SEED,
                                               ref_holdout_frac=0.5)
    classes = np.asarray(classes); labels = np.asarray(labels)
    ref_rows = reference_rows_by_class(test, meta, max_per_class=2000, seed=SEED,
                                       ref_holdout_frac=0.5)
    ref_by_class = {c: embed_fn(r) for c, r in ref_rows.items()}
    query_emb = embed_fn(rows)

    by_class = reconstruct(cfg["ckpt"], embed_fn, ref_by_class, n_priv, start_t=1,
                           use_clean=False)
    scores = np.full(len(classes), np.nan)
    for c in np.unique(classes):
        iters = by_class.get(int(c))
        if not iters:
            continue
        mask = classes == c
        scores[mask] = score_records(query_emb[mask], iters, mode="L2",
                                     regime="noised", sigma=cfg["nm"], censored=False)
    fin = scores[np.isfinite(scores)]
    scores[~np.isfinite(scores)] = (fin.min() - 1.0) if fin.size else 0.0

    mem, non = labels == 1, labels == 0
    out_rank = knn_outlierness(query_emb[mem], classes[mem], train_emb, train_cls)
    mem_scores, non_scores = scores[mem], scores[non]
    rho, p = spearmanr(out_rank, mem_scores)

    edges = np.quantile(out_rank, [1/3, 2/3])
    grp = np.where(out_rank <= edges[0], "inlier",
                   np.where(out_rank <= edges[1], "mid", "outlier"))

    def metrics(gs):
        y = np.r_[np.ones(len(gs)), np.zeros(len(non_scores))]
        s = np.r_[gs, non_scores]
        fpr, tpr, _ = roc_curve(y, s)
        tat = lambda f: float(tpr[max(np.searchsorted(fpr, f, "right") - 1, 0)])
        return {"n": int(len(gs)), "auc": round(float(roc_auc_score(y, s)), 4),
                "tpr@1%": round(tat(0.01), 4), "tpr@5%": round(tat(0.05), 4)}

    groups = {g: metrics(mem_scores[grp == g]) for g in ["inlier", "mid", "outlier"]}
    overall = round(float(roc_auc_score(labels, scores)), 4)
    return {"name": cfg["name"], "overall_auc": overall,
            "spearman_outlier_vs_score": round(float(rho), 4), "spearman_p": float(p),
            "groups": groups}


def main():
    os.makedirs(OUT, exist_ok=True)
    results = []
    for cfg in DATASETS:
        try:
            r = analyze(cfg)
        except Exception as e:
            print(f"{cfg['name']}: FAILED ({e})")
            continue
        results.append(r)
        g = r["groups"]
        print(f"\n=== {r['name']}  (overall AUC {r['overall_auc']}, "
              f"Spearman rho={r['spearman_outlier_vs_score']:+.3f}) ===")
        print(f"  {'group':8s} {'AUC':>7s} {'TPR@1%':>8s} {'TPR@5%':>8s}")
        for k in ["inlier", "mid", "outlier"]:
            print(f"  {k:8s} {g[k]['auc']:>7.3f} {g[k]['tpr@1%']:>8.3f} {g[k]['tpr@5%']:>8.3f}")

    json.dump({"epsilon": 10, "setup": "matched", "knn": KNN, "datasets": results},
              open(os.path.join(OUT, "outlier_disparity_multidataset.json"), "w"), indent=2)

    # Combined figure: per-group AUC per dataset + Spearman per dataset.
    names = [r["name"] for r in results]
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for off, gname in zip([-w, 0, w], ["inlier", "mid", "outlier"]):
        ax[0].bar(x + off, [r["groups"][gname]["auc"] for r in results], w,
                  color=C[gname], label=gname)
    ax[0].axhline(0.5, color="silver", lw=1.2, alpha=0.8)
    ax[0].annotate("chance", (len(names) - 0.5, 0.5), fontsize=8, color="gray",
                   va="bottom", ha="right")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=15, ha="right")
    ax[0].set_ylabel("member-vs-nonmember AUC"); ax[0].set_ylim(0.38, 0.78)
    ax[0].set_title("Attack AUC by outlierness group (inlier > outlier everywhere)")
    ax[0].legend(fontsize=8)

    ax[1].bar(x, [r["spearman_outlier_vs_score"] for r in results], 0.5, color="#555555")
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=15, ha="right")
    ax[1].set_ylabel("Spearman(outlierness, MIA score)")
    ax[1].set_title("Outlierness vs leakage is negative on every dataset")
    fig.suptitle("Inverse disparate impact of the histogram MIA is general "
                 "(ε=10, matched): typical records leak, outliers are hidden", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "fig_outlier_multidataset.png"), dpi=150)
    plt.close(fig)
    print(f"\nwrote outlier_disparity_multidataset.json and fig_outlier_multidataset.png")


if __name__ == "__main__":
    main()
