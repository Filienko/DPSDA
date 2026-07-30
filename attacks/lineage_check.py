"""Does the geometry-aware LINEAGE signal recover outlier leakage the count attack
misses?

The count-based histogram MIA hides private outliers (see outlier_disparity.py):
an outlier lands in a low-count / uncovered cell and scores like a non-member. But
``histogram_mia.lineage_density_records`` scores a different signal -- the time-
averaged proximity of *surviving* synthetic cells to the record, calibrated by a
reference density. A member outlier may still have seeded a survivor cell near
itself, so lineage could expose outliers even when their vote count does not. This
script compares, per outlierness group, the member-vs-nonmember AUC under:
  count   : the released-histogram noised LLR (the standard attack),
  lineage : the calibrated survivor-proximity score,
  combined: z-scored sum of the two.
If the outlier-group AUC under lineage >> under count, "outliers are safe" is
attack-specific. Matched setup, epsilon=10.
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attacks.audit_set import build_audit_set, reference_rows_by_class
from attacks.reconstruct import reconstruct
from attacks.histogram_mia import score_records, lineage_density_records
from attacks.tabular_embedding import load_private, make_embed_fn
from attacks.outlier_disparity import DATASETS, _paths, knn_outlierness, OUT
from pe.constant.data import LABEL_ID_COLUMN_NAME, TABULAR_DATA_COLUMN_NAME

SEED = 0
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"count": "#D55E00", "lineage": "#0072B2", "combined": "#009E73"}


def _z(x):
    x = np.asarray(x, float); f = x[np.isfinite(x)]
    if f.size == 0:
        return np.zeros_like(x)
    mu, sd = f.mean(), f.std()
    return (x - mu) / sd if sd > 0 else x - mu


def analyze(cfg):
    train, test, meta = _paths(cfg)
    priv, info = load_private(train, meta)
    embed_fn = make_embed_fn(priv, info)
    n_priv = {int(k): int(v) for k, v in
              priv.data_frame[LABEL_ID_COLUMN_NAME].value_counts().items()}
    train_emb = embed_fn([list(r) for r in priv.data_frame[TABULAR_DATA_COLUMN_NAME].tolist()])
    train_cls = priv.data_frame[LABEL_ID_COLUMN_NAME].to_numpy().astype(int)

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

    count = np.full(len(classes), np.nan)
    lin = np.full(len(classes), np.nan)
    for c in np.unique(classes):
        iters = by_class.get(int(c))
        if not iters:
            continue
        mask = classes == c
        count[mask] = score_records(query_emb[mask], iters, mode="L2",
                                    regime="noised", sigma=cfg["nm"], censored=False)
        lin[mask] = lineage_density_records(query_emb[mask], iters, mode="L2",
                                            reference_features=ref_by_class.get(int(c)))
    for arr in (count, lin):
        fin = arr[np.isfinite(arr)]
        arr[~np.isfinite(arr)] = (fin.min() - 1.0) if fin.size else 0.0
    combined = _z(count) + _z(lin)

    mem, non = labels == 1, labels == 0
    out_rank = knn_outlierness(query_emb[mem], classes[mem], train_emb, train_cls)
    edges = np.quantile(out_rank, [1/3, 2/3])
    grp = np.where(out_rank <= edges[0], "inlier",
                   np.where(out_rank <= edges[1], "mid", "outlier"))

    def group_auc(score):
        sm, sn = score[mem], score[non]
        res = {}
        for g in ["inlier", "outlier"]:
            gs = sm[grp == g]
            y = np.r_[np.ones(len(gs)), np.zeros(len(sn))]
            res[g] = round(float(roc_auc_score(y, np.r_[gs, sn])), 4)
        return res

    return {"name": cfg["name"],
            "count": group_auc(count), "lineage": group_auc(lin),
            "combined": group_auc(combined)}


def main():
    results = [analyze(cfg) for cfg in DATASETS]
    print(f"{'dataset':22s} | {'count in/out':>14s} | {'lineage in/out':>16s} "
          f"| {'combined in/out':>16s}")
    for r in results:
        print(f"{r['name']:22s} | {r['count']['inlier']:.3f}/{r['count']['outlier']:.3f}   "
              f"| {r['lineage']['inlier']:.3f}/{r['lineage']['outlier']:.3f}     "
              f"| {r['combined']['inlier']:.3f}/{r['combined']['outlier']:.3f}")
    json.dump({"epsilon": 10, "setup": "matched", "datasets": results},
              open(os.path.join(OUT, "lineage_check.json"), "w"), indent=2)

    # Figure: outlier-group AUC under count vs lineage vs combined, per dataset.
    names = [r["name"] for r in results]
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for off, m in zip([-w, 0, w], ["count", "lineage", "combined"]):
        ax.bar(x + off, [r[m]["outlier"] for r in results], w, color=C[m], label=m)
    ax.axhline(0.5, color="silver", lw=1.2, alpha=0.8)
    ax.annotate("chance", (len(names) - 0.5, 0.5), fontsize=8, color="gray", va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("OUTLIER-group member-vs-nonmember AUC"); ax.set_ylim(0.38, 0.85)
    ax.set_title("Does a geometry-aware (lineage) attack recover private outliers\n"
                 "that the count attack hides?  (ε=10, matched)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_lineage_check.png"), dpi=150)
    plt.close(fig)
    print(f"\nwrote lineage_check.json and fig_lineage_check.png under "
          f"{os.path.relpath(OUT, os.path.dirname(os.path.abspath(__file__)))}/")


if __name__ == "__main__":
    main()
