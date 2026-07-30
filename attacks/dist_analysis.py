"""Distribution-vs-membership analysis for the natural-shift sweep.

For each variant (a member/aux split at a known distributional distance) this:

  1. Rebuilds the exact MIA audit set (members vs aux non-members) and scores it
     with the released histogram MIA (noised regime, with D_ref) -- the real attack.
  2. Scores the SAME audit set with a distribution-only detector: the record's
     PCA-1 projection (the axis the split was made on), which an attacker could
     compute from public marginals WITHOUT any access to the synthetic histogram.
     If this baseline's AUC rises with the shift, the apparent "membership leak"
     is really distribution detection -- the classical MIA assumption (members,
     non-members, reference all i.i.d.) is what makes AUC meaningful, and it is
     violated here.
  3. Compares the released VOTE HISTOGRAM (per-cell private occupancy) against the
     aux-induced occupancy the null assumes, per round -> TVD / cosine. This is the
     "how close are the private and aux distributions in the histogram" question.
  4. Correlates all of the above with the realized member<->aux distance and with
     downstream utility.

Writes results/tabular/dist_shift/analysis/summary.json and prints a table.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from attacks.audit_set import build_audit_set, reference_rows_by_class
from attacks.reconstruct import reconstruct
from attacks.histogram_mia import (score_records, nearest_cell, cell_occupancy,
                                   occupancy_mmd)
from attacks.tabular_embedding import load_private, make_embed_fn
from pe.constant.data import LABEL_ID_COLUMN_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "results", "tabular", "dist_shift")
DATA = os.path.join(HERE, "..", "example", "tabular", "dist_shift", "data")
NAT = os.path.join(DATA, "natural")
META = os.path.join(DATA, "metadata.json")
NM = 1.7105092023506527           # epsilon=10 noise multiplier (N=1500 members)
TAGS = ["iid", "q50", "q35", "q25", "q18"]


def pca1_axis():
    """Recompute the pool PCA-1 axis exactly as make_natural_splits did."""
    meta = json.load(open(META))
    feat_cols = meta["int_columns"] + meta["float_columns"]
    pool = pd.concat([pd.read_csv(os.path.join(DATA, "train.csv")),
                      pd.read_csv(os.path.join(DATA, "test.csv"))], ignore_index=True)
    X = pool[feat_cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    return feat_cols, mu, sd, Vt[0]


def pca1_project(rows, feat_cols, mu, sd, v1):
    X = np.asarray(rows, dtype=float)
    return ((X - mu) / sd) @ v1


def mia_scores(checkpoint, embed_fn, ref_by_class, n_priv_by_class,
               query_emb, classes, labels):
    """Released-histogram MIA (noised, with D_ref) on the given audit set."""
    by_class = reconstruct(checkpoint, embed_fn, ref_by_class, n_priv_by_class,
                           start_t=1, use_clean=False)
    classes = np.asarray(classes)
    scores = np.full(len(classes), np.nan)
    sigma = NM
    for cls in np.unique(classes):
        iters = by_class.get(int(cls))
        if not iters:
            continue
        mask = classes == cls
        scores[mask] = score_records(query_emb[mask], iters, mode="L2",
                                     regime="noised", sigma=sigma, censored=False)
    finite = scores[np.isfinite(scores)]
    scores[np.isnan(scores)] = (finite.min() - 1.0) if finite.size else 0.0
    scores[~np.isfinite(scores)] = finite.min() - 1.0 if finite.size else 0.0
    return scores, by_class


def vote_hist_gap(by_class):
    """Per-round TVD and kernel-MMD between released private occupancy (DP
    histogram) and aux-induced occupancy (the null q_j), averaged over
    rounds/classes. MMD uses an RBF kernel over the cell embeddings, so it is
    geometry-aware (mass on nearby cells counts as similar), unlike TVD."""
    tvds, mmds = [], []
    for cls, iters in by_class.items():
        for it in iters:
            counts = np.asarray(it["counts"], float)
            counts = np.clip(counts, 0, None)
            if counts.sum() <= 0:
                continue
            p = counts / counts.sum()
            q = cell_occupancy(it["reference_features"], it["cell_features"],
                               mode="L2", k=1, alpha=1.0)
            q = q / q.sum()
            tvds.append(0.5 * np.abs(p - q).sum())
            mmds.append(occupancy_mmd(p, q, it["cell_features"]))
    return (float(np.mean(tvds)) if tvds else float("nan"),
            float(np.mean(mmds)) if mmds else float("nan"))


def _last_val(folder, pattern):
    hits = glob.glob(os.path.join(folder, pattern))
    if not hits:
        return None
    df = pd.read_csv(hits[0], header=None, names=["it", "v"])
    return float(df.sort_values("it")["v"].iloc[-1]) if len(df) else None


def analyze_tag(tag, axis):
    feat_cols, mu, sd, v1 = axis
    members = os.path.join(NAT, f"members_{tag}.csv")
    aux = os.path.join(NAT, f"aux_{tag}.csv")
    exp = os.path.join(ROOT, f"natural_{tag}")
    checkpoint = os.path.join(exp, "checkpoint")

    priv_data, info = load_private(members, META)
    embed_fn = make_embed_fn(priv_data, info)
    n_priv_by_class = {int(k): int(v) for k, v in
                       priv_data.data_frame[LABEL_ID_COLUMN_NAME].value_counts().items()}

    rows, classes, labels, _ = build_audit_set(
        members, aux, META, n_members=500, n_nonmembers=500, seed=0,
        ref_holdout_frac=0.5)
    ref_rows = reference_rows_by_class(aux, META, max_per_class=2000, seed=0,
                                       ref_holdout_frac=0.5)
    ref_by_class = {c: embed_fn(r) for c, r in ref_rows.items()}
    query_emb = embed_fn(rows)

    mia, by_class = mia_scores(checkpoint, embed_fn, ref_by_class, n_priv_by_class,
                               query_emb, classes, labels)
    auc_mia = roc_auc_score(labels, mia)

    # Distribution-only detector: members are the low-PCA-1 tail -> score = -s.
    s = pca1_project(rows, feat_cols, mu, sd, v1)
    auc_distonly = roc_auc_score(labels, -s)

    tvd, mmd = vote_hist_gap(by_class)

    return {
        "tag": tag,
        "auc_mia_noised_ref": round(float(auc_mia), 4),
        "auc_distribution_only": round(float(auc_distonly), 4),
        "vote_hist_tvd_priv_vs_aux": round(tvd, 4),
        "vote_hist_mmd_priv_vs_aux": round(mmd, 4),
        "utility_acc_on_aux": _last_val(exp, "tabular_classifier_tabicl_filter_*_test_acc.csv"),
        "fidelity_tvd2_vs_priv": _last_val(exp, "2way-tvd_*_{*}.csv"),
        "fid_vs_priv": _last_val(exp, "fid_*_{*}.csv"),
    }


def main():
    axis = pca1_axis()
    split = json.load(open(os.path.join(NAT, "split_summary.json")))["variants"]
    out = {"noise_multiplier": NM, "rows": []}
    for tag in TAGS:
        if not os.path.isdir(os.path.join(ROOT, f"natural_{tag}", "checkpoint")):
            print(f"skip {tag}: no checkpoint yet"); continue
        r = analyze_tag(tag, axis)
        r["mean_embedding_gap"] = split[tag]["mean_embedding_gap"]
        r["pca1_wasserstein"] = split[tag]["pca1_wasserstein"]
        out["rows"].append(r)
        print(f"{tag:5s} gap={r['mean_embedding_gap']:.3f} | "
              f"MIA_AUC={r['auc_mia_noised_ref']:.3f} "
              f"dist_only_AUC={r['auc_distribution_only']:.3f} | "
              f"voteTVD={r['vote_hist_tvd_priv_vs_aux']:.3f} "
              f"voteMMD={r['vote_hist_mmd_priv_vs_aux']:.4f} | "
              f"util_aux={r['utility_acc_on_aux']}")

    outdir = os.path.join(ROOT, "analysis")
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    print(f"\nwrote {os.path.relpath(os.path.join(outdir, 'summary.json'), HERE)}")


if __name__ == "__main__":
    main()
