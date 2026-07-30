"""Isolate the effect of the PRIVATE <-> AUX distribution gap on the attack.

The classical MIA threat model assumes the attacker's auxiliary/reference set
(here D_ref, used to estimate the occupancy null q_j) is drawn from the SAME
distribution as the private data. Aug-PE breaks that: the public/aux pool need not
match the private distribution. This script measures what that mismatch alone does
-- to the attack AUC and to the vote histogram (private occupancy vs the aux-
induced null it is scored against).

Everything is held fixed except the aux distribution:
  * generator: one checkpoint (natural_iid) -> synthetic cells + released vote
    counts are byte-identical throughout,
  * private/members: the iid member set behind that checkpoint,
  * member AND non-member challenges: both drawn matched to the private
    distribution (member/non-member gap ~ 0, so it cannot confound),
Only D_ref varies, from matched-to-private to progressively shifted PCA-1 bands.

Reports, per aux distribution: AUC, and cosine(private vote histogram, aux null)
-- the "how close are the private and aux distributions in the histogram" number.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from attacks.reconstruct import reconstruct
from attacks.histogram_mia import (score_records, cell_occupancy, occupancy_mmd,
                                   sample_mmd)
from attacks.tabular_embedding import load_private, make_embed_fn
from attacks.audit_set import build_audit_set, reference_rows_by_class
from pe.constant.data import LABEL_ID_COLUMN_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "example", "tabular", "dist_shift", "data")
NAT = os.path.join(DATA, "natural")
META = os.path.join(DATA, "metadata.json")
RUN = "natural_iid"
CKPT = os.path.join(HERE, "..", "results", "tabular", "dist_shift", RUN, "checkpoint")
MEMBERS_CSV = os.path.join(NAT, "members_iid.csv")
AUX_CSV = os.path.join(NAT, "aux_iid.csv")     # matched holdout for the challenges
NM = 1.7105092023506527
SEED = 0
N_REF = 1500
# aux / D_ref distributions to score the FIXED challenges against. "matched" is
# the in-distribution holdout the real attack uses (reproduces the sweep AUC); the
# rest sample D_ref from increasingly shifted per-class PCA-1 windows of the pool.
AUX = {"matched": None, "shift[30-100]": (30, 100), "shift[50-100]": (50, 100),
       "shift[65-100]": (65, 100), "shift[78-100]": (78, 100),
       "shift[88-100]": (88, 100)}


def pca1(pool, fc):
    X = pool[fc].to_numpy(float); mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Z = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    return Z @ Vt[0]


def _band_by_class(pool, s, labels, lo, hi, n, exclude, rng):
    picks, classes = [], np.unique(labels)
    sizes = {c: int((labels == c).sum()) for c in classes}; tot = sum(sizes.values())
    for c in classes:
        idx = np.array([i for i in np.where(labels == c)[0] if i not in exclude])
        if len(idx) == 0:
            continue
        order = idx[np.argsort(s[idx])]
        a, b = int(lo / 100 * len(order)), int(hi / 100 * len(order))
        band = order[a:max(b, a + 1)]
        k = min(max(1, int(round(n * sizes[c] / tot))), len(band))
        picks.append(rng.choice(band, size=k, replace=False))
    return np.concatenate(picks)


def votehist_mmd(by_class):
    """Kernel-MMD between the released private occupancy and the aux-induced null,
    averaged over rounds/classes (RBF kernel over cell embeddings; larger = the
    private vote histogram sits farther from the aux null)."""
    ms = []
    for cls, iters in by_class.items():
        for it in iters:
            c = np.clip(np.asarray(it["counts"], float), 0, None)
            if c.sum() <= 0:
                continue
            p = c / c.sum()
            q = cell_occupancy(it["reference_features"], it["cell_features"], mode="L2")
            q = q / q.sum()
            ms.append(occupancy_mmd(p, q, it["cell_features"]))
    return float(np.mean(ms)) if ms else float("nan")


def main():
    meta = json.load(open(META)); fc = meta["int_columns"] + meta["float_columns"]
    lab = meta["label_columns"][0]

    priv, info = load_private(MEMBERS_CSV, META)
    embed_fn = make_embed_fn(priv, info)
    n_priv = {int(k): int(v) for k, v in
              priv.data_frame[LABEL_ID_COLUMN_NAME].value_counts().items()}
    m_df = pd.read_csv(MEMBERS_CSV)
    rng = np.random.default_rng(SEED)

    # FIXED challenges built exactly as the real attack does (members vs the matched
    # iid holdout), so the matched-D_ref row reproduces the sweep's iid AUC.
    rows, classes, y, _ = build_audit_set(MEMBERS_CSV, AUX_CSV, META,
                                          n_members=500, n_nonmembers=500, seed=SEED,
                                          ref_holdout_frac=0.5)
    classes = np.asarray(classes)
    emb = embed_fn(rows)
    priv_emb = embed_fn(m_df[fc].values.tolist())     # all private/member records
    priv_mean_emb = priv_emb.mean(0)

    pool = pd.concat([pd.read_csv(os.path.join(DATA, "train.csv")),
                      pd.read_csv(os.path.join(DATA, "test.csv"))], ignore_index=True)
    s = pca1(pool, fc); labels = pool[lab].to_numpy().astype(int)

    print(f"FIXED generator ({RUN}), FIXED matched member & non-member challenges; "
          "only the AUX / D_ref distribution varies.\n")
    print(f"{'aux (D_ref) dist':18s} {'priv-aux MMD':>12s} {'voteHist MMD':>13s} {'MIA AUC':>9s}")
    out = {"checkpoint": RUN, "fixed": ["generator", "members", "nonmembers"], "aux": {}}
    for name, band in AUX.items():
        if band is None:
            ref_rows = reference_rows_by_class(AUX_CSV, META, max_per_class=2000,
                                               seed=SEED, ref_holdout_frac=0.5)
            ref_by_class = {c: embed_fn(r) for c, r in ref_rows.items()}
            aux_emb = np.concatenate([np.asarray(e, float)
                                      for e in ref_by_class.values()])
        else:
            lo, hi = band
            ref_idx = _band_by_class(pool, s, labels, lo, hi, N_REF, set(), rng)
            ref_by_class = {}
            for c in np.unique(labels[ref_idx]):
                ii = ref_idx[labels[ref_idx] == c]
                ref_by_class[int(c)] = embed_fn(pool.iloc[ii][fc].values.tolist())
            aux_emb = embed_fn(pool.iloc[ref_idx][fc].values.tolist())

        # actual private<->aux distribution distance: two-sample RBF-MMD on records.
        priv_aux_mmd = sample_mmd(priv_emb, aux_emb, seed=SEED)
        aux_gap = float(np.linalg.norm(aux_emb.mean(0) - priv_mean_emb))

        by_class = reconstruct(CKPT, embed_fn, ref_by_class, n_priv,
                               start_t=1, use_clean=False)
        scores = np.full(len(classes), np.nan)
        for c in np.unique(classes):
            iters = by_class.get(int(c))
            if not iters:
                continue
            mask = classes == c
            scores[mask] = score_records(emb[mask], iters, mode="L2",
                                         regime="noised", sigma=NM, censored=False)
        fin = scores[np.isfinite(scores)]
        scores[~np.isfinite(scores)] = (fin.min() - 1.0) if fin.size else 0.0
        auc = roc_auc_score(y, scores)
        mmd = votehist_mmd(by_class)
        out["aux"][name] = {"priv_aux_mmd": round(priv_aux_mmd, 4),
                            "aux_priv_embedding_gap": round(aux_gap, 4),
                            "votehist_mmd": round(mmd, 4), "auc": round(float(auc), 4)}
        print(f"{name:18s} {priv_aux_mmd:12.4f} {mmd:13.4f} {auc:9.4f}")

    outp = os.path.join(HERE, "..", "results", "tabular", "dist_shift",
                        "analysis", "aux_shift_test.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
