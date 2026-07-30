"""Build member/aux splits with a NATURAL, sampling-induced distribution gap.

Instead of injecting an artificial coordinate offset, we partition the *real*
pooled records so that the private (member) set and the auxiliary/holdout (aux)
set naturally sit at a controllable distributional distance -- every row is a
genuine record.

How:
  1. Pool all real records (train + test) and z-score the 7 numeric features.
  2. Fit the top principal component (PCA-1) -- the data-driven axis of maximum
     variance -- and score each record s(x) = <z(x), v1>.
  3. WITHIN each class (so the class prior P(y) is preserved and every class stays
     represented -- a pure covariate shift), sort by s and take:
        members  <- the LOW-s band,   aux <- the HIGH-s band.
     A narrower band => the two pools are pulled toward opposite tails => a larger
     natural member<->aux distance. The `iid` variant instead splits each class at
     random => matched distributions (the classical MIA assumption, distance ~ 0).
  4. Subsample members and aux to FIXED sizes (N_MEM / N_AUX) so N is constant
     across variants and only the *separation* changes, not the set size.

Outputs, per variant tag, into data/natural/:
    members_{tag}.csv   -- private set fed to PE (and the MIA member pool)
    aux_{tag}.csv       -- holdout: MIA non-members + reference D_ref
It also writes split_summary.json with the realized member<->aux distances
(mean-embedding gap in the attack's TabularEmbedding space, plus PCA-1 Wasserstein).
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from pe.data import TabularCSV
from pe.embedding import TabularEmbedding
from pe.constant.data import TABULAR_DATA_COLUMN_NAME, LABEL_ID_COLUMN_NAME
from attacks.tabular_embedding import make_embed_fn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "natural")

N_MEM = 1500          # fixed private-set size across variants
N_AUX = 1500          # fixed aux (non-member + reference) size across variants
SEED = 0

# tag -> band fraction f (members = lowest f of each class by PCA-1, aux = highest
# f). None => random iid split (matched distributions, the ρ≈0 baseline).
VARIANTS = {"iid": None, "q50": 0.50, "q35": 0.35, "q25": 0.25, "q18": 0.18}


def _feature_matrix(df, feat_cols):
    return df[feat_cols].to_numpy(dtype=np.float64)


def _pca1_scores(X):
    """Signed PCA-1 projection of z-scored X (fit on the whole pool)."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    # top right singular vector of centered Z
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    return Z @ Vt[0], Vt[0]


def _stratified_take(idx_by_class, s, frac, n_target, high, rng):
    """From each class, take the low- (high=False) or high- (high=True) s band of
    width `frac`, then subsample the union to n_target, stratified by class."""
    band = {}
    for c, idx in idx_by_class.items():
        order = idx[np.argsort(s[idx])]          # ascending s
        k = max(1, int(round(frac * len(order))))
        band[c] = order[-k:] if high else order[:k]
    return _subsample(band, n_target, rng)


def _stratified_random(idx_by_class, n_a, n_b, rng):
    """Random disjoint stratified split of each class into two fixed-size pools."""
    a, b = {}, {}
    fa = n_a / sum(len(v) for v in idx_by_class.values())
    fb = n_b / sum(len(v) for v in idx_by_class.values())
    for c, idx in idx_by_class.items():
        perm = rng.permutation(idx)
        ka = max(1, int(round(fa * len(idx))))
        kb = max(1, int(round(fb * len(idx))))
        a[c], b[c] = perm[:ka], perm[ka:ka + kb]
    return _subsample(a, n_a, rng), _subsample(b, n_b, rng)


def _subsample(band_by_class, n_target, rng):
    total = sum(len(v) for v in band_by_class.values())
    keep = []
    for c, idx in band_by_class.items():
        k = max(1, int(round(n_target * len(idx) / total)))
        k = min(k, len(idx))
        keep.append(rng.choice(idx, size=k, replace=False))
    return np.concatenate(keep)


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(DATA, "metadata.json")))
    feat_cols = meta["int_columns"] + meta["float_columns"]

    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    pool = pd.concat([tr, te], ignore_index=True)
    labels = pool["Class"].to_numpy()
    X = _feature_matrix(pool, feat_cols)
    s, v1 = _pca1_scores(X)
    print("PCA-1 loadings:", dict(zip(feat_cols, np.round(v1, 3))))

    idx_by_class = {int(c): np.where(labels == c)[0] for c in np.unique(labels)}

    # Embedding used by the attack (bounds from the full pool -> shared frame so the
    # realized distance is measured in the same space the histogram MIA operates in).
    pool_tab = TabularCSV(csv_path=os.path.join(DATA, "train.csv"),
                          metadata_path=os.path.join(DATA, "metadata.json"))
    embed_fn = make_embed_fn(pool_tab, pool_tab.get_tab_info())

    summary = {"n_mem": N_MEM, "n_aux": N_AUX, "pca1_loadings":
               dict(zip(feat_cols, [round(float(x), 4) for x in v1])), "variants": {}}

    for tag, frac in VARIANTS.items():
        rng = np.random.default_rng([SEED, hash(tag) & 0xFFFF])
        if frac is None:
            m_idx, a_idx = _stratified_random(idx_by_class, N_MEM, N_AUX, rng)
        else:
            m_idx = _stratified_take(idx_by_class, s, frac, N_MEM, high=False, rng=rng)
            a_idx = _stratified_take(idx_by_class, s, frac, N_AUX, high=True, rng=rng)
        # guarantee disjoint (bands are disjoint for frac<=0.5; iid is disjoint by
        # construction) -- assert to be safe.
        assert len(set(m_idx) & set(a_idx)) == 0, f"{tag}: member/aux overlap"

        m_df = pool.iloc[m_idx].reset_index(drop=True)
        a_df = pool.iloc[a_idx].reset_index(drop=True)
        m_df.to_csv(os.path.join(OUT, f"members_{tag}.csv"), index=False)
        a_df.to_csv(os.path.join(OUT, f"aux_{tag}.csv"), index=False)

        # Realized distances.
        emb_m = embed_fn(m_df[feat_cols].values.tolist())
        emb_a = embed_fn(a_df[feat_cols].values.tolist())
        mean_gap = float(np.linalg.norm(emb_m.mean(0) - emb_a.mean(0)))
        w_pca1 = float(wasserstein_distance(s[m_idx], s[a_idx]))
        summary["variants"][tag] = {
            "band_frac": frac, "n_members": len(m_idx), "n_aux": len(a_idx),
            "mean_embedding_gap": round(mean_gap, 4),
            "pca1_wasserstein": round(w_pca1, 4),
            "members_pca1_mean": round(float(s[m_idx].mean()), 4),
            "aux_pca1_mean": round(float(s[a_idx].mean()), 4),
        }
        print(f"{tag:5s} frac={frac} mean_emb_gap={mean_gap:.4f} "
              f"pca1_W={w_pca1:.4f} (mem_s={s[m_idx].mean():+.3f} "
              f"aux_s={s[a_idx].mean():+.3f})")

    json.dump(summary, open(os.path.join(OUT, "split_summary.json"), "w"), indent=2)
    print(f"\nwrote {os.path.relpath(OUT, HERE)}/ (members_*, aux_*, split_summary.json)")


if __name__ == "__main__":
    main()
