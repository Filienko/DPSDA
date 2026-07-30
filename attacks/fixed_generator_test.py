"""Decisive causal test: does the member<->non-member DISTRIBUTION gap drive the
histogram-MIA AUC, holding the generator (and its released vote histogram) FIXED?

In the sweep, every variant reran PE, so "distribution gap" was confounded with
"which records are members / what the generator memorized". Here we remove that
confound. We take ONE fixed PE run (the q18 checkpoint), so the synthetic cells
and released vote counts are byte-identical throughout. We also FIX:
  * the member challenges (500 real members of that run),
  * the reference D_ref used for the null.
and vary ONLY the non-member challenge set: real records drawn from PCA-1 bands
at increasing distance from the member band (near -> far). Everything the attack
reads from the model is unchanged; the sole independent variable is the
non-member distribution.

If AUC stays ~flat  -> the sweep's rise was memorization / member-set identity,
                       and "driven by the distribution gap" is WRONG.
If AUC rises        -> with the histogram fixed, only the non-member distribution
                       changed, so the gap is causal.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from attacks.reconstruct import reconstruct
from attacks.histogram_mia import score_records
from attacks.tabular_embedding import load_private, make_embed_fn
from pe.constant.data import LABEL_ID_COLUMN_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "example", "tabular", "dist_shift", "data")
NAT = os.path.join(DATA, "natural")
META = os.path.join(DATA, "metadata.json")
RUN = "natural_q50"          # members = bottom-50% band, subsampled to 1500
CKPT = os.path.join(HERE, "..", "results", "tabular", "dist_shift",
                    RUN, "checkpoint")
MEMBERS_CSV = os.path.join(NAT, "members_q50.csv")
NM = 1.7105092023506527
SEED = 0

# non-member PCA-1 percentile bands (per class), increasing distance from the
# member band (members = subsample of bottom 50%). "matched" draws real records
# from the SAME [0-50] region (disjoint from the member rows) -> the pure
# memorization floor at matched distribution; the rest move progressively away.
BANDS = {"matched[0-50]": (0, 50), "near[55-70]": (55, 70),
         "mid[70-85]": (70, 85), "far[85-100]": (85, 100)}
N_MEM_CH = 500
N_NM_CH = 500
N_REF = 1500


def pca1(pool, fc):
    X = pool[fc].to_numpy(float); mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Z = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    return Z @ Vt[0]


def _stratified_band(pool, s, labels, lo, hi, n, exclude, rng):
    """Sample n rows (stratified by class) whose per-class PCA-1 percentile is in
    [lo, hi], excluding row indices in `exclude`."""
    picks = []
    classes = np.unique(labels)
    per = {c: 0 for c in classes}
    # target per-class proportional to class size
    sizes = {c: int((labels == c).sum()) for c in classes}
    tot = sum(sizes.values())
    for c in classes:
        idx = np.where(labels == c)[0]
        idx = np.array([i for i in idx if i not in exclude])
        sc = s[idx]
        order = idx[np.argsort(sc)]
        a, b = int(lo / 100 * len(order)), int(hi / 100 * len(order))
        band = order[a:b]
        k = max(1, int(round(n * sizes[c] / tot)))
        k = min(k, len(band))
        picks.append(rng.choice(band, size=k, replace=False))
    return np.concatenate(picks)


def main():
    meta = json.load(open(META))
    fc = meta["int_columns"] + meta["float_columns"]
    lab = meta["label_columns"][0]

    # Fixed generator: the members behind the fixed checkpoint are the private set.
    members_csv = MEMBERS_CSV
    priv, info = load_private(members_csv, META)
    embed_fn = make_embed_fn(priv, info)
    n_priv_by_class = {int(k): int(v) for k, v in
                       priv.data_frame[LABEL_ID_COLUMN_NAME].value_counts().items()}

    m_df = pd.read_csv(members_csv)
    rng = np.random.default_rng(SEED)

    # Fixed member challenges: 500 real members.
    m_ch_idx = rng.choice(len(m_df), size=min(N_MEM_CH, len(m_df)), replace=False)
    m_rows = m_df.iloc[m_ch_idx][fc].values.tolist()
    m_cls = m_df.iloc[m_ch_idx][lab].astype(int).tolist()

    # Pool of real records for non-members + reference, with PCA-1 scores.
    pool = pd.concat([pd.read_csv(os.path.join(DATA, "train.csv")),
                      pd.read_csv(os.path.join(DATA, "test.csv"))], ignore_index=True)
    s = pca1(pool, fc)
    labels = pool[lab].to_numpy().astype(int)
    member_keys = set(map(tuple, m_df[fc].values.tolist()))
    excl = set(i for i in range(len(pool))
               if tuple(pool.iloc[i][fc].tolist()) in member_keys)

    # Fixed reference D_ref: a neutral held-out sample from the whole pool
    # (same for every band). Drawn first and added to the exclusion set.
    avail = np.array([i for i in range(len(pool)) if i not in excl])
    ref_idx = rng.choice(avail, size=min(N_REF, len(avail)), replace=False)
    excl_ref = excl | set(int(i) for i in ref_idx)
    ref_rows_by_class = {}
    for c in np.unique(labels[ref_idx]):
        ii = ref_idx[labels[ref_idx] == c]
        ref_rows_by_class[int(c)] = pool.iloc[ii][fc].values.tolist()
    ref_by_class = {c: embed_fn(r) for c, r in ref_rows_by_class.items()}

    # Fixed histogram from the fixed checkpoint (identical across all bands).
    by_class = reconstruct(CKPT, embed_fn, ref_by_class, n_priv_by_class,
                           start_t=1, use_clean=False)

    def score_audit(rows, classes, labels_vec):
        classes = np.asarray(classes)
        emb = embed_fn(rows)
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
        return roc_auc_score(labels_vec, scores)

    print(f"FIXED generator ({RUN} checkpoint), FIXED members & D_ref; "
          "only the non-member distribution varies.\n")
    print(f"{'non-member band':18s} {'mean|s|-gap':>11s} {'MIA AUC':>9s}")
    out = {"checkpoint": RUN, "fixed": ["members", "histogram", "D_ref"],
           "bands": {}}
    m_s = float(np.mean([s[i] for i in excl if i < len(s)]))  # member-band mean s
    m_s = float(np.mean(s[list(excl)])) if excl else 0.0
    for name, (lo, hi) in BANDS.items():
        nm_idx = _stratified_band(pool, s, labels, lo, hi, N_NM_CH, excl_ref, rng)
        nm_rows = pool.iloc[nm_idx][fc].values.tolist()
        nm_cls = labels[nm_idx].tolist()
        rows = m_rows + nm_rows
        classes = m_cls + nm_cls
        y = np.array([1] * len(m_rows) + [0] * len(nm_rows))
        auc = score_audit(rows, classes, y)
        gap = abs(float(np.mean(s[nm_idx])) - m_s)
        out["bands"][name] = {"pca1_gap_vs_members": round(gap, 3),
                              "auc": round(float(auc), 4),
                              "n_nonmembers": int(len(nm_idx))}
        print(f"{name:18s} {gap:11.3f} {auc:9.4f}")

    outp = os.path.join(HERE, "..", "results", "tabular", "dist_shift",
                        "analysis", "fixed_generator_test.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
