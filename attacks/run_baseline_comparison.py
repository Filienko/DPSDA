"""Three-way comparison on the same 4 tabular runs (eps~10): GAN-Leaks/FBB
(density-only baseline, final release only) vs histogram-LLR (our mechanism-aware
count attack) vs selection-only (our mechanism-aware, no-count attack).

Reuses ONE shared, untimed embedding/data-loading pass per dataset (audit set +
reference set + all per-round candidate pools) -- every method needs embeddings,
so that cost is common overhead and excluded from all three. What IS timed,
separately per method, is only the core attack compute: NearestNeighbors
fit+query for FBB; the score_records log-likelihood-ratio loop for histogram-LLR;
the score_records_selection Bernoulli-LLR + threshold-bisection loop for
selection-only. Timed with time.time() bracketing exactly that call.
"""

import time
import numpy as np

from attacks.audit_set import build_audit_set, reference_rows_by_class
from attacks.reconstruct import reconstruct
from attacks.selection_mia import reconstruct_selection, score_records_selection
from attacks.gan_leaks_fbb import fbb_scores
from attacks.evaluate import evaluate
from attacks.histogram_mia import score_records
from attacks.tabular_embedding import load_private, make_embed_fn
from pe.constant.data import LABEL_ID_COLUMN_NAME

BASE = "https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/heads/main/tabular/real"

RUNS = [
    dict(slug="adult", checkpoint="results/tabular/adult_composite_population/checkpoint",
        noise_multiplier=2.8139087367478592, n_members=500, n_nonmembers=500),
    dict(slug="artificial-characters",
        checkpoint="results/tabular/artificial-characters_composite_population/checkpoint",
        noise_multiplier=1.8385747404944874, n_members=500, n_nonmembers=500),
    dict(slug="breast-cancer",
        checkpoint="results/tabular/breast-cancer_composite_population/checkpoint",
        noise_multiplier=1.853642779702104, n_members=500, n_nonmembers=500),
    dict(slug="person-activity",
        checkpoint="results/tabular/person-activity_composite_population/checkpoint",
        noise_multiplier=2.0436299197813614, n_members=500, n_nonmembers=500),
]


def run_one(cfg):
    slug = cfg["slug"]
    train_csv = f"{BASE}/{slug}/{slug}_train.csv"
    test_csv = f"{BASE}/{slug}/{slug}_test.csv"
    metadata = f"{BASE}/{slug}/{slug}_metadata.json"

    # --- shared, UNTIMED embedding / data-loading pass -------------------------
    priv_data, info = load_private(train_csv, metadata)
    embed_fn = make_embed_fn(priv_data, info)
    n_private_by_class = {int(k): int(v) for k, v in
                          priv_data.data_frame[LABEL_ID_COLUMN_NAME].value_counts().items()}

    rows, classes, labels, _ = build_audit_set(
        train_csv, test_csv, metadata, n_members=cfg["n_members"],
        n_nonmembers=cfg["n_nonmembers"], seed=0, ref_holdout_frac=0.5)
    ref_rows = reference_rows_by_class(test_csv, metadata, max_per_class=2000,
                                       seed=0, ref_holdout_frac=0.5)
    ref_by_class = {cls: embed_fn(r) for cls, r in ref_rows.items()}
    query_emb = embed_fn(rows)
    classes_arr = np.asarray(classes)

    by_class_hist = reconstruct(cfg["checkpoint"], embed_fn, ref_by_class,
                                n_private_by_class, start_t=1, use_clean=False)
    by_class_sel = reconstruct_selection(cfg["checkpoint"], embed_fn, ref_by_class,
                                         n_private_by_class, start_t=1)
    # -----------------------------------------------------------------------

    sigma = cfg["noise_multiplier"]
    out = {}

    # --- GAN-Leaks / FBB: TIMED (fit + kneighbors per class) --------------------
    fbb_raw = np.full(len(classes), np.nan)
    t0 = time.time()
    for cls in np.unique(classes_arr):
        iters = by_class_hist.get(int(cls))
        if not iters:
            continue
        gen_features = iters[-1]["cell_features"]  # last released round = "generator output"
        mask = classes_arr == cls
        fbb_raw[mask] = fbb_scores(query_emb[mask], gen_features)
    t_fbb = time.time() - t0
    finite = fbb_raw[np.isfinite(fbb_raw)]
    fbb_raw[np.isnan(fbb_raw)] = (finite.min() - 1.0) if finite.size else 0.0
    out["fbb"] = {"auc": evaluate(labels, fbb_raw)["auc"], "time_s": t_fbb}

    # --- histogram-LLR: TIMED (score_records loop per class) --------------------
    hist_raw = np.full(len(classes), np.nan)
    t0 = time.time()
    caches = {}
    for cls in np.unique(classes_arr):
        iters = by_class_hist.get(int(cls))
        if not iters:
            continue
        mask = classes_arr == cls
        cache = caches.setdefault(int(cls), {})
        hist_raw[mask] = score_records(
            query_emb[mask], iters, mode="L2", regime="noised", gen_k=1,
            sigma=sigma, threshold=0.0, censored=False, ref_alpha=0.05,
            soft_tau=0.02, dispersion=1.8, occupancy_cache=cache)
    t_hist = time.time() - t0
    finite = hist_raw[np.isfinite(hist_raw)]
    hist_raw[np.isnan(hist_raw)] = (finite.min() - 1.0) if finite.size else 0.0
    out["histogram_llr"] = {"auc": evaluate(labels, hist_raw)["auc"], "time_s": t_hist}

    # --- selection-only: TIMED (score_records_selection loop per class) --------
    sel_raw = np.full(len(classes), np.nan)
    t0 = time.time()
    caches = {}
    for cls in np.unique(classes_arr):
        iters = by_class_sel.get(int(cls))
        if not iters:
            continue
        mask = classes_arr == cls
        cache = caches.setdefault(int(cls), {})
        sel_raw[mask] = score_records_selection(
            query_emb[mask], iters, mode="L2", sigma=sigma, dispersion=1.8,
            ref_alpha=0.05, soft_tau=0.02, occupancy_cache=cache)
    t_sel = time.time() - t0
    finite = sel_raw[np.isfinite(sel_raw)]
    sel_raw[np.isnan(sel_raw)] = (finite.min() - 1.0) if finite.size else 0.0
    out["selection_only"] = {"auc": evaluate(labels, sel_raw)["auc"], "time_s": t_sel}

    return out


def main():
    print(f"{'dataset':<24}{'method':<18}{'AUC':>8}{'core_time_s':>14}")
    for cfg in RUNS:
        res = run_one(cfg)
        for method, m in res.items():
            print(f"{cfg['slug']:<24}{method:<18}{m['auc']:>8.4f}{m['time_s']:>14.4f}")


if __name__ == "__main__":
    main()
