"""End-to-end driver for the tabular-PE nearest-neighbor histogram
membership-inference attack / privacy audit.

Same pipeline as ``aug-pe-baseline/attacks/run_mia.py``:
  1. Build a labelled audit set (members = private records used for generation,
     non-members = an in-distribution holdout split).
  2. Reconstruct each PE iteration's synthetic candidate set + released vote
     counts (here: straight from the saved checkpoints).
  3. Score every candidate with the multi-round likelihood-ratio statistic, using
     a separate in-distribution non-private reference set for the null model.
  4. Report ROC AUC, TPR at low FPR, certificate accounting, and a rounds ablation.

Per the run on the breast-cancer data the default regime is **noised** --- it reads
the released ``PE.DP_HISTOGRAM`` (clean + Gaussian noise), i.e. what the attacker
actually sees.

Example:
  python -m attacks.run_mia \
      --checkpoint_folder results/tabular/breast-cancer_composite_population/checkpoint \
      --train_csv  <breast-cancer_train.csv>  --test_csv <breast-cancer_test.csv> \
      --metadata   <breast-cancer_metadata.json> \
      --noise_multiplier 1.853642779702104 --regime noised
"""

import argparse
import json
import numpy as np

from attacks.audit_set import build_audit_set, reference_rows_by_class
from attacks.reconstruct import reconstruct
from attacks.evaluate import evaluate, format_report
from attacks.histogram_mia import score_records
from attacks.tabular_embedding import load_private, make_embed_fn

# Default hosted breast-cancer files (same as example/tabular/breast_cancer.py).
BC_BASE = ("https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/"
           "heads/main/tabular/")


def _score_all(by_class, query_emb, classes, labels, mode, regime, sigma,
               threshold, censored, max_iters=None, caches=None, ref_alpha=1.0,
               soft_tau=0.0, dispersion=1.0, gen_k=1):
    """Score every audit record within its class; returns a score array."""
    classes = np.asarray(classes)
    scores = np.full(len(classes), np.nan)
    for cls in np.unique(classes):
        iters = by_class.get(int(cls))
        if not iters:
            continue
        if max_iters is not None:
            iters = iters[:max_iters]
        if not iters:
            continue
        mask = classes == cls
        cache = None if caches is None else caches.setdefault(int(cls), {})
        scores[mask] = score_records(
            query_emb[mask], iters, mode=mode, regime=regime, gen_k=gen_k,
            sigma=sigma, threshold=threshold, censored=censored,
            ref_alpha=ref_alpha, soft_tau=soft_tau, dispersion=dispersion,
            occupancy_cache=cache)
    # Records whose class never appeared get the lowest score (no evidence).
    finite = scores[np.isfinite(scores)]
    scores[np.isnan(scores)] = (finite.min() - 1.0) if finite.size else 0.0
    return scores


def _lineage_all(by_class, query_emb, classes, mode, bandwidth,
                 ref_by_class=None, max_iters=None):
    """Geometric survivor-proximity score per record, within its class. When
    ``ref_by_class`` is given, the score is calibrated as a survivor/reference log
    density ratio (removes the local data-density confound)."""
    from attacks.histogram_mia import lineage_density_records
    classes = np.asarray(classes)
    scores = np.full(len(classes), np.nan)
    for cls in np.unique(classes):
        iters = by_class.get(int(cls))
        if not iters:
            continue
        if max_iters is not None:
            iters = iters[:max_iters]
        if not iters:
            continue
        mask = classes == cls
        ref = None if ref_by_class is None else ref_by_class.get(int(cls))
        scores[mask] = lineage_density_records(
            query_emb[mask], iters, mode=mode, bandwidth=bandwidth,
            reference_features=ref)
    finite = scores[np.isfinite(scores)]
    scores[np.isnan(scores)] = (finite.min() - 1.0) if finite.size else 0.0
    return scores


def _zscore(x):
    """Standardise a score vector over its finite entries (label-free)."""
    x = np.asarray(x, dtype=np.float64)
    f = x[np.isfinite(x)]
    if f.size == 0:
        return np.zeros_like(x)
    mu, sd = float(f.mean()), float(f.std())
    return (x - mu) / sd if sd > 0 else (x - mu)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_folder", required=True,
                   help="results/.../<run>/checkpoint")
    p.add_argument("--train_csv", default=BC_BASE + "breast-cancer_train.csv",
                   help="private members CSV")
    p.add_argument("--test_csv", default=BC_BASE + "breast-cancer_test.csv",
                   help="in-distribution non-member holdout CSV")
    p.add_argument("--metadata", default=BC_BASE + "breast-cancer_metadata.json")
    p.add_argument("--reference_csv", default="",
                   help="in-distribution non-private split for the null model "
                        "(defaults to --test_csv)")
    p.add_argument("--no_ref", action="store_true",
                   help="run WITHOUT an auxiliary reference set D_ref: the per-cell "
                        "occupancy null falls back to uniform (q_j = k/n_cells), so "
                        "the attacker assumes no in-distribution knowledge")
    p.add_argument("--ref_holdout_frac", type=float, default=0.5,
                   help="when D_ref defaults to the test CSV, reserve this fraction "
                        "of it for the null model so D_ref is DISJOINT from the "
                        "non-member challenge pool (0 = old behaviour, full overlap)")
    p.add_argument("--nn_mode", default="L2", choices=["L2", "IP", "cos_sim"])
    p.add_argument("--regime", default="noised",
                   choices=["raw", "pure", "noised", "both"],
                   help="default 'noised' = released DP_HISTOGRAM; 'both' runs "
                        "raw + pure + noised")
    p.add_argument("--noise_multiplier", type=float, default=0.0,
                   help="Gaussian noise scale used at generation (from the run log)")
    p.add_argument("--num_nearest_neighbor", type=int, default=1)
    p.add_argument("--ref_alpha", type=float, default=0.05,
                   help="Laplace smoothing on the reference cell-occupancy q_j "
                        "(original 1.0 over-smooths sparse cells; 0.05 is tuned).")
    p.add_argument("--soft_tau", type=float, default=0.02,
                   help="Temperature for soft (kernel) reference binning of q_j; "
                        "0 = original hard nearest-cell histogram.")
    p.add_argument("--dispersion", type=float, default=1.8,
                   help="Null cell-count model: var = dispersion*mean. 1.0 = the "
                        "original Poisson null; ~1.8 (negative binomial) fixes the "
                        "low-FPR calibration for overdispersed vote counts.")
    p.add_argument("--count_threshold", type=float, default=0.0)
    p.add_argument("--n_members", type=int, default=500)
    p.add_argument("--n_nonmembers", type=int, default=500)
    p.add_argument("--max_iters", type=int, default=0, help="0 = all iterations")
    p.add_argument("--ref_max_per_class", type=int, default=2000)
    p.add_argument("--start_t", type=int, default=1)
    p.add_argument("--use_lineage", action="store_true",
                   help="also score the geometric survivor-proximity (lineage) "
                        "signal and its z-scored combination with the count LLR; "
                        "reported alongside the count-only baseline (noised regime)")
    p.add_argument("--lineage_bandwidth", type=float, default=0.0,
                   help="Gaussian kernel bandwidth for the lineage signal "
                        "(0 = median nearest-survivor distance, auto)")
    p.add_argument("--out_json", default="")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sigma = args.noise_multiplier * np.sqrt(args.num_nearest_neighbor)
    max_iters = args.max_iters or None
    reference_csv = args.reference_csv or args.test_csv
    # Only carve a disjoint reference partition out of the holdout when D_ref *is*
    # the test CSV. If the user passed a distinct --reference_csv, it is already
    # separate from the non-member pool, so use all of both.
    same_source = (reference_csv == args.test_csv)
    ref_frac = args.ref_holdout_frac if same_source else 0.0

    print("Loading private data / embedding model...")
    priv_data, info = load_private(args.train_csv, args.metadata)
    embed_fn = make_embed_fn(priv_data, info)
    # Public per-class private sizes (the histogram votes once per private record).
    from pe.constant.data import LABEL_ID_COLUMN_NAME
    n_private_by_class = {int(k): int(v) for k, v in
                          priv_data.data_frame[LABEL_ID_COLUMN_NAME].value_counts().items()}
    print(f"  n_private per class: {n_private_by_class}")

    print("Building audit set...")
    # Pass ref_frac to the challenge builder regardless of --no_ref so the member /
    # non-member challenge pool is IDENTICAL for the with- and without-D_ref runs
    # (they must be scored on the same records to be comparable).
    rows, classes, labels, sizes = build_audit_set(
        args.train_csv, args.test_csv, args.metadata,
        n_members=args.n_members, n_nonmembers=args.n_nonmembers, seed=args.seed,
        ref_holdout_frac=ref_frac)
    print(f"  members={int(labels.sum())} non-members={int((labels == 0).sum())} "
          f"(ref_holdout_frac={ref_frac}, D_ref {'DISJOINT' if ref_frac > 0 else 'OVERLAPS'} "
          f"non-members)")

    if args.no_ref:
        # No auxiliary D_ref: leave ref_by_class empty so cell_occupancy falls back
        # to a uniform per-cell null (q_j = k/n_cells) -- the attacker has no
        # in-distribution data to estimate cell popularity from.
        print("Running WITHOUT D_ref -> uniform occupancy null.")
        ref_by_class = {}
        sizes["aux_used"] = 0
    else:
        print("Embedding reference (null model) data...")
        ref_rows = reference_rows_by_class(reference_csv, args.metadata,
                                           max_per_class=args.ref_max_per_class,
                                           seed=args.seed, ref_holdout_frac=ref_frac)
        ref_by_class = {cls: embed_fn(r) for cls, r in ref_rows.items()}
        sizes["aux_used"] = int(sum(len(r) for r in ref_rows.values()))
        print(f"  D_ref size per class: "
              f"{ {c: len(r) for c, r in ref_rows.items()} } (total {sizes['aux_used']})")

    print("Embedding audit records...")
    query_emb = embed_fn(rows)

    report = {"args": vars(args), "n_private_by_class": n_private_by_class,
              "use_ref": not args.no_ref, "ref_frac_effective": ref_frac,
              "n_members": int(labels.sum()), "n_nonmembers": int((labels == 0).sum()),
              "data_sizes": sizes, "regimes": {}}
    regimes = ["raw", "pure", "noised"] if args.regime == "both" else [args.regime]
    for regime in regimes:
        # 'pure' uses the clean histogram; 'raw'/'noised' read the released
        # (noised) DP_HISTOGRAM -- what the attacker actually sees.
        use_clean = (regime == "pure")
        # The tabular Gaussian mechanism is *uncensored* (no max/threshold clip).
        censored = False
        print(f"\nReconstructing iterations (regime={regime}, "
              f"use_clean={use_clean})...")
        by_class = reconstruct(
            args.checkpoint_folder, embed_fn, ref_by_class, n_private_by_class,
            start_t=args.start_t, max_iters=max_iters, use_clean=use_clean)
        n_it = {c: len(v) for c, v in by_class.items()}
        print(f"  aligned iterations per class: {n_it}")
        if not by_class:
            print("  no aligned iterations -- check --start_t / the run.")
            continue

        caches = {}
        scores = _score_all(by_class, query_emb, classes, labels, args.nn_mode,
                            regime, sigma, args.count_threshold, censored,
                            caches=caches, ref_alpha=args.ref_alpha,
                            soft_tau=args.soft_tau, dispersion=args.dispersion,
                            gen_k=args.num_nearest_neighbor)
        m = evaluate(labels, scores)
        print("\n" + format_report(m, title=f"regime={regime} (all rounds)"))
        report["regimes"][regime] = {"all_rounds": m, "ablation": {}}

        # Lineage / survivor-geometry signal: score it on its own and combined
        # (z-scored sum) with the count LLR. Only meaningful for the released
        # (noised/pure) regimes that carry the surviving cells.
        if args.use_lineage and regime != "raw":
            bw = args.lineage_bandwidth or None
            # Calibrated (survivor/reference density-ratio) when D_ref is available.
            lin = _lineage_all(by_class, query_emb, classes, args.nn_mode, bw,
                               ref_by_class=ref_by_class or None)
            lin_raw = _lineage_all(by_class, query_emb, classes, args.nn_mode, bw)
            combined = _zscore(scores) + _zscore(lin)
            m_lin = evaluate(labels, lin)
            m_lin_raw = evaluate(labels, lin_raw)
            m_comb = evaluate(labels, combined)
            print(format_report(m_lin, title=f"regime={regime} LINEAGE-calibrated"))
            print(format_report(m_lin_raw, title=f"regime={regime} LINEAGE-raw"))
            print(format_report(m_comb, title=f"regime={regime} count+lineage(calib)"))
            report["regimes"][regime]["lineage_calibrated"] = m_lin
            report["regimes"][regime]["lineage_raw"] = m_lin_raw
            report["regimes"][regime]["count_plus_lineage"] = m_comb

        # Rounds ablation: AUC vs number of aggregated iterations.
        max_t = max(len(v) for v in by_class.values())
        for T in sorted(set([1, 2, 5, 10, max_t]) & set(range(1, max_t + 1))):
            sc = _score_all(by_class, query_emb, classes, labels, args.nn_mode,
                           regime, sigma, args.count_threshold, censored,
                           max_iters=T, caches=caches, ref_alpha=args.ref_alpha,
                           soft_tau=args.soft_tau, dispersion=args.dispersion,
                           gen_k=args.num_nearest_neighbor)
            mt = evaluate(labels, sc)
            report["regimes"][regime]["ablation"][T] = {
                "auc": mt["auc"], "tpr@fpr=0.01": mt["tpr@fpr=0.01"]}
            print(f"  T={T:>3}: AUC={mt['auc']:.4f}  "
                  f"TPR@1%FPR={mt['tpr@fpr=0.01']:.4f}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
