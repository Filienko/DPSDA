"""Ad-hoc driver: run selection_mia against a real saved tabular-PE checkpoint and
compare to the published histogram-LLR AUC. Mirrors run_mia.py's plumbing."""
import argparse
import numpy as np

from attacks.audit_set import build_audit_set, reference_rows_by_class
from attacks.selection_mia import reconstruct_selection, score_records_selection
from attacks.evaluate import evaluate, format_report
from attacks.tabular_embedding import load_private, make_embed_fn

BASE = "https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/heads/main/tabular/real"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_folder", required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--noise_multiplier", type=float, required=True)
    p.add_argument("--n_members", type=int, default=500)
    p.add_argument("--n_nonmembers", type=int, default=500)
    p.add_argument("--ref_holdout_frac", type=float, default=0.5)
    args = p.parse_args()

    train_csv = f"{BASE}/{args.slug}/{args.slug}_train.csv"
    test_csv = f"{BASE}/{args.slug}/{args.slug}_test.csv"
    metadata = f"{BASE}/{args.slug}/{args.slug}_metadata.json"

    print("Loading private data / embedding model...")
    priv_data, info = load_private(train_csv, metadata)
    embed_fn = make_embed_fn(priv_data, info)
    from pe.constant.data import LABEL_ID_COLUMN_NAME
    n_private_by_class = {int(k): int(v) for k, v in
                          priv_data.data_frame[LABEL_ID_COLUMN_NAME].value_counts().items()}
    print(f"  n_private per class: {n_private_by_class}")

    print("Building audit set...")
    rows, classes, labels, sizes = build_audit_set(
        train_csv, test_csv, metadata, n_members=args.n_members,
        n_nonmembers=args.n_nonmembers, seed=0, ref_holdout_frac=args.ref_holdout_frac)
    print(f"  members={int(labels.sum())} non-members={int((labels == 0).sum())}")

    print("Embedding reference (null model) data...")
    ref_rows = reference_rows_by_class(test_csv, metadata, max_per_class=2000,
                                       seed=0, ref_holdout_frac=args.ref_holdout_frac)
    ref_by_class = {cls: embed_fn(r) for cls, r in ref_rows.items()}
    print(f"  D_ref size per class: { {c: len(r) for c, r in ref_rows.items()} }")

    print("Embedding audit records...")
    query_emb = embed_fn(rows)

    print("Reconstructing selection-only rounds (no counts read)...")
    by_class = reconstruct_selection(args.checkpoint_folder, embed_fn, ref_by_class,
                                     n_private_by_class, start_t=1)
    n_it = {c: len(v) for c, v in by_class.items()}
    print(f"  aligned selection-rounds per class: {n_it}")

    sigma = args.noise_multiplier
    classes_arr = np.asarray(classes)
    scores = np.full(len(classes), np.nan)
    caches = {}
    for cls in np.unique(classes_arr):
        iters = by_class.get(int(cls))
        if not iters:
            continue
        mask = classes_arr == cls
        cache = caches.setdefault(int(cls), {})
        scores[mask] = score_records_selection(query_emb[mask], iters, mode="L2",
                                               sigma=sigma, dispersion=1.8,
                                               ref_alpha=0.05, soft_tau=0.02,
                                               occupancy_cache=cache)
    finite = scores[np.isfinite(scores)]
    scores[np.isnan(scores)] = (finite.min() - 1.0) if finite.size else 0.0

    m = evaluate(labels, scores)
    print("\n" + format_report(m, title=f"REAL DATA selection-only: {args.slug} (sigma={sigma})"))


if __name__ == "__main__":
    main()
