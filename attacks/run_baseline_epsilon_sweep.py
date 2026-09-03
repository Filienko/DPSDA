"""Full epsilon sweep for the two density-only baselines (GAN-Leaks/FBB, DOMIAS)
across all 4 tabular datasets. Same methodology as run_baseline_comparison.py /
run_domias_timing.py (last released round = "generator output", same audit set),
just looped over all 7 privacy budgets per dataset. AUC only, no timing (the
timing comparison already exists at eps~10).

Per dataset, the private train/test CSVs (and therefore the audit set + reference
embeddings) are IDENTICAL across epsilon variants -- only the generation run
(and so the released synthetic pool) differs -- so embedding is done ONCE per
dataset and reused across all 7 checkpoints.
"""

import numpy as np

from attacks.run_baseline_comparison import BASE
from attacks.audit_set import build_audit_set, reference_rows_by_class
from attacks.reconstruct import reconstruct
from attacks.gan_leaks_fbb import fbb_scores
from attacks.domias_baseline import domias_scores
from attacks.evaluate import evaluate
from attacks.tabular_embedding import load_private, make_embed_fn
from pe.constant.data import LABEL_ID_COLUMN_NAME

EPS_SUFFIXES = [
    ("inf", "_nonoise"),
    ("100", "_eps100"),
    ("10", ""),
    ("5", "_eps5"),
    ("1", "_eps1"),
    ("0.5", "_eps0p5"),
    ("0.25", "_eps0p25"),
]
DATASETS = ["adult", "artificial-characters", "breast-cancer", "person-activity"]
N_MEMBERS = 500
N_NONMEMBERS = 500


def run_dataset(slug):
    train_csv = f"{BASE}/{slug}/{slug}_train.csv"
    test_csv = f"{BASE}/{slug}/{slug}_test.csv"
    metadata = f"{BASE}/{slug}/{slug}_metadata.json"

    priv_data, info = load_private(train_csv, metadata)
    embed_fn = make_embed_fn(priv_data, info)
    n_private_by_class = {int(k): int(v) for k, v in
                          priv_data.data_frame[LABEL_ID_COLUMN_NAME].value_counts().items()}
    rows, classes, labels, _ = build_audit_set(
        train_csv, test_csv, metadata, n_members=N_MEMBERS,
        n_nonmembers=N_NONMEMBERS, seed=0, ref_holdout_frac=0.5)
    ref_rows = reference_rows_by_class(test_csv, metadata, max_per_class=2000,
                                       seed=0, ref_holdout_frac=0.5)
    ref_by_class = {cls: embed_fn(r) for cls, r in ref_rows.items()}
    query_emb = embed_fn(rows)
    classes_arr = np.asarray(classes)

    out = {}
    for eps_label, suffix in EPS_SUFFIXES:
        checkpoint = f"results/tabular/{slug}_composite_population{suffix}/checkpoint"
        by_class = reconstruct(checkpoint, embed_fn, ref_by_class,
                               n_private_by_class, start_t=1, use_clean=False)

        fbb_raw = np.full(len(classes), np.nan)
        domias_raw = np.full(len(classes), np.nan)
        for cls in np.unique(classes_arr):
            iters = by_class.get(int(cls))
            ref = ref_by_class.get(int(cls))
            if not iters:
                continue
            synth_set = iters[-1]["cell_features"]
            mask = classes_arr == cls
            fbb_raw[mask] = fbb_scores(query_emb[mask], synth_set)
            if ref is not None and len(ref) > 0:
                domias_raw[mask], _meta = domias_scores(query_emb[mask], synth_set, ref)

        f = fbb_raw[np.isfinite(fbb_raw)]
        fbb_raw[np.isnan(fbb_raw)] = (f.min() - 1.0) if f.size else 0.0
        d = domias_raw[np.isfinite(domias_raw)]
        domias_raw[np.isnan(domias_raw)] = (d.min() - 1.0) if d.size else 0.0

        out[eps_label] = {
            "fbb_auc": evaluate(labels, fbb_raw)["auc"],
            "domias_auc": evaluate(labels, domias_raw)["auc"],
            "n_rounds": max((len(v) for v in by_class.values()), default=0),
        }
        print(f"  {slug:<24} eps={eps_label:<6} "
             f"FBB={out[eps_label]['fbb_auc']:.4f}  DOMIAS={out[eps_label]['domias_auc']:.4f}  "
             f"(rounds={out[eps_label]['n_rounds']})", flush=True)
    return out


def main():
    results = {}
    for slug in DATASETS:
        print(f"=== {slug} ===")
        results[slug] = run_dataset(slug)

    print("\n\n=== SUMMARY (AUC) ===")
    header = f"{'dataset':<24}{'eps':<8}{'GAN-Leaks(FBB)':>16}{'DOMIAS':>10}"
    print(header)
    for slug in DATASETS:
        for eps_label, _ in EPS_SUFFIXES:
            r = results[slug][eps_label]
            print(f"{slug:<24}{eps_label:<8}{r['fbb_auc']:>16.4f}{r['domias_auc']:>10.4f}")


if __name__ == "__main__":
    main()
