"""Adds the DOMIAS baseline (density-ratio, KDE) to the existing FBB / histogram-LLR
/ selection-only comparison in ``run_baseline_comparison.py`` (imported, not
modified) -- same 4 datasets, same shared untimed embedding pass, same eps~10
audit config. Only the DOMIAS fit+evaluate call is timed here; FBB/histogram-LLR/
selection-only numbers are unchanged from that script and simply reproduced from
its own run for the combined table (see this module's report).
"""

import time
import numpy as np

from attacks.run_baseline_comparison import RUNS, BASE
from attacks.audit_set import build_audit_set, reference_rows_by_class
from attacks.reconstruct import reconstruct
from attacks.domias_baseline import domias_scores
from attacks.evaluate import evaluate
from attacks.tabular_embedding import load_private, make_embed_fn
from pe.constant.data import LABEL_ID_COLUMN_NAME


def run_one(cfg):
    slug = cfg["slug"]
    train_csv = f"{BASE}/{slug}/{slug}_train.csv"
    test_csv = f"{BASE}/{slug}/{slug}_test.csv"
    metadata = f"{BASE}/{slug}/{slug}_metadata.json"

    # --- shared, UNTIMED embedding / data-loading pass (identical to
    # run_baseline_comparison.py's run_one) -------------------------------------
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
    # -----------------------------------------------------------------------

    domias_raw = np.full(len(classes), np.nan)
    modes = {}
    t0 = time.time()
    for cls in np.unique(classes_arr):
        iters = by_class_hist.get(int(cls))
        ref = ref_by_class.get(int(cls))
        if not iters or ref is None or len(ref) == 0:
            continue
        synth_set = iters[-1]["cell_features"]  # last released round, same as FBB
        mask = classes_arr == cls
        scores, meta = domias_scores(query_emb[mask], synth_set, ref)
        domias_raw[mask] = scores
        modes[int(cls)] = meta
    t_domias = time.time() - t0

    finite = domias_raw[np.isfinite(domias_raw)]
    domias_raw[np.isnan(domias_raw)] = (finite.min() - 1.0) if finite.size else 0.0
    auc = evaluate(labels, domias_raw)["auc"]
    return auc, t_domias, modes


def main():
    print(f"{'dataset':<24}{'AUC':>8}{'core_time_s':>14}   p_R modes by class")
    for cfg in RUNS:
        auc, t, modes = run_one(cfg)
        pr_modes = {c: m["p_R_mode"] for c, m in modes.items()}
        pg_modes = {c: m["p_G_mode"] for c, m in modes.items()}
        print(f"{cfg['slug']:<24}{auc:>8.4f}{t:>14.4f}   p_R={pr_modes}  p_G={pg_modes}")


if __name__ == "__main__":
    main()
