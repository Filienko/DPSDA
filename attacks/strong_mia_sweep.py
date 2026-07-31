"""Score every finished PE run with the baseline and each strong-attack channel set.

Reuses whatever ``results/tabular/*/checkpoint`` directories already exist, so it
only pays for scoring, not for regenerating PE runs. Loads the audit set and
embeds the candidate pools once per run and reuses them across channel
configurations, which is why this is much faster than looping the
``attacks.strong_mia`` CLI.

Writes a single ``attacks/results_strong_mia.json`` so the numbers in
``running_summary.md`` are generated rather than transcribed.

    python -m attacks.strong_mia_sweep --runs results/tabular/ac_inout ...
"""

import argparse
import json
import os
import sys

import numpy as np

from pe.constant.data import LABEL_ID_COLUMN_NAME, TABULAR_DATA_COLUMN_NAME

from attacks.strong_mia import (
    reconstruct_full, score_all, self_test, _scrape_noise_multiplier, _group_auc,
)
from attacks.reconstruct import reconstruct as reconstruct_baseline
from attacks.histogram_mia import score_records
from attacks.tabular_embedding import load_private, make_embed_fn
from attacks.audit_set import build_audit_set, reference_rows_by_class
from attacks.evaluate import evaluate

CONFIGS = [
    ("count",), ("count", "censored"), ("count", "censored", "mult"),
    ("traj",), ("count", "censored", "mult", "traj"),
]


def outlier_groups(labels, classes, query_emb, priv, embed_fn):
    """Within-class kNN-outlierness tertiles, as in ``outlier_disparity.py:136-138``.

    Depends only on the records, not on any score, so it is computed once per run
    and reused across channel configurations -- recomputing it per config made the
    sweep six times slower than the scoring it was measuring.
    """
    from attacks.outlier_disparity import knn_outlierness
    train_emb = embed_fn([list(r) for r in priv.data_frame[TABULAR_DATA_COLUMN_NAME]])
    train_cls = priv.data_frame[LABEL_ID_COLUMN_NAME].to_numpy()
    mem = labels == 1
    rank = np.zeros(len(labels))
    rank[mem] = knn_outlierness(query_emb[mem], classes[mem], train_emb, train_cls)
    edges = np.quantile(rank[mem], [1 / 3, 2 / 3])
    return np.where(rank <= edges[0], "inlier",
                    np.where(rank <= edges[1], "mid", "outlier"))


def _per_group(labels, scores, grp):
    return {g: _group_auc(labels, scores, grp, g) for g in ("inlier", "mid", "outlier")}


def score_run(run_dir, metadata, members_csv="", nonmembers_csv="", seed=0):
    ckpt = os.path.join(run_dir, "checkpoint")
    m_csv = members_csv or os.path.join(run_dir, "audit_members.csv")
    n_csv = nonmembers_csv or os.path.join(run_dir, "audit_nonmembers.csv")
    nm = _scrape_noise_multiplier(run_dir)
    sigma = float(nm)

    priv, info = load_private(m_csv, metadata)
    embed_fn = make_embed_fn(priv, info)
    npc = {int(k): int(v) for k, v in
           priv.data_frame[LABEL_ID_COLUMN_NAME].value_counts().items()}

    n_mem = len(priv.data_frame)
    import pandas as pd
    n_non = len(pd.read_csv(n_csv))
    rows, classes, labels, sizes = build_audit_set(
        m_csv, n_csv, metadata, n_members=n_mem, n_nonmembers=n_non,
        seed=seed, ref_holdout_frac=0.5)
    ref_rows = reference_rows_by_class(n_csv, metadata, max_per_class=2000,
                                       seed=seed, ref_holdout_frac=0.5)
    ref = {c: embed_fn(r) for c, r in ref_rows.items()}
    qe = embed_fn(rows)

    out = {"run_dir": run_dir, "noise_multiplier": sigma,
           "n_members": int(labels.sum()), "n_nonmembers": int((labels == 0).sum()),
           "data_sizes": sizes, "configs": {}}

    priv_rows = {int(c): [list(r) for r in s[TABULAR_DATA_COLUMN_NAME]]
                 for c, s in priv.data_frame.groupby(LABEL_ID_COLUMN_NAME)}
    out["self_test"] = self_test(ckpt, embed_fn, priv_rows, npc, start_t=1)
    grp = outlier_groups(labels, classes, qe, priv, embed_fn)

    # Baseline: survivor-only pool, histogram_mia.score_records with run_mia.py's
    # tuned defaults.
    base_bc = reconstruct_baseline(ckpt, embed_fn, ref, npc, start_t=1)
    base = np.full(len(classes), np.nan)
    for cls in np.unique(classes):
        it = base_bc.get(int(cls))
        if not it:
            continue
        mask = classes == cls
        base[mask] = score_records(qe[mask], it, mode="L2", regime="noised",
                                   sigma=sigma, censored=False, ref_alpha=0.05,
                                   soft_tau=0.02, dispersion=1.8)
    fin = base[np.isfinite(base)]
    base[~np.isfinite(base)] = (fin.min() - 1.0) if fin.size else 0.0
    out["configs"]["baseline"] = dict(evaluate(labels, base),
                                      per_group=_per_group(labels, base, grp))

    bc = reconstruct_full(ckpt, embed_fn, ref, npc, start_t=1)
    r0 = bc[sorted(bc)[0]]
    out["pool"] = {
        "rounds": len(r0),
        "sample_rounds": sum(1 for i in r0 if i["mode"] == "sample"),
        "cells_per_class": int(r0[-1]["cell_features"].shape[0]),
        "mean_censored_fraction": float(np.mean(
            [1 - i["observed"].mean() for i in r0 if i["mode"] == "rank"])),
    }
    for ch in CONFIGS:
        s = score_all(bc, qe, classes, ref, sigma, ch)
        out["configs"][",".join(ch)] = dict(evaluate(labels, s),
                                            per_group=_per_group(labels, s, grp))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--metadata", nargs="+", required=True,
                   help="one metadata path per run, or a single shared one")
    p.add_argument("--out", default="attacks/results_strong_mia.json")
    args = p.parse_args(argv)

    metas = (args.metadata * len(args.runs)) if len(args.metadata) == 1 else args.metadata
    results = []
    for run, meta in zip(args.runs, metas):
        if not os.path.isdir(os.path.join(run, "checkpoint")):
            print(f"skip (no checkpoint): {run}")
            continue
        print(f"scoring {run} ...")
        try:
            r = score_run(run, meta)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        results.append(r)
        b = r["configs"]["baseline"]["auc"]
        best_k = max((k for k in r["configs"] if k != "baseline"),
                     key=lambda k: r["configs"][k]["auc"])
        print(f"  baseline={b:.4f}  best={best_k} {r['configs'][best_k]['auc']:.4f} "
              f"(delta {r['configs'][best_k]['auc'] - b:+.4f})  "
              f"self_test full_pool={r['self_test']['full_pool']['pass']}")

    with open(args.out, "w") as fh:
        json.dump({"runs": results}, fh, indent=2, default=str)
    print(f"\nwrote {args.out} ({len(results)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
