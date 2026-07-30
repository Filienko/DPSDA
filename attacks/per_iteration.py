"""Per-iteration view: as PE refines the synthetic data toward the private
distribution, how does membership leakage evolve?

For each released PE round t (the keep_selected rounds that publish a histogram)
we measure, on a MATCHED setup (genuine leakage):
  * synth<->private distance: two-sample RBF-MMD between that round's released
    synthetic cells and the private members (smaller = the synthetic distribution
    approximates private better),
  * downstream utility: the run's logged TabICL test accuracy at that iteration,
  * per-round attack AUC: members vs non-members scored from ONLY round t's
    histogram,
  * cumulative attack AUC: aggregating rounds up to t (what the full attack uses).

The key question is the relationship between the first and the third: does the
attack leak more precisely when the synthetic data has converged onto private?
Writes a summary JSON and a 2x2 figure.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attacks.audit_set import build_audit_set, reference_rows_by_class
from attacks.reconstruct import reconstruct, discover_iterations
from attacks.histogram_mia import score_records, sample_mmd
from attacks.tabular_embedding import load_private, make_embed_fn
from pe.constant.data import LABEL_ID_COLUMN_NAME, CLEAN_HISTOGRAM_COLUMN_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "example", "tabular", "dist_shift", "data")
META = os.path.join(DATA, "metadata.json")
TRAIN = os.path.join(DATA, "train.csv")
TEST = os.path.join(DATA, "test.csv")
RUNDIR = os.path.join(HERE, "..", "results", "tabular",
                      "artificial-characters_composite_population")
CKPT = os.path.join(RUNDIR, "checkpoint")
NM = 1.8385747404944874
SEED = 0
OUT = os.path.join(HERE, "..", "results", "tabular", "dist_shift", "analysis")

C = {"mmd": "#0072B2", "util": "#009E73", "per": "#D55E00", "cum": "#0072B2"}
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def usable_rounds():
    ts = []
    for t in discover_iterations(CKPT):
        if t < 1:
            continue
        df = pd.read_pickle(os.path.join(CKPT, f"{t:09d}", "data_frame.pkl"))
        if CLEAN_HISTOGRAM_COLUMN_NAME in df.columns and \
           df[CLEAN_HISTOGRAM_COLUMN_NAME].notna().any():
            ts.append(t)
    return ts


def score_auc(iters_by_class, query_emb, classes, labels):
    classes = np.asarray(classes)
    scores = np.full(len(classes), np.nan)
    for c in np.unique(classes):
        iters = iters_by_class.get(int(c))
        if not iters:
            continue
        mask = classes == c
        scores[mask] = score_records(query_emb[mask], iters, mode="L2",
                                     regime="noised", sigma=NM, censored=False)
    fin = scores[np.isfinite(scores)]
    scores[~np.isfinite(scores)] = (fin.min() - 1.0) if fin.size else 0.0
    return float(roc_auc_score(labels, scores))


def utility_by_iter():
    hits = glob.glob(os.path.join(RUNDIR, "tabular_classifier_tabicl_filter_*_test_acc.csv"))
    if not hits:
        return {}
    df = pd.read_csv(hits[0], header=None, names=["it", "acc"])
    return {int(r.it): float(r.acc) for r in df.itertuples()}


def main():
    priv, info = load_private(TRAIN, META)
    embed_fn = make_embed_fn(priv, info)
    n_priv = {int(k): int(v) for k, v in
              priv.data_frame[LABEL_ID_COLUMN_NAME].value_counts().items()}
    priv_emb = embed_fn([list(r) for r in
                         priv.data_frame["PE.TABULAR"].tolist()])

    rows, classes, labels, _ = build_audit_set(TRAIN, TEST, META, n_members=1500,
                                               n_nonmembers=1500, seed=SEED,
                                               ref_holdout_frac=0.5)
    classes = np.asarray(classes); labels = np.asarray(labels)
    ref_rows = reference_rows_by_class(TEST, META, max_per_class=2000, seed=SEED,
                                       ref_holdout_frac=0.5)
    ref_by_class = {c: embed_fn(r) for c, r in ref_rows.items()}
    query_emb = embed_fn(rows)

    by_class = reconstruct(CKPT, embed_fn, ref_by_class, n_priv, start_t=1,
                           use_clean=False)
    ts = usable_rounds()
    R = min(len(ts), min(len(v) for v in by_class.values()))
    ts = ts[:R]
    util = utility_by_iter()

    recs = []
    for i, t in enumerate(ts):
        single = {c: [by_class[c][i]] for c in by_class if len(by_class[c]) > i}
        cumul = {c: by_class[c][:i + 1] for c in by_class}
        # released synthetic cells this round, pooled across classes.
        cells = np.concatenate([by_class[c][i]["cell_features"] for c in single])
        smmd = sample_mmd(cells, priv_emb, seed=SEED)
        recs.append({
            "iteration": int(t),
            "synth_priv_mmd": round(smmd, 4),
            "utility_acc": util.get(int(t)),
            "per_round_auc": round(score_auc(single, query_emb, classes, labels), 4),
            "cumulative_auc": round(score_auc(cumul, query_emb, classes, labels), 4),
        })
        print(f"t={t:3d}  synth↔priv MMD={smmd:.4f}  util={recs[-1]['utility_acc']}"
              f"  per-round AUC={recs[-1]['per_round_auc']:.4f}"
              f"  cumulative AUC={recs[-1]['cumulative_auc']:.4f}")

    os.makedirs(OUT, exist_ok=True)
    json.dump({"dataset": "artificial-characters", "epsilon": 10, "rows": recs},
              open(os.path.join(OUT, "per_iteration.json"), "w"), indent=2)

    it = [r["iteration"] for r in recs]
    mmd = [r["synth_priv_mmd"] for r in recs]
    ua = [r["utility_acc"] for r in recs]
    pa = [r["per_round_auc"] for r in recs]
    ca = [r["cumulative_auc"] for r in recs]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(it, mmd, "o-", color=C["mmd"], lw=2)
    ax[0, 0].set_xlabel("PE iteration"); ax[0, 0].set_ylabel("synth ↔ private MMD")
    ax[0, 0].set_title("Synthetic distribution converges toward private")

    if any(v is not None for v in ua):
        ax[0, 1].plot(it, ua, "s-", color=C["util"], lw=2)
    ax[0, 1].set_xlabel("PE iteration"); ax[0, 1].set_ylabel("TabICL test accuracy (%)")
    ax[0, 1].set_title("Utility rises over iterations")

    ax[1, 0].plot(it, pa, "o-", color=C["per"], lw=2, label="per-round AUC")
    ax[1, 0].plot(it, ca, "s-", color=C["cum"], lw=2, label="cumulative AUC")
    ax[1, 0].axhline(0.5, color="silver", lw=1, alpha=0.7)
    ax[1, 0].set_xlabel("PE iteration"); ax[1, 0].set_ylabel("MIA ROC AUC")
    ax[1, 0].set_title("Membership leakage over iterations"); ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(mmd, pa, "o-", color=C["per"], lw=2)
    for x, y, t in zip(mmd, pa, it):
        ax[1, 1].annotate(str(t), (x, y), fontsize=7, xytext=(3, 3),
                          textcoords="offset points")
    ax[1, 1].set_xlabel("synth ↔ private MMD  (smaller = better approximation)")
    ax[1, 1].set_ylabel("per-round MIA AUC")
    ax[1, 1].set_title("Leakage vs how well synthetic approximates private")
    fig.suptitle("Per-iteration: synthetic→private convergence and membership "
                 "leakage (artificial-characters, ε=10, matched)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "fig_per_iteration.png"), dpi=150)
    plt.close(fig)
    print(f"\nwrote per_iteration.json and fig_per_iteration.png under "
          f"{os.path.relpath(OUT, HERE)}/")


if __name__ == "__main__":
    main()
