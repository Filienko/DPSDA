"""Validate attacks/strong_mia.py against a faithful simulation of the PE loop.

Run with ``python -m attacks.strong_mia_simtest``. Needs no data and no PE run: it
writes its own checkpoint pickles to a temp dir. Exit code 0 = all checks pass.

Reproduces pe/runner/pe.py + pe/population/pe_population.py + pe/dp/gaussian.py
exactly (per-class histogram, uncensored Gaussian noise, sample-then-rank
composite population, keep_selected, parent pointers, global index reset) and
writes real checkpoint pickles. Then checks:

  1. reconstruct_full recovers the candidate pool, counts, censoring mask, child
     multiplicity and parent chain EXACTLY against recorded ground truth.
  2. self_test passes on the full pool and fails on the survivor-only pool.
  3. The attack separates members from non-members, and beats the survivor-only
     baseline.
"""
import os, sys, shutil
import numpy as np
import pandas as pd

import tempfile

from pe.constant.data import (
    LABEL_ID_COLUMN_NAME as LBL, TABULAR_DATA_COLUMN_NAME as TAB,
    CLEAN_HISTOGRAM_COLUMN_NAME as CLEAN, DP_HISTOGRAM_COLUMN_NAME as DP,
    PARENT_SYN_DATA_INDEX_COLUMN_NAME as PAR, FROM_LAST_FLAG_COLUMN_NAME as FLAG,
    VARIATION_API_FOLD_ID_COLUMN_NAME as FOLD,
)
from attacks.strong_mia import (
    reconstruct_full, self_test, score_all, score_llr_channels,
    llr_observed, llr_censored, llr_multiplicity, _expected_relu_sum,
)
from attacks.histogram_mia import nearest_cell, llr_noised

OUT = os.path.join(tempfile.gettempdir(), "strong_mia_simrun")
CKPT = os.path.join(OUT, "checkpoint")


def embed_fn(rows):
    rows = list(rows)
    if not rows:
        return np.empty((0, DIM), dtype=np.float32)
    return np.asarray([list(r) for r in rows], dtype=np.float32)


DIM = 4
N_CLASS = 2
N_PRIV = 90          # per class
N_SEL = 60           # num_samples per class per round
T_SAMPLE = 4         # rounds 1..4 use selection_mode="sample", keep_selected=False
T_TOTAL = 12         # rounds 5..12 use selection_mode="rank",  keep_selected=True
SIGMA = 1.0
MUT = 0.25


def simulate(seed=0):
    """Mirror the real PE loop and write checkpoint/{t:09d}/data_frame.pkl."""
    rng = np.random.default_rng(seed)
    shutil.rmtree(OUT, ignore_errors=True)

    # Private data: two well separated class clusters.
    priv = {c: rng.normal(loc=c * 3.0, scale=1.0, size=(N_PRIV, DIM)) for c in range(N_CLASS)}

    def save(t, df):
        d = os.path.join(CKPT, f"{t:09d}")
        os.makedirs(d, exist_ok=True)
        df.to_pickle(os.path.join(d, "data_frame.pkl"))

    # Iteration 0: initial random population, no parents.
    frames = []
    for c in range(N_CLASS):
        f = rng.normal(loc=c * 3.0, scale=1.5, size=(N_SEL, DIM))
        frames.append(pd.DataFrame({TAB: [list(r) for r in f], LBL: c}))
    cur = pd.concat(frames).reset_index(drop=True)
    save(0, cur)

    truth = {}   # (t, cls) -> dict of ground truth
    for t in range(1, T_TOTAL + 1):
        mode = "sample" if t <= T_SAMPLE else "rank"
        keep_selected = (mode == "rank")
        n_fold = 1 if mode == "sample" else 3
        out_frames = []
        for c in range(N_CLASS):
            pool = cur[cur[LBL] == c]
            cand_global = pool.index.to_numpy()
            cellF = np.asarray([list(r) for r in pool[TAB]], dtype=np.float32)

            # --- DP NN histogram (pe/histogram/nearest_neighbors.py) ---
            nn = nearest_cell(priv[c], cellF, mode="L2", k=1)
            clean = np.bincount(np.ravel(nn), minlength=len(pool)).astype(np.float64)
            # --- Gaussian mechanism (pe/dp/gaussian.py:175), uncensored ---
            dp = clean + rng.normal(scale=SIGMA, size=len(pool))

            # --- selection (pe/population/pe_population.py:111-136) ---
            if mode == "sample":
                post = np.clip(dp, 0.0, None)          # histogram_threshold=0
                p = post / post.sum()
                sel = rng.choice(len(pool), size=N_SEL, p=p)
            else:
                post = dp                              # histogram_threshold=None
                sel = np.argsort(post)[::-1][:N_SEL]
            parent_global = cand_global[sel]

            selected = pd.DataFrame({
                TAB: [list(r) for r in cellF[sel]], LBL: c,
                PAR: parent_global, FLAG: 1, FOLD: -1,
                CLEAN: clean[sel], DP: dp[sel],
            })
            kids = []
            for fold in range(n_fold):
                mutated = cellF[sel] + rng.uniform(-MUT, MUT, size=(N_SEL, DIM))
                kids.append(pd.DataFrame({
                    TAB: [list(r) for r in mutated], LBL: c,
                    PAR: parent_global, FLAG: 0, FOLD: fold,
                }))
            # concat order matches pe_population.py:179: children, then survivors
            out_frames.append(pd.concat(kids + ([selected] if keep_selected else [])))

            truth[(t, c)] = {
                "cand_global": cand_global, "clean": clean, "dp": dp,
                "sel": np.sort(np.unique(sel)) if mode == "sample" else np.sort(sel),
                "sel_raw": sel, "mode": mode, "n_cand": len(pool),
                "n_draws": N_SEL * n_fold,
            }
        cur = pd.concat(out_frames).reset_index(drop=True)
        save(t, cur)
    return priv, truth


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")
    return bool(ok)


def main():
    print("Simulating PE run...")
    priv, truth = simulate()
    ok_all = True

    # ---------------- 1. reconstruction fidelity ----------------
    print("\n1. reconstruct_full vs ground truth")
    n_priv_by_class = {c: N_PRIV for c in range(N_CLASS)}
    by_class = reconstruct_full(CKPT, embed_fn, {}, n_priv_by_class, start_t=1)

    n_rounds = {c: len(v) for c, v in by_class.items()}
    ok_all &= check("all rounds recovered", all(v == T_TOTAL for v in n_rounds.values()),
                    f"{n_rounds} (expected {T_TOTAL} each)")

    bad_counts = bad_obs = bad_mult = bad_par = bad_pool = 0
    for c, iters in by_class.items():
        for it in iters:
            g = truth[(it["t"], c)]
            if it["cell_features"].shape[0] != g["n_cand"]:
                bad_pool += 1
                continue
            if it["mode"] != g["mode"]:
                bad_obs += 1
            if g["mode"] == "rank":
                exp_obs = np.zeros(g["n_cand"], dtype=bool)
                exp_obs[g["sel"]] = True
                if not np.array_equal(it["observed"], exp_obs):
                    bad_obs += 1
                if not np.allclose(it["counts"][exp_obs], g["dp"][g["sel"]]):
                    bad_counts += 1
            exp_mult = np.bincount(g["sel_raw"], minlength=g["n_cand"])
            exp_mult = exp_mult * (1 if g["mode"] == "sample" else 3)
            if not np.array_equal(it["child_mult"], exp_mult):
                bad_mult += 1
            # parent_pos must point at the previous round's pool position
            if it["t"] > 1:
                prev = truth[(it["t"] - 1, c)]
                pos = {int(gi): i for i, gi in enumerate(prev["cand_global"])}
                # candidates of round t are the rows written at ckpt t-1, whose
                # parents were the survivors selected in round t-1
                exp = np.array([pos.get(int(v), -1) for v in prev["cand_global"][prev["sel_raw"]]])
                got = it["parent_pos"]
                # every candidate's parent_pos must be a valid position, and the
                # multiset of parents must match the round t-1 selection
                if (got < 0).any() or not np.array_equal(
                        np.sort(np.bincount(got, minlength=prev["n_cand"])),
                        np.sort(np.bincount(prev["sel_raw"], minlength=prev["n_cand"])
                                * (1 if prev["mode"] == "sample" else 4))):
                    bad_par += 1
                del exp

    ok_all &= check("pool size == candidate count", bad_pool == 0, f"{bad_pool} bad")
    ok_all &= check("observed mask == selected set", bad_obs == 0, f"{bad_obs} bad")
    ok_all &= check("released counts exact", bad_counts == 0, f"{bad_counts} bad")
    ok_all &= check("child multiplicity exact", bad_mult == 0, f"{bad_mult} bad")
    ok_all &= check("parent chain valid", bad_par == 0, f"{bad_par} bad")

    r0 = by_class[0]
    cens = np.mean([1 - it["observed"].mean() for it in r0 if it["mode"] == "rank"])
    n_samp = sum(1 for it in r0 if it["mode"] == "sample")
    print(f"      pool={r0[-1]['cell_features'].shape[0]} cells, "
          f"censored fraction={cens:.2f}, sample-mode rounds recovered={n_samp}")
    ok_all &= check("sample-mode rounds recovered (baseline drops these)", n_samp == T_SAMPLE)

    # ---------------- 2. self test ----------------
    print("\n2. self_test (clean-histogram reconstruction gate)")
    priv_rows = {c: [list(r) for r in priv[c]] for c in range(N_CLASS)}
    rep = self_test(CKPT, embed_fn, priv_rows, n_priv_by_class, start_t=1)
    print(f"      full_pool      : {rep['full_pool']}")
    print(f"      survivors_only : {rep['survivors_only']}")
    ok_all &= check("full pool reproduces PE.CLEAN_HISTOGRAM exactly",
                    rep["full_pool"]["pass"])
    ok_all &= check("survivor-only pool FAILS the same check (the defect)",
                    not rep["survivors_only"]["pass"],
                    f"max count error {rep['survivors_only']['max_abs_count_error']:.0f}")

    # ---------------- 3. LLR sanity ----------------
    print("\n3. LLR primitives")
    a = llr_observed([5.0], [3.0], [3.0], 1.0)[0]
    b = llr_noised(5.0, 3.0, 3.0, sigma=1.0, censored=False)
    ok_all &= check("llr_observed matches histogram_mia.llr_noised",
                    abs(a - b) < 1e-6, f"{a:.6f} vs {b:.6f}")
    hi = llr_observed([8.0], [2.0], [2.0], 1.0)[0]
    lo = llr_observed([0.0], [2.0], [2.0], 1.0)[0]
    ok_all &= check("higher count -> higher LLR", hi > lo, f"{hi:.3f} > {lo:.3f}")
    cz = llr_censored(4.0, [2.0], [2.0], 1.0)[0]
    ok_all &= check("censored cell is evidence AGAINST membership", cz < 0, f"{cz:.4f}")
    rs = _expected_relu_sum(np.full(50, 2.0), 1.0)
    m_hi = llr_multiplicity([6], 100, rs, [2.0], [2.0], 1.0)[0]
    m_lo = llr_multiplicity([0], 100, rs, [2.0], [2.0], 1.0)[0]
    ok_all &= check("more children -> higher multiplicity LLR", m_hi > m_lo,
                    f"{m_hi:.3f} > {m_lo:.3f}")

    # ---------------- 4. attack power ----------------
    print("\n4. attack power (members vs fresh non-members)")
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(123)
    rows, classes, labels = [], [], []
    for c in range(N_CLASS):
        non = rng.normal(loc=c * 3.0, scale=1.0, size=(N_PRIV, DIM))
        rows += [list(r) for r in priv[c]] + [list(r) for r in non]
        classes += [c] * (2 * N_PRIV)
        labels += [1] * N_PRIV + [0] * N_PRIV
    classes = np.asarray(classes); labels = np.asarray(labels)
    qe = embed_fn(rows)
    ref = {c: embed_fn([list(r) for r in
                        rng.normal(loc=c * 3.0, scale=1.0, size=(400, DIM))])
           for c in range(N_CLASS)}
    by_class = reconstruct_full(CKPT, embed_fn, ref, n_priv_by_class, start_t=1)
    surv_only = reconstruct_full(CKPT, embed_fn, ref, n_priv_by_class, start_t=1,
                                 survivors_only=True)

    def auc(bc, ch, calib="perexample"):
        s = score_all(bc, qe, classes, ref, SIGMA, ch, calib=calib)
        return float(roc_auc_score(labels, s))

    base = auc(surv_only, ("count",))
    res = {
        "survivor-only, count            ": base,
        "full pool,     count            ": auc(by_class, ("count",)),
        "full pool,     count+censored   ": auc(by_class, ("count", "censored")),
        "full pool,     +mult (rounds1-4)": auc(by_class, ("count", "censored", "mult")),
        "full pool,     +traj (all)      ": auc(by_class, ("count", "censored", "mult", "traj")),
    }
    for k, v in res.items():
        print(f"      {k}  AUC = {v:.4f}")
    ok_all &= check("baseline reconstruction gives signal", base > 0.5)
    ok_all &= check("full pool beats survivor-only",
                    res["full pool,     count            "] > base)
    ok_all &= check("all channels beat survivor-only baseline",
                    res["full pool,     +traj (all)      "] > base)

    # multiplicity channel alone, on the rounds the baseline cannot see at all
    only_sample = {c: [it for it in v if it["mode"] == "sample"]
                   for c, v in by_class.items()}
    m_auc = auc(only_sample, ("mult",))
    print(f"      multiplicity channel ALONE on rounds 1-{T_SAMPLE}: AUC = {m_auc:.4f}")
    ok_all &= check("discarded rounds 1-4 carry real signal", m_auc > 0.5)

    only_rank = {c: [it for it in v if it["mode"] == "rank"] for c, v in by_class.items()}
    t_auc = auc(only_rank, ("traj",))
    print(f"      trajectory channel ALONE:                 AUC = {t_auc:.4f}")

    print("\n" + ("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
