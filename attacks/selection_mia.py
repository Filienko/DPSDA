"""Selection-only membership inference for tabular PE (port of the aug-pe-baseline
attack of the same name).

Same motivation as the text version: the histogram-LLR attack (``histogram_mia.py``)
reads the released noised vote count directly, but in the realistic deployment the
3rd-party generation API only ever sees which of ITS OWN prior outputs got resent as
seeds for the next round -- never the vote count. This module scores membership from
that binary survival bit alone.

Mechanism (tabular specifics)
------------------------------
``pe.population.pe_population.PEPopulation`` (the population used for every dataset
here, in its ``selection_mode="rank"`` phase -- see ``example/tabular/*.py``'s
``CompositePopulation``): each round, ``_post_process_histogram`` floors the noised
count at ``histogram_threshold`` (0 for every dataset here): ``Y = max(count+noise,
threshold) - threshold``. ``_select_data`` then keeps the top-``num_samples`` pool
rows by that floored value. So "selected" is exactly ``Y >= tau_t`` for the round's
(unpublished) K-th order statistic -- structurally IDENTICAL to the text pipeline's
censored-Gaussian model, so the threshold-inversion math below is copied unchanged
from ``aug-pe-baseline/attacks/selection_mia.py``. (Rounds before the rank phase use
``selection_mode="sample"`` with no histogram persisted at all --
``histogram_mia.reconstruct`` already skips them, so this module does too, giving an
apples-to-apples comparison over the same rounds.)

Ground truth for "was cell j selected" comes for free and EXACTLY (no text-matching
heuristics needed): every row in checkpoint ``t+1`` carries
``PE.PARENT_SYN_DATA_INDEX``, the pandas index (within checkpoint ``t``'s per-class
rows) of the pool candidate it was derived from. So
``selected_mask = checkpoint_t.index.isin(checkpoint_{t+1}[parent_idx].unique())``
recovers the exact selection outcome. Verified empirically on the real adult run
(checkpoint 5->6): class-0 pool had 3032 candidates, 758 unique parents in checkpoint
6, and a full 758/758 overlap with checkpoint 5's index -- zero cross-class collisions.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

from attacks.histogram_mia import nearest_cell, cell_occupancy, _count_logpmf
from attacks.reconstruct import discover_iterations, _cell_embeddings
from pe.constant.data import (
    CLEAN_HISTOGRAM_COLUMN_NAME,
    PARENT_SYN_DATA_INDEX_COLUMN_NAME,
    LABEL_ID_COLUMN_NAME,
)

NEG_INF = -np.inf


# --------------------------------------------------------------------------- #
# Bernoulli selection-probability model -- byte-for-byte the aug-pe-baseline
# censored/clipped model (``Y = max(k+noise, threshold) - threshold``), which is
# exactly what ``_post_process_histogram`` implements for every dataset here.
# --------------------------------------------------------------------------- #
def _tail_prob_given_k(tau, ks, sigma, threshold):
    ks = np.asarray(ks, dtype=np.float64)
    if tau <= 1e-12:
        return np.ones_like(ks)
    if sigma <= 0:
        return (ks >= tau + threshold).astype(np.float64)
    return norm.sf((tau + threshold - ks) / sigma)


def selection_prob(tau, mu, sigma, threshold, dispersion, member=False, k_pad=12.0):
    mu = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    hi = max(float(mu.max()) if mu.size else 0.0, tau)
    hi = hi + k_pad * np.sqrt(max(hi, 1.0)) + 20.0
    ks = np.arange(0, int(np.ceil(hi)) + 1)
    tail = _tail_prob_given_k(tau, ks, sigma, threshold)
    if member:
        logpmf = np.full((mu.size, ks.size), NEG_INF)
        if ks.size > 1:
            logpmf[:, 1:] = _count_logpmf(ks[1:][None, :] - 1, mu[:, None], dispersion)
    else:
        logpmf = _count_logpmf(ks[None, :], mu[:, None], dispersion)
    pmf = np.exp(logpmf)
    return pmf @ tail


def solve_threshold(mu0_all, K, sigma, threshold, dispersion, iters=40):
    n = mu0_all.shape[0]
    if K <= 0 or K >= n:
        return None
    mu_max = float(mu0_all.max()) if n else 0.0
    hi = mu_max + 12.0 * np.sqrt(max(mu_max, 1.0)) + 30.0
    lo = -5.0 * max(sigma, 1.0)

    def expected_selected(tau):
        return float(selection_prob(tau, mu0_all, sigma, threshold, dispersion,
                                    member=False).sum())

    f_lo, f_hi = expected_selected(lo), expected_selected(hi)
    if f_lo <= K:
        return lo
    if f_hi >= K:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if expected_selected(mid) > K:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def score_records_selection(query_embeddings, iterations, mode="L2", ref_alpha=1.0,
                            soft_tau=0.0, dispersion=1.0, sigma=1.0, threshold=0.0,
                            occupancy_cache=None):
    """Selection-only per-round Bernoulli LLR, summed over rounds. Never reads a
    vote count -- only ``iterations[i]["selected_mask"]``. Same signature/semantics
    as the aug-pe-baseline version."""
    m = query_embeddings.shape[0]
    total = np.zeros(m, dtype=np.float64)
    eps = 1e-12

    for it_i, it in enumerate(iterations):
        F = it["cell_features"]
        mask = np.asarray(it["selected_mask"], dtype=bool)
        N = it["n_private"]
        K = int(mask.sum())
        n = F.shape[0]
        if K <= 0 or K >= n:
            continue

        if occupancy_cache is not None and (it_i, "q") in occupancy_cache:
            q = occupancy_cache[(it_i, "q")]
        else:
            q = cell_occupancy(it["reference_features"], F, mode=mode, k=1,
                               alpha=ref_alpha, soft_tau=soft_tau)
            if occupancy_cache is not None:
                occupancy_cache[(it_i, "q")] = q

        mu0_all = N * q
        if occupancy_cache is not None and (it_i, "tau") in occupancy_cache:
            tau = occupancy_cache[(it_i, "tau")]
        else:
            tau = solve_threshold(mu0_all, K, sigma, threshold, dispersion)
            if occupancy_cache is not None:
                occupancy_cache[(it_i, "tau")] = tau
        if tau is None:
            continue

        cells = nearest_cell(query_embeddings, F, mode=mode, k=1)
        mu0_q = mu0_all[cells]
        mu1_q = (N - 1) * q[cells]
        p0 = np.clip(selection_prob(tau, mu0_q, sigma, threshold, dispersion,
                                    member=False), eps, 1 - eps)
        p1 = np.clip(selection_prob(tau, mu1_q, sigma, threshold, dispersion,
                                    member=True), eps, 1 - eps)
        z = mask[cells].astype(np.float64)
        total += z * (np.log(p1) - np.log(p0)) + (1 - z) * (np.log(1 - p1) - np.log(1 - p0))
    return total


# --------------------------------------------------------------------------- #
# Data loading: pairs consecutive histogrammed checkpoints (t, t+1) and reads the
# selection outcome off ``PE.PARENT_SYN_DATA_INDEX`` -- exact, no text matching.
# --------------------------------------------------------------------------- #
def reconstruct_selection(checkpoint_folder, embed_fn, reference_features_by_class,
                          n_private_by_class, start_t=1, max_iters=None):
    iters = [t for t in discover_iterations(checkpoint_folder) if t >= start_t]
    if max_iters is not None:
        iters = iters[:max_iters]

    # Keep only checkpoints that actually carry a histogram (the rank-mode phase --
    # the initial sample-mode rounds have none and are excluded here exactly as
    # histogram_mia.reconstruct excludes them, for an apples-to-apples comparison).
    hist_t = []
    dfs = {}
    for t in iters:
        df = pd.read_pickle(os.path.join(checkpoint_folder, f"{t:09d}", "data_frame.pkl"))
        if CLEAN_HISTOGRAM_COLUMN_NAME in df.columns and df[CLEAN_HISTOGRAM_COLUMN_NAME].notna().any():
            hist_t.append(t)
            dfs[t] = df

    by_class = {}
    for i in range(len(hist_t) - 1):
        t, t_next = hist_t[i], hist_t[i + 1]
        if t_next != t + 1:
            continue  # gap in the checkpoint sequence -- selection link not valid
        df_t, df_next = dfs[t], dfs[t_next]
        voted = df_t[df_t[CLEAN_HISTOGRAM_COLUMN_NAME].notna()]
        if len(voted) == 0:
            continue
        for cls, sub in voted.groupby(LABEL_ID_COLUMN_NAME):
            cls = int(cls)
            sub_next = df_next[df_next[LABEL_ID_COLUMN_NAME] == cls]
            selected_parents = set(sub_next[PARENT_SYN_DATA_INDEX_COLUMN_NAME].dropna().unique())
            selected_mask = sub.index.isin(selected_parents)
            cell_features = _cell_embeddings(sub, embed_fn)
            by_class.setdefault(cls, []).append({
                "cell_features": cell_features,
                "selected_mask": np.asarray(selected_mask, dtype=bool),
                "reference_features": reference_features_by_class.get(
                    cls, np.empty((0, cell_features.shape[1]), dtype=np.float32)),
                "n_private": max(int(n_private_by_class.get(cls, 1)), 1),
            })
    return by_class


# --------------------------------------------------------------------------- #
# Embedder-free self-test: validates just the math (threshold inversion + Bernoulli
# LLR) on synthetic iteration dicts, independent of the checkpoint/pandas plumbing
# above (which was instead validated empirically against the real adult run -- see
# module docstring and the fork report that produced this file).
# --------------------------------------------------------------------------- #
def _selftest():
    from attacks.evaluate import evaluate, format_report

    rng = np.random.default_rng(0)
    dim, n_cand, n_priv, T = 12, 400, 250, 10
    threshold = 0.0

    def make_run(sigma):
        center = rng.normal(0, 5, size=dim).astype(np.float32)
        private = center + rng.normal(0, 1.0, size=(n_priv, dim)).astype(np.float32)
        members = private[rng.choice(n_priv, 120, replace=False)]
        nonmembers = center + rng.normal(0, 1.0, size=(120, dim)).astype(np.float32)
        reference = center + rng.normal(0, 1.0, size=(400, dim)).astype(np.float32)

        iterations = []
        for _ in range(T):
            cand = center + rng.normal(0, 1.3, size=(n_cand, dim)).astype(np.float32)
            assign = nearest_cell(private, cand, mode="L2", k=1)
            clean = np.bincount(assign, minlength=n_cand).astype(np.float64)
            noised = np.clip(clean + rng.normal(0, sigma, size=n_cand), threshold, None) - threshold
            K = n_cand // 4
            selected_mask = np.zeros(n_cand, dtype=bool)
            selected_mask[np.argsort(-noised, kind="stable")[:K]] = True
            iterations.append({"cell_features": cand, "selected_mask": selected_mask,
                               "reference_features": reference, "n_private": n_priv})
        return iterations, members, nonmembers

    ok = True
    for sigma in (2.0, 4.0, 8.0):
        iterations, members, nonmembers = make_run(sigma)
        q_emb = np.concatenate([members, nonmembers])
        labels = np.array([1] * len(members) + [0] * len(nonmembers))
        scores = score_records_selection(q_emb, iterations, mode="L2", sigma=sigma)
        m = evaluate(labels, scores)
        print(f"  sigma={sigma:>4}: selection-only AUC={m['auc']:.4f}  "
             f"TPR@1%FPR={m['tpr@fpr=0.01']:.4f}")
        if m["auc"] <= 0.55:
            print(f"FAIL: AUC too low at sigma={sigma}"); ok = False

        # Negative control: shuffle which candidate survived each round.
        shuffled = [dict(it, selected_mask=rng.permutation(it["selected_mask"]))
                   for it in iterations]
        sc = score_records_selection(q_emb, shuffled, mode="L2", sigma=sigma)
        m_shuf = evaluate(labels, sc)
        print(f"    shuffled control AUC={m_shuf['auc']:.4f} (expect ~0.5)")
        if abs(m_shuf["auc"] - 0.5) > 0.08:
            print("FAIL: shuffled control not at chance"); ok = False

    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())
    raise SystemExit(
        "Real-data runs: build reference_features_by_class / n_private_by_class as "
        "in run_mia.py, then call reconstruct_selection(checkpoint_folder, embed_fn, "
        "reference_features_by_class, n_private_by_class) and "
        "score_records_selection() per class. Requires the run's population to use "
        "selection_mode='rank' (true for every dataset here in its post-initial "
        "phase). Use --selftest to validate the math offline first.")


if __name__ == "__main__":
    main()
