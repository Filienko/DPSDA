"""A stronger membership-inference attack against the tabular PE histogram.

``attacks/histogram_mia.py`` scores a record by finding its nearest *surviving*
synthetic cell in each round and forming a noised Poisson/NB likelihood ratio on
that cell's released count. That reads a strict subset of what PE actually
releases, and the subset is chosen in a way that correlates with membership. This
module reconstructs the full release and adds the channels the baseline drops.

What PE releases, and what the baseline misses
----------------------------------------------
At iteration ``t`` (``pe/runner/pe.py:203-207``) the histogram is computed over
**every** row of checkpoint ``t-1`` for that class -- call these the *candidates*.
``pe/population/pe_population.py:129-134`` then keeps only the top
``num_samples`` by noised count, and ``pe/api/tabular/tabular_api.py:150-157``
rebuilds the child frame with just ``PE.TABULAR`` + ``PE.LABEL_ID``, so children
carry no histogram or embedding column. ``attacks/reconstruct.py:91`` keys off
``PE.CLEAN_HISTOGRAM.notna()`` and therefore recovers only the ~1-in-4 survivors.

Four consequences, one per channel below:

1. **Wrong cell, wrong null.** A private record voted for its nearest of ~400
   candidates; the baseline scores its nearest of ~100 survivors. When the true
   cell did not survive, the baseline reads an unrelated, more popular, more
   distant cell. ``cell_occupancy`` also renormalises ``q`` over the survivor set,
   inflating ``mu0 = N*q_j`` roughly fourfold -- the misspecification that
   ``histogram_mia.estimate_dispersion`` describes fighting in its own docstring.
   Fixed here by :func:`reconstruct_full`, which recovers the candidate pool from
   checkpoint ``t-1`` (every row has ``PE.TABULAR``, and the embedding is
   deterministic, so the pool is fully recomputable -- it is also in the public
   ``synthetic_tab/*.csv``).

2. **Selection censoring is evidence.** A candidate that did not survive satisfies
   ``y_j < tau`` with ``tau`` the smallest surviving count. That is a likelihood
   term, not a missing value. :func:`llr_censored` scores it.

3. **Rounds 1-4 are thrown away** (they use ``keep_selected=False``, so those
   checkpoints carry no counts at all) even though they consume the same per-round
   privacy budget and run at the largest mutation rate. They are not signal-free:
   those rounds use ``selection_mode="sample"``, i.e. multinomial resampling with
   replacement over ``p ∝ ReLU(y_j)``, so the number of children pointing at a
   candidate is an ``n_draws``-sample readout of the entire noisy histogram --
   including cells whose counts were never persisted anywhere.
   :func:`llr_multiplicity` scores it.

4. **The ancestry graph is unread.** ``PE.PARENT_SYN_DATA_INDEX`` is an exact row
   index into the previous checkpoint (``pe/data/data.py:62-73`` masks without
   reindexing; ``pe/runner/pe.py:219`` resets the index only after concat), so
   every cell's full trajectory is recoverable.
   ``histogram_mia.lineage_density_records`` averages a Parzen density over rounds
   and so cannot tell "one lineage persisted beside this record for ten rounds"
   from "ten unrelated cells passed by once". :func:`trajectory_scores` uses the
   forest: lineage persistence, and the selection-induced *drift* of the local
   synthetic cloud toward the record -- a count-free signal that DP noise touches
   only through which cells were selected.

Scale and fusion
----------------
Channels 1-3 are genuine log-likelihood ratios and are therefore additive with no
tuned weights -- unlike ``improved.py``'s z-scored sum, which loses power because
``lineage_density_records`` is not an LLR and its scale is arbitrary. The
trajectory channel is not an LLR, so both it and the LLR sum are standardised
against a D_ref null before being combined, putting them on a common
"standard deviations above the non-member null" scale.

That standardisation is also the fix for ``improved.py``'s ``calibrate="lira"``,
which applies one affine map per class and is therefore rank-preserving *within* a
class -- it can only move a pooled AUC. Here the null is estimated per record, from
D_ref records with a comparable ``mu0`` profile (:func:`calibrate_per_example`),
which is what matters for TPR at low FPR and for comparing inliers against
outliers on one scale.

Nothing in ``attacks/`` is modified; this module only imports from it. Run
``--self_test`` first: it recomputes the clean histogram from the private records
over the reconstructed pool and checks it against ``PE.CLEAN_HISTOGRAM``
cell-for-cell. That check passes only if the candidate pool is right, and fails by
construction on the survivor-only pool.

Validation on a simulated PE run
--------------------------------
Against a simulation that reproduces the real loop exactly (per-class histogram,
uncensored Gaussian noise, sample-then-rank composite population, keep_selected,
parent pointers), in a regime tuned to look like eps=10 -- ``sigma=1.84`` and a
``mu0`` near 1.7, which reproduces the survivor-only baseline's reported AUC:

    survivor-only count (baseline)   0.559
    full pool, count                 0.642
      + censored                     0.661
      + mult (the discarded rounds)   0.667
      + traj, flat weight            0.645
      + traj, router weight          0.662

So the three likelihood-ratio channels are the win, and they compose. The geometry
channel is real on its own (0.573 alone, and it reads rounds and records the count
channel cannot) but does **not** add on top of a correctly specified count channel
-- it is a noisier view of the same evidence, and adding it costs AUC under either
fusion. It is therefore off by default (``--channels llr``); ``--channels all``
turns it on, and ``--traj_fuse router`` beats the flat z-sum consistently when it
is on. The ordering above held in an easier regime too (baseline 0.698 -> 0.870),
so it is not an artifact of one operating point.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, nbinom
from scipy.special import logsumexp

from pe.constant.data import (
    CLEAN_HISTOGRAM_COLUMN_NAME,
    DP_HISTOGRAM_COLUMN_NAME,
    POST_PROCESSED_DP_HISTOGRAM_COLUMN_NAME,
    LABEL_ID_COLUMN_NAME,
    TABULAR_DATA_COLUMN_NAME,
    PARENT_SYN_DATA_INDEX_COLUMN_NAME,
    FROM_LAST_FLAG_COLUMN_NAME,
)

from attacks.histogram_mia import nearest_cell, cell_occupancy
from attacks.reconstruct import discover_iterations
from attacks.tabular_embedding import load_private, make_embed_fn
from attacks.audit_set import build_audit_set, reference_rows_by_class
from attacks.evaluate import evaluate, format_report

# Floor on the observation-noise scale. sigma == 0 (the clean/"pure" regime) would
# make the Gaussian observation term a delta and break the count grid; 1e-2 is
# narrow enough that logsumexp still selects the exact integer count, so the pure
# regime degenerates correctly to the exact count likelihood.
SIGMA_FLOOR = 1e-2

ALL_CHANNELS = ("count", "censored", "mult", "traj")
#: The additive log-likelihood-ratio channels -- the default, and the strongest
#: combination measured. ``traj`` is excluded; see the module docstring.
LLR_CHANNELS = ("count", "censored", "mult")


# --------------------------------------------------------------------------- #
# Batched count priors
# --------------------------------------------------------------------------- #
def _count_logpmf_batch(k, mean, dispersion):
    """``log P(count == k)`` with ``var = dispersion*mean``, broadcasting.

    ``dispersion == 1`` is Poisson; ``> 1`` is the negative binomial with the same
    mean. Mirrors ``histogram_mia._count_logpmf`` but broadcasts over a batch of
    records so a whole round is one call instead of one scipy call per record.
    """
    mean = np.maximum(np.asarray(mean, dtype=np.float64), 1e-9)
    disp = np.maximum(np.asarray(dispersion, dtype=np.float64), 1.0)
    if float(np.max(disp)) <= 1.0 + 1e-9:
        return poisson.logpmf(k, mean)
    # nbinom with dispersion -> 1 has n -> inf, which is numerically Poisson but
    # produces inf/nan at exactly 1; hold it just above.
    disp = np.maximum(disp, 1.0 + 1e-4)
    p = 1.0 / disp
    n = mean / (disp - 1.0)
    return nbinom.logpmf(k, n, p)


def _count_grid(mu0, mu1, y=None, pad=20.0):
    """Shared integer count grid wide enough for every record in the round."""
    hi = max(float(np.max(mu0)), float(np.max(mu1)), 1.0)
    if y is not None and np.size(y):
        finite = np.asarray(y, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            hi = max(hi, float(np.max(finite)))
    hi = hi + 12.0 * np.sqrt(max(hi, 1.0)) + pad
    return np.arange(0, int(np.ceil(hi)) + 1)


def _log_priors(ks, mu0, mu1, disp):
    """``(m, K)`` log priors under H0 (``NB(mu0)``) and H1 (``1 + NB(mu1)``)."""
    kk = ks[None, :]
    m0 = np.asarray(mu0, dtype=np.float64)[:, None]
    m1 = np.asarray(mu1, dtype=np.float64)[:, None]
    d = np.asarray(disp, dtype=np.float64)
    d = d[:, None] if d.ndim else np.full((m0.shape[0], 1), float(d))
    log_pri0 = _count_logpmf_batch(kk, m0, d)
    log_pri1 = np.full((m0.shape[0], ks.shape[0]), -np.inf)
    log_pri1[:, 1:] = _count_logpmf_batch(kk[:, 1:] - 1, m1, d)
    return log_pri0, log_pri1


def _finalize(logP1, logP0):
    out = logP1 - logP0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Channel A -- observed and censored count LLRs
# --------------------------------------------------------------------------- #
def llr_observed(y, mu0, mu1, sigma, dispersion=1.0):
    """Batched LLR for cells whose noised count was released.

    Same model as ``histogram_mia.llr_noised(censored=False)`` -- the tabular
    mechanism (``pe/dp/gaussian.py:175``) releases ``clean + N(0, sigma^2)`` with no
    clipping -- but vectorised over records and carrying the NB dispersion that
    ``histogram_mia.llr_noised_vec`` drops.
    """
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    mu0 = np.atleast_1d(np.asarray(mu0, dtype=np.float64))
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    if y.size == 0:
        return np.zeros(0)
    sigma = max(float(sigma), SIGMA_FLOOR)
    ks = _count_grid(mu0, mu1, y)
    obs = norm.logpdf(y[:, None], loc=ks[None, :], scale=sigma)
    log_pri0, log_pri1 = _log_priors(ks, mu0, mu1, dispersion)
    return _finalize(logsumexp(log_pri1 + obs, axis=1),
                     logsumexp(log_pri0 + obs, axis=1))


def llr_censored(tau, mu0, mu1, sigma, dispersion=1.0):
    """Batched LLR for cells that did *not* survive selection.

    Top-``k`` selection (``pe_population.py:131``) means a non-surviving candidate
    satisfies ``y_j < tau``, where ``tau`` is the smallest count among the
    survivors. The observation term is therefore the left tail
    ``log P(Y < tau | k) = log Phi((tau - k)/sigma)`` in place of the Gaussian
    density; the priors are unchanged.

    Members shift the count up by one and so are *more* likely to survive, which
    makes a censored cell mild evidence against membership -- correctly signed, and
    coherent across rounds, where the baseline instead scored an unrelated distant
    survivor and got noise.

    Selection is jointly top-``k``, so cells are not strictly independent; treating
    ``tau`` as a known constant is the usual conditional-on-the-order-statistic
    approximation and is what makes the per-cell factorisation valid.
    """
    mu0 = np.atleast_1d(np.asarray(mu0, dtype=np.float64))
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    if mu0.size == 0:
        return np.zeros(0)
    sigma = max(float(sigma), SIGMA_FLOOR)
    ks = _count_grid(mu0, mu1, np.array([tau]))
    obs = norm.logcdf((float(tau) - ks[None, :]) / sigma)
    log_pri0, log_pri1 = _log_priors(ks, mu0, mu1, dispersion)
    obs = np.broadcast_to(obs, log_pri0.shape)
    return _finalize(logsumexp(log_pri1 + obs, axis=1),
                     logsumexp(log_pri0 + obs, axis=1))


# --------------------------------------------------------------------------- #
# Channel B -- parent-multiplicity LLR (the only channel in the "sample" rounds)
# --------------------------------------------------------------------------- #
def _expected_relu_sum(mu, sigma):
    """``E[sum_j ReLU(k_j + N(0, sigma^2))]`` given per-cell means ``mu``.

    ``E[ReLU(Z)] = m*Phi(m/s) + s*phi(m/s)`` for ``Z ~ N(m, s^2)``. Summed over
    hundreds of cells this concentrates tightly, which is what lets
    :func:`llr_multiplicity` treat the normaliser as a known constant.
    """
    mu = np.asarray(mu, dtype=np.float64)
    s = max(float(sigma), SIGMA_FLOOR)
    z = mu / s
    return float(np.sum(mu * norm.cdf(z) + s * norm.pdf(z)))


def llr_multiplicity(c, n_draws, relu_sum, mu0, mu1, sigma, dispersion=1.0,
                     n_y=129):
    """Batched LLR from how many children a candidate cell produced.

    In the ``selection_mode="sample"`` rounds (``pe_population.py:122-128``)::

        prob = count / count.sum()
        indices = np.random.choice(len(df), size=num_samples, p=prob)

    is multinomial *with replacement* over ``p_j = ReLU(y_j) / S``. So the number of
    children pointing at candidate ``j`` is ``Binomial(n_draws, p_j)``, and since
    ``p_j`` is small this is well approximated by ``Poisson(n_draws * p_j)``. Those
    rounds persist no counts at all, so this is the only readout of them -- and it
    covers every candidate, including the ones that produced no children, where
    zero offspring is itself evidence that ``y_j <= 0``.

    ``y_j = k_j + N(0, sigma^2)`` is integrated on a grid and ``k_j`` marginalised
    over the two priors. The inner integral depends only on ``(c, k)``, so it is
    tabulated once per round and indexed per record.
    """
    c = np.atleast_1d(np.asarray(c)).astype(np.int64)
    mu0 = np.atleast_1d(np.asarray(mu0, dtype=np.float64))
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    if c.size == 0 or relu_sum <= 0 or n_draws <= 0:
        return np.zeros(c.size)
    sigma = max(float(sigma), SIGMA_FLOOR)
    ks = _count_grid(mu0, mu1)

    # Shared y grid spanning the count grid plus Gaussian tails.
    lo = -6.0 * sigma
    hi = float(ks[-1]) + 6.0 * sigma
    yg = np.linspace(lo, hi, n_y)
    dy = yg[1] - yg[0]
    lam = np.maximum(n_draws * np.maximum(yg, 0.0) / relu_sum, 1e-12)   # (n_y,)

    # log G[c_val, k] = log \int Poisson(c_val; lam(y)) N(y; k, sigma) dy
    c_vals = np.unique(c)
    log_pois = poisson.logpmf(c_vals[:, None], lam[None, :])            # (C, n_y)
    log_norm = norm.logpdf(yg[None, :], loc=ks[:, None], scale=sigma)   # (K, n_y)
    logG = logsumexp(log_pois[:, None, :] + log_norm[None, :, :],
                     axis=2) + np.log(dy)                               # (C, K)

    rows = np.searchsorted(c_vals, c)
    G = logG[rows]                                                      # (m, K)
    log_pri0, log_pri1 = _log_priors(ks, mu0, mu1, dispersion)
    return _finalize(logsumexp(log_pri1 + G, axis=1),
                     logsumexp(log_pri0 + G, axis=1))


# --------------------------------------------------------------------------- #
# Full-release reconstruction
# --------------------------------------------------------------------------- #
def _parent_positions(values, pos_of):
    """Map a ``PE.PARENT_SYN_DATA_INDEX`` column to positions in the candidate pool.

    Returns -1 for anything unmappable. The NaN guard matters: ``Data.concat``
    (``pe/population/pe_population.py:179``) unions rows that carry different
    column sets, so a real frame can hold a NaN parent where the simulation never
    produced one, and a bare ``int(nan)`` would raise mid-run.
    """
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        out[i] = pos_of.get(int(v), -1) if pd.notna(v) else -1
    return out


def _read_checkpoint(folder, t, cache):
    if t not in cache:
        path = os.path.join(folder, f"{t:09d}", "data_frame.pkl")
        cache[t] = pd.read_pickle(path) if os.path.isfile(path) else None
    return cache[t]


def reconstruct_full(checkpoint_folder, embed_fn, reference_features_by_class,
                     n_private_by_class, start_t=1, max_iters=None,
                     use_clean=False, survivors_only=False):
    """Recover, per class and per round, the **full** candidate pool plus every
    released observation about it.

    Round ``t`` votes on the rows of checkpoint ``t-1`` and writes its outcome into
    checkpoint ``t``, so each round dict pairs *cells from ``t-1``* with
    *counts from ``t``*. Survivors are the checkpoint-``t`` rows with
    ``PE.FROM_LAST_FLAG == 1``, and their ``PE.PARENT_SYN_DATA_INDEX`` is literally
    their own row index back in checkpoint ``t-1``, which is what makes the
    survivor-to-candidate map exact rather than a nearest-neighbour guess.

    Each round dict carries the four keys ``histogram_mia.score_records`` expects
    (``cell_features``, ``counts``, ``reference_features``, ``n_private``) plus:

    ``observed``      boolean mask of candidates whose count was released
    ``tau``           smallest surviving count, i.e. the selection threshold
    ``child_mult``    number of children each candidate produced
    ``n_draws``       total children this round (the multinomial sample size)
    ``mode``          ``"rank"`` or ``"sample"``
    ``parent_pos``    each candidate's parent's position in the previous round's
                      pool, or -1 -- the edge list of the ancestry forest

    ``survivors_only=True`` reproduces the baseline's survivor-only pool, for the
    side-by-side comparison and for the ``--self_test`` contrast.
    """
    ts = [t for t in discover_iterations(checkpoint_folder) if t >= max(start_t, 1)]
    if max_iters is not None:
        ts = ts[:max_iters]

    cache = {}
    by_class = {}
    prev_pos_of = {}      # class -> {global row index in ckpt t-2 : pool position}
    for t in ts:
        prev = _read_checkpoint(checkpoint_folder, t - 1, cache)
        cur = _read_checkpoint(checkpoint_folder, t, cache)
        cache.pop(t - 2, None)
        if prev is None or cur is None:
            prev_pos_of = {}
            continue
        has_parent = PARENT_SYN_DATA_INDEX_COLUMN_NAME in cur.columns
        noised_col = (DP_HISTOGRAM_COLUMN_NAME if DP_HISTOGRAM_COLUMN_NAME in cur.columns
                      else POST_PROCESSED_DP_HISTOGRAM_COLUMN_NAME)
        count_col = CLEAN_HISTOGRAM_COLUMN_NAME if use_clean else noised_col
        if count_col not in cur.columns and not has_parent:
            prev_pos_of = {}
            continue

        next_pos_of = {}
        for cls in sorted(prev[LABEL_ID_COLUMN_NAME].unique()):
            cls = int(cls)
            pool = prev[prev[LABEL_ID_COLUMN_NAME] == cls]
            if len(pool) == 0:
                continue
            cand_global = pool.index.to_numpy()
            pos_of = {int(g): i for i, g in enumerate(cand_global)}
            n_cand = len(cand_global)

            sub = cur[cur[LABEL_ID_COLUMN_NAME] == cls]
            if FROM_LAST_FLAG_COLUMN_NAME in sub.columns:
                surv = sub[sub[FROM_LAST_FLAG_COLUMN_NAME] == 1]
                kids = sub[sub[FROM_LAST_FLAG_COLUMN_NAME] != 1]
            elif count_col in sub.columns:
                keep = sub[CLEAN_HISTOGRAM_COLUMN_NAME].notna() \
                    if CLEAN_HISTOGRAM_COLUMN_NAME in sub.columns else sub[count_col].notna()
                surv, kids = sub[keep], sub[~keep]
            else:
                surv, kids = sub.iloc[:0], sub

            counts = np.full(n_cand, np.nan, dtype=np.float64)
            if len(surv) and has_parent and count_col in surv.columns:
                sp = _parent_positions(surv[PARENT_SYN_DATA_INDEX_COLUMN_NAME], pos_of)
                sv = surv[count_col].to_numpy(dtype=np.float64)
                ok = sp >= 0
                counts[sp[ok]] = sv[ok]
            observed = np.isfinite(counts)

            # Candidates with no released count and no ancestry are unusable.
            if not observed.any() and not has_parent:
                continue

            child_mult = np.zeros(n_cand, dtype=np.int64)
            n_draws = 0
            if len(kids) and has_parent:
                kp = _parent_positions(kids[PARENT_SYN_DATA_INDEX_COLUMN_NAME], pos_of)
                kp = kp[kp >= 0]
                if kp.size:
                    child_mult = np.bincount(kp, minlength=n_cand).astype(np.int64)
                n_draws = int(kp.size)

            mode = "rank" if observed.any() else "sample"

            # Each candidate's parent position in the *previous* round's pool.
            parent_pos = np.full(n_cand, -1, dtype=np.int64)
            back = prev_pos_of.get(cls)
            if back is not None and PARENT_SYN_DATA_INDEX_COLUMN_NAME in pool.columns:
                parent_pos = _parent_positions(
                    pool[PARENT_SYN_DATA_INDEX_COLUMN_NAME], back)

            if survivors_only:
                if not observed.any():
                    continue
                sel = np.where(observed)[0]
                feats = embed_fn(pool[TABULAR_DATA_COLUMN_NAME].iloc[sel].tolist())
                rec = {
                    "t": int(t), "cell_features": feats, "counts": counts[sel],
                    "observed": np.ones(sel.size, dtype=bool), "tau": None,
                    "child_mult": child_mult[sel], "n_draws": n_draws, "mode": mode,
                    "parent_pos": np.full(sel.size, -1, dtype=np.int64),
                }
            else:
                feats = embed_fn(pool[TABULAR_DATA_COLUMN_NAME].tolist())
                rec = {
                    "t": int(t), "cell_features": feats, "counts": counts,
                    "observed": observed,
                    "tau": float(counts[observed].min()) if observed.any() else None,
                    "child_mult": child_mult, "n_draws": n_draws, "mode": mode,
                    "parent_pos": parent_pos,
                }
            rec["reference_features"] = reference_features_by_class.get(
                cls, np.empty((0, rec["cell_features"].shape[1]), dtype=np.float32))
            rec["n_private"] = max(int(n_private_by_class.get(cls, 1)), 1)
            by_class.setdefault(cls, []).append(rec)
            next_pos_of[cls] = pos_of
        prev_pos_of = next_pos_of
    return by_class


# --------------------------------------------------------------------------- #
# Channels A+B over all rounds
# --------------------------------------------------------------------------- #
def score_llr_channels(query_emb, iters, sigma, channels, mode="L2",
                       ref_alpha=1.0, soft_tau=0.0, dispersion=1.0,
                       occ_cache=None, return_mu0=False):
    """Sum the (additive, genuinely log-likelihood-ratio) channels over rounds.

    Per round the query is assigned to its nearest candidate over the **full**
    pool, then routed by what the release says about that cell: its count was
    published (``count``), it was censored by top-k selection (``censored``), or
    the round published no counts and we read the child multiplicity instead
    (``mult``).

    Also returns each record's mean ``log(1 + mu0)`` across rounds, which
    :func:`calibrate_per_example` uses to find comparable D_ref records.
    """
    m = query_emb.shape[0]
    total = np.zeros(m, dtype=np.float64)
    mu0_prof = np.zeros(m, dtype=np.float64)
    n_rounds = 0
    occ_cache = {} if occ_cache is None else occ_cache

    for ti, it in enumerate(iters):
        F = it["cell_features"]
        if F.shape[0] == 0:
            continue
        N = it["n_private"]
        if ti in occ_cache:
            q = occ_cache[ti]
        else:
            q = cell_occupancy(it["reference_features"], F, mode=mode, k=1,
                               alpha=ref_alpha, soft_tau=soft_tau)
            occ_cache[ti] = q

        cells = nearest_cell(query_emb, F, mode=mode, k=1)
        mu0 = N * q[cells]
        mu1 = (N - 1) * q[cells]
        mu0_prof += np.log1p(mu0)
        n_rounds += 1

        obs = it["observed"][cells]
        ell = np.zeros(m, dtype=np.float64)

        if it["mode"] == "rank":
            if "count" in channels and obs.any():
                ell[obs] = llr_observed(np.asarray(it["counts"])[cells[obs]],
                                        mu0[obs], mu1[obs], sigma, dispersion)
            if "censored" in channels and (~obs).any() and it["tau"] is not None:
                ell[~obs] = llr_censored(it["tau"], mu0[~obs], mu1[~obs],
                                         sigma, dispersion)
        elif "mult" in channels and it["n_draws"] > 0:
            relu_sum = _expected_relu_sum(N * q, sigma)
            ell = llr_multiplicity(np.asarray(it["child_mult"])[cells],
                                   it["n_draws"], relu_sum, mu0, mu1,
                                   sigma, dispersion)
        total += ell

    if return_mu0:
        return total, mu0_prof / max(n_rounds, 1)
    return total


# --------------------------------------------------------------------------- #
# Channel C -- ancestry-aware trajectory
# --------------------------------------------------------------------------- #
def _lineage_roots(iters):
    """Root id per candidate per round, by chaining ``parent_pos`` backwards.

    Two cells share a root exactly when they descend from the same round-1
    candidate, which is what distinguishes one persistent lineage beside a record
    from a succession of unrelated visitors.
    """
    roots, nxt = [], 0
    prev = None
    for it in iters:
        n = it["cell_features"].shape[0]
        cur = np.full(n, -1, dtype=np.int64)
        pp = it["parent_pos"]
        if prev is not None and pp is not None:
            ok = (pp >= 0) & (pp < prev.size)
            cur[ok] = prev[pp[ok]]
        fresh = cur < 0
        if fresh.any():
            cur[fresh] = np.arange(nxt, nxt + int(fresh.sum()))
            nxt += int(fresh.sum())
        roots.append(cur)
        prev = cur
    return roots


def _pairwise_min(queries, points, chunk=2048):
    """Nearest point index and distance for each query."""
    m = queries.shape[0]
    if points.shape[0] == 0:
        return np.full(m, -1, dtype=np.int64), np.full(m, np.inf)
    q = np.ascontiguousarray(queries, dtype=np.float32)
    f = np.ascontiguousarray(points, dtype=np.float32)
    f_sq = np.einsum("ij,ij->i", f, f)
    q_sq = np.einsum("ij,ij->i", q, q)
    idx = np.empty(m, dtype=np.int64)
    dist = np.empty(m, dtype=np.float64)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        d2 = q_sq[s:e, None] + f_sq[None, :] - 2.0 * (q[s:e] @ f.T)
        j = np.argmin(d2, axis=1)
        idx[s:e] = j
        dist[s:e] = np.sqrt(np.clip(np.take_along_axis(d2, j[:, None], 1)[:, 0], 0.0, None))
    return idx, dist


def trajectory_scores(query_emb, iters, mode="L2", bandwidth=None):
    """Ancestry-aware geometric signal: lineage persistence plus selection drift.

    Both parts read cell *positions* and the ancestry forest, never a vote count,
    so neither is diluted by the DP noise directly -- it enters only through which
    cells were selected.

    **Persistence** -- the longest run of consecutive rounds in which the query's
    nearest survivor is within ``bandwidth`` *and belongs to the same lineage*.
    ``histogram_mia.lineage_density_records`` averages a Parzen density over rounds
    and so scores a persistent neighbour the same as a sequence of unrelated ones.

    **Drift** -- variations are isotropic (``tabular_api.py:129-145`` adds
    ``U(-r, r)`` per column), so a child's displacement from its parent carries no
    information on its own. Selection is what makes it informative: among a
    parent's children, the ones that moved toward a *member* pick up that member's
    vote and are the ones that survive. Averaging
    ``<child - parent, x - parent>`` over surviving children near ``x`` measures
    that pull. This is the part that should work where counts fail, because it does
    not require the record's cell to have won a top-k contest.

    Returns ``(m, 2)`` of ``[persistence, drift]``, standardised by the caller.
    """
    m = query_emb.shape[0]
    out = np.zeros((m, 2), dtype=np.float64)
    if not iters:
        return out

    # Bandwidth: median nearest-cell distance pooled over rounds.
    pooled = []
    for it in iters:
        if it["cell_features"].shape[0]:
            pooled.append(_pairwise_min(query_emb, it["cell_features"])[1])
    if not pooled:
        return out
    h = bandwidth if bandwidth else float(np.median(np.concatenate(pooled)))
    h = max(h, 1e-6)

    roots = _lineage_roots(iters)

    run = np.zeros(m, dtype=np.int64)
    best = np.zeros(m, dtype=np.int64)
    last_root = np.full(m, -1, dtype=np.int64)
    drift = np.zeros(m, dtype=np.float64)
    drift_w = np.zeros(m, dtype=np.float64)

    for ti, it in enumerate(iters):
        F = it["cell_features"]
        obs = it["observed"]
        if F.shape[0] == 0:
            continue

        # --- persistence over surviving cells ---
        if obs.any():
            sel = np.where(obs)[0]
            j, d = _pairwise_min(query_emb, F[sel])
            r = roots[ti][sel[j]]
            near = d <= h
            same = near & (r == last_root)
            run = np.where(same, run + 1, np.where(near, 1, 0))
            best = np.maximum(best, run)
            last_root = np.where(near, r, -1)

        # --- drift of surviving children toward the query ---
        pp = it["parent_pos"]
        if ti == 0 or pp is None:
            continue
        par_F = iters[ti - 1]["cell_features"]
        live = obs & (pp >= 0) & (pp < par_F.shape[0])
        if not live.any():
            continue
        sel = np.where(live)[0]
        parent = par_F[pp[sel]]                       # (s, d)
        step = F[sel] - parent                        # (s, d)
        # <step, x - parent> / ||x - parent||, kernel-weighted by parent proximity.
        # Blocked at 128 queries: the (block, cells, dim) intermediate is the
        # largest allocation here, and `dim` is wide once categoricals are one-hot.
        for s0 in range(0, m, 128):
            s1 = min(s0 + 128, m)
            rel = query_emb[s0:s1, None, :] - parent[None, :, :]     # (b, s, d)
            nrm = np.linalg.norm(rel, axis=2)                        # (b, s)
            proj = np.einsum("bsd,sd->bs", rel, step) / np.maximum(nrm, 1e-9)
            w = np.exp(-0.5 * (nrm / h) ** 2)
            drift[s0:s1] += (w * proj).sum(axis=1)
            drift_w[s0:s1] += w.sum(axis=1)

    out[:, 0] = best.astype(np.float64)
    out[:, 1] = drift / np.maximum(drift_w, 1e-9)
    return out


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibrate_per_example(scores, key, ref_scores, ref_key, n_bins=10):
    """Standardise each record against D_ref records with a comparable null.

    ``improved.py``'s ``calibrate="lira"`` fits one Gaussian per class and applies
    an affine map, which is rank-preserving *within* a class -- it can only shift a
    pooled AUC, never reorder records inside a class. The null here actually varies
    per record: a record sitting in a dense region has a large ``mu0`` and a wide,
    high-mean score null, while an outlier's ``mu0`` is near zero. Comparing the two
    on one raw scale is what makes outliers look uninformative.

    ``key`` is the record's mean ``log(1 + mu0)`` across rounds. D_ref records are
    binned by quantiles of their own key, the null mean and spread are measured per
    bin, and both are interpolated at the query's key. No shadow models are needed:
    D_ref records are known non-members, which is exactly the null.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if ref_scores is None or np.size(ref_scores) < 2 * n_bins:
        f = scores[np.isfinite(scores)]
        if f.size == 0:
            return np.zeros_like(scores)
        sd = float(f.std()) or 1.0
        return (scores - float(f.mean())) / sd

    ref_scores = np.asarray(ref_scores, dtype=np.float64)
    ref_key = np.asarray(ref_key, dtype=np.float64)
    good = np.isfinite(ref_scores) & np.isfinite(ref_key)
    ref_scores, ref_key = ref_scores[good], ref_key[good]
    if ref_scores.size < 2 * n_bins:
        sd = float(ref_scores.std()) or 1.0
        return (scores - float(ref_scores.mean())) / sd

    edges = np.quantile(ref_key, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    centers, mus, sds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ref_key >= lo) & (ref_key <= hi)
        if sel.sum() < 5:
            continue
        centers.append(0.5 * (lo + hi))
        mus.append(float(ref_scores[sel].mean()))
        sds.append(float(ref_scores[sel].std()))
    if len(centers) < 2:
        sd = float(ref_scores.std()) or 1.0
        return (scores - float(ref_scores.mean())) / sd

    centers = np.asarray(centers)
    mu_hat = np.interp(key, centers, np.asarray(mus))
    sd_hat = np.interp(key, centers, np.maximum(np.asarray(sds), 1e-9))
    return (scores - mu_hat) / np.maximum(sd_hat, 1e-9)


def _zref(x, ref):
    """Standardise ``x`` against a D_ref sample of the same statistic."""
    x = np.asarray(x, dtype=np.float64)
    base = ref if (ref is not None and np.size(ref) > 1) else x
    base = np.asarray(base, dtype=np.float64)
    base = base[np.isfinite(base)]
    if base.size == 0:
        return np.zeros_like(x)
    sd = float(base.std()) or 1.0
    return (x - float(base.mean())) / sd


# --------------------------------------------------------------------------- #
# Top-level scoring
# --------------------------------------------------------------------------- #
def _router_weights(key, scale=1.0):
    """Per-record weight on the geometry channel, from how weak the count channel
    is for that record.

    The count LLR's per-round signal-to-noise for the ``+1`` self-vote is
    ``1 / sqrt(sigma^2 + phi*mu0)``, so it degrades as ``mu0`` grows: a record in a
    busy region has its single vote buried under the other ``N*q_j``. The geometry
    channel has no such denominator. ``key`` is the record's mean ``log(1 + mu0)``
    across rounds, so routing weight toward geometry when ``key`` is high --
    membership-blind, since ``mu0`` depends only on the released cells and D_ref --
    is the adaptive count-for-dense / geometry-for-sparse combination that a flat
    sum cannot express.
    """
    key = np.asarray(key, dtype=np.float64)
    f = key[np.isfinite(key)]
    if f.size < 2:
        return np.ones_like(key)
    m, s = float(np.median(f)), float(f.std())
    if s <= 0:
        return np.ones_like(key)
    return 1.0 / (1.0 + np.exp(-(key - m) / (scale * s)))


def score_all(by_class, query_emb, classes, ref_by_class, sigma, channels,
              mode="L2", ref_alpha=1.0, soft_tau=0.0, dispersion=1.0,
              calib="perexample", traj_weight=1.0, traj_fuse="router"):
    """Score every record, per class, and fuse the channels.

    Channels ``count``/``censored``/``mult`` are log-likelihood ratios and so are
    summed directly -- no fusion weights. The trajectory channel is not an LLR, so
    it and the calibrated LLR sum are each put on a "standard deviations above the
    D_ref null" scale before being combined.

    ``traj_fuse="flat"`` adds the geometry channel with a constant weight, which is
    what ``improved.py:147``'s z-sum does. ``"router"`` (default) makes the weight
    per-record via :func:`_router_weights`: geometry is trusted where the count
    channel is structurally weak. ``running_summary.md`` §5 flags the flat sum as
    the reason naive fusion fails to recover outliers -- count's near-chance scores
    there add variance without adding signal.
    """
    classes = np.asarray(classes)
    out = np.full(len(classes), np.nan, dtype=np.float64)
    llr_channels = [c for c in channels if c in ("count", "censored", "mult")]

    for cls in np.unique(classes):
        iters = by_class.get(int(cls))
        if not iters:
            continue
        mask = classes == cls
        qe = query_emb[mask]
        ref = ref_by_class.get(int(cls)) if ref_by_class else None
        ref = np.asarray(ref) if ref is not None and len(ref) else None
        occ = {}

        # Always computed: with no LLR channels selected the score stays zero but
        # the mu0 profile is still needed for calibration and routing.
        s, key = score_llr_channels(qe, iters, sigma, llr_channels, mode=mode,
                                    ref_alpha=ref_alpha, soft_tau=soft_tau,
                                    dispersion=dispersion, occ_cache=occ,
                                    return_mu0=True)
        if llr_channels and calib != "none":
            if ref is not None:
                rs, rk = score_llr_channels(ref, iters, sigma, llr_channels, mode=mode,
                                            ref_alpha=ref_alpha, soft_tau=soft_tau,
                                            dispersion=dispersion, occ_cache=occ,
                                            return_mu0=True)
                if calib == "perexample":
                    s = calibrate_per_example(s, key, rs, rk)
                else:                                   # "class": affine, per class
                    s = _zref(s, rs)
            else:
                s = _zref(s, None)

        if "traj" in channels:
            tj = trajectory_scores(qe, iters, mode=mode)
            tj_ref = trajectory_scores(ref, iters, mode=mode) if ref is not None else None
            t_sum = np.zeros(qe.shape[0], dtype=np.float64)
            for c in range(tj.shape[1]):
                t_sum += _zref(tj[:, c], None if tj_ref is None else tj_ref[:, c])
            w = traj_weight * (_router_weights(key) if traj_fuse == "router" else 1.0)
            s = s + w * _zref(t_sum, None)

        out[mask] = s

    finite = out[np.isfinite(out)]
    out[~np.isfinite(out)] = (finite.min() - 1.0) if finite.size else 0.0
    return out


# --------------------------------------------------------------------------- #
# Reconstruction self-test
# --------------------------------------------------------------------------- #
def self_test(checkpoint_folder, embed_fn, priv_rows_by_class, n_private_by_class,
              start_t=1, max_iters=None, mode="L2"):
    """Prove the candidate pool is right by rebuilding the clean histogram.

    Assign every private record of a class to its nearest cell over the
    reconstructed pool and compare the resulting vote counts against the released
    ``PE.CLEAN_HISTOGRAM``. This is exactly the assignment
    ``pe/histogram/nearest_neighbors.py:197-224`` performed, so on the correct pool
    it must agree cell-for-cell.

    The same check is run against the baseline's survivor-only pool. It fails there
    by construction -- votes that belong to censored cells have nowhere to go and
    pile up on surviving neighbours -- which is the defect this module fixes.
    """
    report = {}
    for label, only in (("full_pool", False), ("survivors_only", True)):
        by_class = reconstruct_full(checkpoint_folder, embed_fn, {},
                                    n_private_by_class, start_t=start_t,
                                    max_iters=max_iters, use_clean=True,
                                    survivors_only=only)
        rounds = exact = 0
        worst = 0.0
        for cls, iters in by_class.items():
            rows = priv_rows_by_class.get(int(cls))
            if not rows:
                continue
            pemb = embed_fn(rows)
            for it in iters:
                if not it["observed"].any():
                    continue
                cells = nearest_cell(pemb, it["cell_features"], mode=mode, k=1)
                recomputed = np.bincount(np.ravel(cells),
                                         minlength=it["cell_features"].shape[0])
                obs = it["observed"]
                diff = np.abs(recomputed[obs] - np.nan_to_num(it["counts"][obs]))
                rounds += 1
                exact += int(diff.max() == 0)
                worst = max(worst, float(diff.max()))
        report[label] = {
            "rounds_checked": rounds,
            "rounds_exact": exact,
            "max_abs_count_error": worst,
            "pass": bool(rounds > 0 and exact == rounds),
        }
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _scrape_noise_multiplier(run_dir):
    """Read sigma out of the run's log, same regex as ``ablate.py:61-67``."""
    path = os.path.join(run_dir, "log.txt")
    if not os.path.isfile(path):
        return None
    with open(path, "r", errors="ignore") as fh:
        hits = re.findall(r"noise_multiplier=([0-9.eE+-]+)", fh.read())
    return float(hits[-1]) if hits else None


def _group_auc(labels, scores, member_group, group_name):
    """AUC of one member subgroup against all non-members.

    Same construction as ``outlier_disparity.py:140-148`` so the numbers are
    directly comparable to the existing Exp E / Exp F tables.
    """
    from sklearn.metrics import roc_auc_score
    sel = (labels == 1) & (member_group == group_name)
    if sel.sum() < 5:
        return None
    y = np.concatenate([np.ones(int(sel.sum())), np.zeros(int((labels == 0).sum()))])
    s = np.concatenate([scores[sel], scores[labels == 0]])
    return float(roc_auc_score(y, s))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run_dir", required=True,
                   help="PE run folder (containing checkpoint/ and log.txt)")
    p.add_argument("--checkpoint_folder", default="",
                   help="defaults to <run_dir>/checkpoint")
    p.add_argument("--train_csv", required=True, help="private CSV (members)")
    p.add_argument("--test_csv", required=True, help="holdout CSV (non-members + D_ref)")
    p.add_argument("--metadata", required=True)
    p.add_argument("--reference_csv", default="")
    p.add_argument("--noise_multiplier", type=float, default=0.0,
                   help="0 = scrape from <run_dir>/log.txt")
    p.add_argument("--num_nearest_neighbor", type=int, default=1)
    p.add_argument("--channels", default="llr",
                   help="comma list of " + ",".join(ALL_CHANNELS) + ", or 'llr' "
                        "(the three additive LLR channels; default and strongest "
                        "measured) or 'all' (adds the geometry channel, which is "
                        "informative alone but costs AUC when fused)")
    p.add_argument("--calib", default="perexample",
                   choices=["perexample", "class", "none"])
    p.add_argument("--traj_weight", type=float, default=1.0)
    p.add_argument("--traj_fuse", default="router", choices=["router", "flat"],
                   help="how the (non-LLR) geometry channel joins the LLR sum: "
                        "'router' weights it per record by how weak the count "
                        "channel is there; 'flat' is the constant-weight z-sum "
                        "that improved.py uses.")
    p.add_argument("--nn_mode", default="L2", choices=["L2", "IP", "cos_sim"])
    p.add_argument("--ref_alpha", type=float, default=1.0,
                   help="Laplace smoothing on q_j. The baseline needs 0.05 to "
                        "offset its survivor-only renormalisation; on the full "
                        "pool the untuned 1.0 should be appropriate.")
    p.add_argument("--soft_tau", type=float, default=0.0)
    p.add_argument("--dispersion", type=float, default=1.0,
                   help="null var/mean. The baseline's 1.8 compensates for the "
                        "misspecified mu0; on the full pool start at 1.0.")
    p.add_argument("--n_members", type=int, default=1500)
    p.add_argument("--n_nonmembers", type=int, default=1500)
    p.add_argument("--ref_holdout_frac", type=float, default=0.5)
    p.add_argument("--ref_max_per_class", type=int, default=2000)
    p.add_argument("--start_t", type=int, default=1)
    p.add_argument("--max_iters", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self_test", action="store_true",
                   help="run the reconstruction check and exit")
    p.add_argument("--compare_baseline", action="store_true",
                   help="also score histogram_mia.score_records on the SAME audit set")
    p.add_argument("--per_group", action="store_true",
                   help="report AUC by within-class kNN-outlierness tertile")
    p.add_argument("--out", default="", help="write the report JSON here")
    args = p.parse_args(argv)

    ckpt = args.checkpoint_folder or os.path.join(args.run_dir, "checkpoint")
    nm = args.noise_multiplier or _scrape_noise_multiplier(args.run_dir)
    if not nm:
        print("ERROR: no --noise_multiplier and none found in log.txt", file=sys.stderr)
        return 2
    sigma = nm * np.sqrt(args.num_nearest_neighbor)
    max_iters = args.max_iters or None
    if args.channels == "all":
        channels = tuple(ALL_CHANNELS)
    elif args.channels == "llr":
        channels = tuple(LLR_CHANNELS)
    else:
        channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())
    bad = [c for c in channels if c not in ALL_CHANNELS]
    if bad:
        print(f"ERROR: unknown channel(s) {bad}; known: {ALL_CHANNELS}", file=sys.stderr)
        return 2

    print(f"sigma = {sigma:.4f}  channels = {channels}  calib = {args.calib}")
    print("Loading private data / embedding...")
    priv_data, info = load_private(args.train_csv, args.metadata)
    embed_fn = make_embed_fn(priv_data, info)
    n_private_by_class = {int(k): int(v) for k, v in
                          priv_data.data_frame[LABEL_ID_COLUMN_NAME].value_counts().items()}

    if args.self_test:
        priv_rows_by_class = {}
        for cls, sub in priv_data.data_frame.groupby(LABEL_ID_COLUMN_NAME):
            priv_rows_by_class[int(cls)] = [list(r) for r in
                                            sub[TABULAR_DATA_COLUMN_NAME].tolist()]
        rep = self_test(ckpt, embed_fn, priv_rows_by_class, n_private_by_class,
                        start_t=args.start_t, max_iters=max_iters, mode=args.nn_mode)
        print(json.dumps(rep, indent=2))
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(rep, fh, indent=2)
        return 0 if rep.get("full_pool", {}).get("pass") else 1

    reference_csv = args.reference_csv or args.test_csv
    ref_frac = args.ref_holdout_frac if reference_csv == args.test_csv else 0.0

    print("Building audit set...")
    rows, classes, labels, sizes = build_audit_set(
        args.train_csv, args.test_csv, args.metadata,
        n_members=args.n_members, n_nonmembers=args.n_nonmembers,
        seed=args.seed, ref_holdout_frac=ref_frac)
    print(f"  members={int(labels.sum())} non-members={int((labels == 0).sum())}")

    ref_rows = reference_rows_by_class(reference_csv, args.metadata,
                                       max_per_class=args.ref_max_per_class,
                                       seed=args.seed, ref_holdout_frac=ref_frac)
    ref_by_class = {cls: embed_fn(r) for cls, r in ref_rows.items()}
    query_emb = embed_fn(rows)

    print("Reconstructing full candidate pools...")
    by_class = reconstruct_full(ckpt, embed_fn, ref_by_class, n_private_by_class,
                                start_t=args.start_t, max_iters=max_iters)
    if not by_class:
        print("ERROR: no rounds reconstructed -- check --start_t / --run_dir",
              file=sys.stderr)
        return 1
    for cls, iters in sorted(by_class.items())[:1]:
        modes = {}
        for it in iters:
            modes[it["mode"]] = modes.get(it["mode"], 0) + 1
        cens = np.mean([1.0 - it["observed"].mean() for it in iters
                        if it["mode"] == "rank"] or [0.0])
        print(f"  class {cls}: {len(iters)} rounds {modes}, "
              f"pool={iters[-1]['cell_features'].shape[0]} cells, "
              f"mean censored fraction={cens:.2f}")

    report = {"run_dir": args.run_dir, "sigma": sigma, "channels": list(channels),
              "calib": args.calib, "n_members": int(labels.sum()),
              "n_nonmembers": int((labels == 0).sum()), "data_sizes": sizes,
              "args": vars(args), "results": {}}

    print("Scoring (strong attack)...")
    scores = score_all(by_class, query_emb, classes, ref_by_class, sigma, channels,
                       mode=args.nn_mode, ref_alpha=args.ref_alpha,
                       soft_tau=args.soft_tau, dispersion=args.dispersion,
                       calib=args.calib, traj_weight=args.traj_weight,
                       traj_fuse=args.traj_fuse)
    metrics = evaluate(labels, scores)
    report["results"]["strong"] = metrics
    print(format_report(metrics, f"strong [{','.join(channels)}]"))

    if args.compare_baseline:
        print("\nScoring (baseline histogram_mia.score_records, same audit set)...")
        from attacks.reconstruct import reconstruct as reconstruct_baseline
        from attacks.histogram_mia import score_records
        base_by_class = reconstruct_baseline(ckpt, embed_fn, ref_by_class,
                                             n_private_by_class,
                                             start_t=args.start_t, max_iters=max_iters)
        base = np.full(len(classes), np.nan)
        for cls in np.unique(classes):
            iters = base_by_class.get(int(cls))
            if not iters:
                continue
            mask = classes == cls
            base[mask] = score_records(query_emb[mask], iters, mode=args.nn_mode,
                                       regime="noised", sigma=sigma, censored=False,
                                       ref_alpha=0.05, soft_tau=0.02, dispersion=1.8)
        fin = base[np.isfinite(base)]
        base[~np.isfinite(base)] = (fin.min() - 1.0) if fin.size else 0.0
        bm = evaluate(labels, base)
        report["results"]["baseline"] = bm
        print(format_report(bm, "baseline (survivor-only pool)"))
        report["results"]["delta_auc"] = metrics["auc"] - bm["auc"]
        print(f"\ndelta AUC (strong - baseline) = {report['results']['delta_auc']:+.4f}")

    if args.per_group:
        print("\nPer-outlierness-tertile AUC (member subgroup vs all non-members)...")
        try:
            from attacks.outlier_disparity import knn_outlierness
        except Exception as exc:                      # hard-coded paths in that module
            print(f"  skipped: could not import knn_outlierness ({exc})")
            knn_outlierness = None
        if knn_outlierness is not None:
            train_emb = embed_fn([list(r) for r in
                                  priv_data.data_frame[TABULAR_DATA_COLUMN_NAME].tolist()])
            train_cls = priv_data.data_frame[LABEL_ID_COLUMN_NAME].to_numpy()
            mem = labels == 1
            rank = np.zeros(len(labels))
            rank[mem] = knn_outlierness(query_emb[mem], classes[mem], train_emb, train_cls)
            edges = np.quantile(rank[mem], [1 / 3, 2 / 3])
            grp = np.where(rank <= edges[0], "inlier",
                           np.where(rank <= edges[1], "mid", "outlier"))
            groups = {}
            for name in ("inlier", "mid", "outlier"):
                entry = {"strong": _group_auc(labels, scores, grp, name)}
                if args.compare_baseline:
                    entry["baseline"] = _group_auc(labels, base, grp, name)
                groups[name] = entry
                line = f"  {name:8s} strong={entry['strong']}"
                if "baseline" in entry:
                    line += f"  baseline={entry['baseline']}"
                print(line)
            report["results"]["per_group"] = groups

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
