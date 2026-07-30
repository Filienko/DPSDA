"""Improved scoring with independently-toggleable components, for the Phase 2
ablation. Every component defaults OFF, and with all OFF ``score_config`` matches
the baseline ``attacks.histogram_mia.score_records`` (noised regime).

Components
----------
- ``pool_m`` (int, 1)        : aggregate the per-round LLR over the top-M nearest
                               cells, proximity-weighted (M=1 = baseline).
- ``round_weight`` (str)     : 'uniform' (baseline) | 'snr' | 'invvar' -- weight
                               each round's LLR by informativeness instead of an
                               equal-weight sum.
- ``calibrate`` (str)        : 'none' (baseline) | 'lira' -- per-class Gaussian
                               calibration of the aggregate against D_ref records
                               scored the same way (removes per-class scale).
- ``selection_model`` (bool) : condition the noised likelihood on the cell being
                               selected (count >= round threshold) -- selection bias.
- ``density`` (bool)         : also compute the geometric survivor-density (lineage)
                               signal and combine via z-scored sum.

Everything reuses primitives from ``attacks.histogram_mia``.
"""

from dataclasses import dataclass

import numpy as np

from attacks.histogram_mia import (
    nearest_cell, cell_occupancy, llr_noised_vec, lineage_density_records,
)


@dataclass
class Config:
    pool_m: int = 1
    round_weight: str = "uniform"      # uniform | snr | invvar
    calibrate: str = "none"            # none | lira
    selection_model: bool = False
    density: bool = False
    mode: str = "L2"


def _per_round_llr(query_emb, iters, cfg, sigma, occ):
    """(m, T) per-round (pooled) LLR for one class. Fills ``occ[t]`` with the
    per-round occupancy q so round weights can reuse it."""
    m, T = query_emb.shape[0], len(iters)
    out = np.zeros((m, T), dtype=np.float64)
    for t, it in enumerate(iters):
        F = it["cell_features"]
        counts = np.asarray(it["counts"], dtype=np.float64)
        N = it["n_private"]
        if t not in occ:
            occ[t] = cell_occupancy(it["reference_features"], F, mode=cfg.mode,
                                    k=1, alpha=1.0)
        q = occ[t]
        tau = float(counts.min()) if cfg.selection_model else None
        k = min(cfg.pool_m, F.shape[0])
        idx = nearest_cell(query_emb, F, mode=cfg.mode, k=k)
        if k == 1:
            idx = idx[:, None]
        # Proximity weights over the M nearest cells (softmax of -distance).
        if k > 1:
            sel = F[idx]                                   # (m, k, d)
            d = np.linalg.norm(query_emb[:, None, :] - sel, axis=2)  # (m, k)
            wsoft = np.exp(-(d - d.min(axis=1, keepdims=True)) /
                           (d.mean() + 1e-9))
            wsoft /= wsoft.sum(axis=1, keepdims=True)
        else:
            wsoft = np.ones((m, 1))
        pooled = np.zeros(m, dtype=np.float64)
        for col in range(k):
            j = idx[:, col]
            ell = llr_noised_vec(counts[j], N * q[j], (N - 1) * q[j], sigma,
                                 select_tau=tau)
            pooled += wsoft[:, col] * ell
        out[:, t] = pooled
    return out


def _round_weights(iters, scheme, sigma, occ):
    T = len(iters)
    if scheme == "uniform":
        return np.ones(T)
    w = np.zeros(T)
    for t, it in enumerate(iters):
        q = occ.get(t)
        mu0 = it["n_private"] * (float(np.mean(q)) if q is not None else 0.0)
        if scheme == "snr":            # ~ inverse std of the count
            w[t] = 1.0 / (sigma + np.sqrt(max(mu0, 1.0)))
        elif scheme == "invvar":       # ~ inverse variance
            w[t] = 1.0 / (sigma ** 2 + mu0 + 1.0)
        else:
            w[t] = 1.0
    s = w.sum()
    return (w / s * T) if s > 0 else np.ones(T)


def _fill_nan(scores):
    finite = scores[np.isfinite(scores)]
    scores[np.isnan(scores)] = (finite.min() - 1.0) if finite.size else 0.0
    return scores


def _zscore(x):
    x = np.asarray(x, dtype=np.float64)
    f = x[np.isfinite(x)]
    if f.size == 0:
        return np.zeros_like(x)
    mu, sd = float(f.mean()), float(f.std())
    return (x - mu) / sd if sd > 0 else (x - mu)


def score_config(by_class, query_emb, classes, ref_by_class, cfg, sigma,
                 max_iters=None):
    """Final per-record membership score under ``cfg``. ``ref_by_class`` may be
    ``None`` (then 'lira' and 'density' calibration silently fall back)."""
    classes = np.asarray(classes)
    count_scores = np.full(len(classes), np.nan)
    dens_scores = np.full(len(classes), np.nan)
    for cls in np.unique(classes):
        iters = by_class.get(int(cls))
        if not iters:
            continue
        if max_iters is not None:
            iters = iters[:max_iters]
        if not iters:
            continue
        mask = classes == cls
        occ = {}
        pr = _per_round_llr(query_emb[mask], iters, cfg, sigma, occ)
        w = _round_weights(iters, cfg.round_weight, sigma, occ)
        s = (pr * w[None, :]).sum(axis=1)
        ref = None if ref_by_class is None else ref_by_class.get(int(cls))
        if cfg.calibrate == "lira" and ref is not None and len(ref) > 1:
            pr_ref = _per_round_llr(np.asarray(ref), iters, cfg, sigma, occ)
            s_ref = (pr_ref * w[None, :]).sum(axis=1)
            fin = s_ref[np.isfinite(s_ref)]
            if fin.size > 1:                       # per-class Gaussian null
                mu, sd = float(fin.mean()), float(fin.std())
                s = (s - mu) / (sd if sd > 0 else 1.0)
        count_scores[mask] = s
        if cfg.density:
            dens_scores[mask] = lineage_density_records(
                query_emb[mask], iters, mode=cfg.mode, reference_features=ref)
    count_scores = _fill_nan(count_scores)
    if cfg.density:
        dens_scores = _fill_nan(dens_scores)
        return _zscore(count_scores) + _zscore(dens_scores)
    return count_scores
