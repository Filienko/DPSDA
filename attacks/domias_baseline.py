"""DOMIAS baseline, reimplemented directly against our tabular embeddings -- a
density-RATIO test (not nearest-neighbor), read-only reference:
``/home/daniilf/DOMIAS/src/domias/evaluator.py:218-315`` (the ``kde`` /
``prior`` branches) and ``src/domias/baselines.py``. We keep only the KDE
density-ratio core (no BNAF neural flow -- explicitly excluded as too heavy for
this comparison):

    density_gen  = stats.gaussian_kde(synth_set.T)
    p_G_evaluated = density_gen(X_test.T)
    density_data = stats.gaussian_kde(reference_set.T)
    p_R_evaluated = density_data(X_test.T)
    p_rel = p_G_evaluated / (p_R_evaluated + 1e-10)

``synth_set`` = the LAST released round's synthetic pool per class (same
"generator output" FBB uses); ``reference_set`` = our existing D_ref embeddings.

Numerical-stability adaptation (NOT the p_R fallback the task anticipated --
see below): tabular embeddings one-hot-encode every categorical column, and a
one-hot block's columns sum to a constant 1 row-wise, which makes the sample
covariance matrix EXACTLY singular for ANY sample size -- confirmed empirically
(a synthetic one-hot-collinearity test reproduces scipy's
"data appears to lie in a lower-dimensional subspace" LinAlgError deterministically).
So a plain ``gaussian_kde`` call fails on both p_G and p_R for every dataset here
except artificial-characters (whose embedding is fully numeric, no one-hot). The
fix scipy's own error message suggests -- PCA dimensionality reduction before
fitting -- is applied to BOTH densities whenever the raw fit is singular: project
onto the data's own non-degenerate principal subspace (rank = numerical rank of
the centered data, additionally capped so the KDE always has at least 3x as many
points as dimensions), fit the KDE there, and project query points into that same
basis before evaluating. This is a numerical necessity to run ``kde`` mode AT ALL
on one-hot tabular data, not a substitution of a different attack.

The p_R "prior" FALLBACK the task explicitly authorized is used separately, only
when a class's reference set is too small for a stable fit even after PCA
(breast-cancer's per-class D_ref is 18-25 points): p_R(x) is then the standard
multivariate normal density evaluated on X_test standardized (z-scored) by the
reference set's own mean/std -- "assume standard-normal density on whitened
features" -- rather than evaluator.py's literal ``norm.pdf(X_test)`` line, which
is elementwise (per-feature), not a joint density, and looks like a latent bug in
the reference repo rather than the intended semantics.
"""

import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal

MIN_POINTS_PER_DIM = 3  # require n_points >= MIN_POINTS_PER_DIM * rank for a KDE fit


def _pca_rank(Xc, tol_scale=100.0):
    """Numerical rank of centered data ``Xc`` (rows = points)."""
    if Xc.shape[0] < 2:
        return 0
    s = np.linalg.svd(Xc, compute_uv=False)
    tol = s.max() * max(Xc.shape) * np.finfo(np.float64).eps * tol_scale
    return int((s > tol).sum())


def _fit_kde_safe(points):
    """Fit ``gaussian_kde`` on ``points`` (n, d), PCA-deranking if singular or if
    there aren't enough points per dimension. Returns ``(kde, project_fn)`` or
    ``(None, None)`` if no stable fit is possible (caller should use the "prior"
    fallback)."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 2:
        return None, None
    mean = points.mean(axis=0)
    Xc = points - mean
    rank = _pca_rank(Xc)
    rank = min(rank, max(1, n // MIN_POINTS_PER_DIM))
    if rank < 1 or n <= rank + 1:
        return None, None
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Vt[:rank].T  # (d, rank)
    reduced = Xc @ proj
    try:
        kde = stats.gaussian_kde(reduced.T)
    except np.linalg.LinAlgError:
        return None, None

    def project(z):
        return (np.asarray(z, dtype=np.float64) - mean) @ proj

    return kde, project


def domias_scores(query_embeddings, synth_set, reference_set):
    """DOMIAS membership score ``p_G(x) / (p_R(x) + 1e-10)``, higher = more
    member-like. Returns ``(scores, meta)`` where ``meta`` records which mode
    (``"kde"`` or ``"prior"``) was actually used for p_R (p_G always uses KDE,
    PCA-deranked if needed -- see module docstring)."""
    X_test = np.asarray(query_embeddings, dtype=np.float64)
    meta = {"p_G_mode": "kde", "p_R_mode": "kde"}

    kde_g, proj_g = _fit_kde_safe(synth_set)
    if kde_g is None:
        # p_G has no stable KDE fit even after PCA-derank (e.g. too few pool
        # points) -- fall back to the same standardized-Gaussian prior used for
        # p_R below; DOMIAS degenerates toward the density-ratio baseline being
        # ~flat under the numerator, which we report plainly (see run script).
        meta["p_G_mode"] = "prior"
        mu, sd = synth_set.mean(axis=0), synth_set.std(axis=0) + 1e-8
        p_G = multivariate_normal.pdf((X_test - mu) / sd, mean=np.zeros(X_test.shape[1]),
                                      cov=np.eye(X_test.shape[1]))
    else:
        p_G = kde_g(proj_g(X_test).T)

    kde_r, proj_r = _fit_kde_safe(reference_set)
    if kde_r is None:
        meta["p_R_mode"] = "prior"
        mu, sd = reference_set.mean(axis=0), reference_set.std(axis=0) + 1e-8
        p_R = multivariate_normal.pdf((X_test - mu) / sd, mean=np.zeros(X_test.shape[1]),
                                      cov=np.eye(X_test.shape[1]))
    else:
        p_R = kde_r(proj_r(X_test).T)

    return p_G / (p_R + 1e-10), meta
