"""Core membership-inference scoring against the PE nearest-neighbor histogram.

This is the *same* attack as ``aug-pe-baseline/attacks/histogram_mia.py`` --- the
NN assignment, the reference-occupancy Poisson null, the per-round pure / noised
log-likelihood ratios, and the multi-round aggregation are unchanged. The only
adaptation for the tabular PE pipeline (``pe/dp/gaussian.py``) is the noised
observation model: that mechanism releases ``clean_count + N(0, sigma^2)`` with
**no** left-censoring / thresholding, whereas the text pipeline stored
``max(k + noise, threshold) - threshold``. The ``censored`` flag selects between
the two; everything else is byte-for-byte the original logic.

The PE DP histogram (``pe/histogram/nearest_neighbors.py``) assigns every private
record to the Voronoi cell of its nearest synthetic sample and counts the votes
per cell. A member therefore adds exactly one vote (per nearest neighbor) to the
cell it falls in. Because the released synthetic embeddings are public, an
attacker can recompute that same nearest-neighbor assignment for any candidate
record and read the vote count in the cell it would land in.
"""

import numpy as np
from scipy.stats import poisson, norm, nbinom
from scipy.special import logsumexp

NEG_INF = -np.inf


def _count_logpmf(k, mean, dispersion=1.0):
    """log P(count == k) under the null, mean ``mean``, ``var = dispersion*mean``.

    ``dispersion == 1`` -> Poisson exactly; ``dispersion > 1`` -> negative binomial
    with ``var = dispersion*mean`` (the vote counts are overdispersed relative to
    Poisson, so a Poisson null manufactures false positives at low FPR).
    """
    mean = np.maximum(mean, 1e-9)
    if dispersion <= 1.0 + 1e-9:
        return poisson.logpmf(k, mean)
    p = 1.0 / dispersion                 # nbinom: mean=n(1-p)/p, var=mean/p
    n = mean / (dispersion - 1.0)
    return nbinom.logpmf(k, n, p)


# --------------------------------------------------------------------------- #
# Nearest-neighbor assignment (matches pe/histogram distance modes)
# --------------------------------------------------------------------------- #
def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def nearest_cell(queries, cell_features, mode="L2", k=1, chunk=4096):
    """Return the indices of the ``k`` nearest cells for each query row.

    Mirrors the L2 / IP / cos_sim search in ``pe/histogram/nearest_neighbor_backend``.
    Returns an ``(m, k)`` int array (``(m,)`` if k == 1).
    """
    q = np.ascontiguousarray(queries, dtype=np.float32)
    f = np.ascontiguousarray(cell_features, dtype=np.float32)
    if mode == "cos_sim":
        q, f = _normalize(q), _normalize(f)
    out = np.empty((q.shape[0], k), dtype=np.int64)
    f_sq = np.einsum("ij,ij->i", f, f) if mode == "L2" else None
    for s in range(0, q.shape[0], chunk):
        e = min(s + chunk, q.shape[0])
        qb = q[s:e]
        if mode == "L2":
            # argmin ||q-f||^2  ==  argmin (f_sq - 2 q.f)
            scores = f_sq[None, :] - 2.0 * (qb @ f.T)
            idx = np.argpartition(scores, kth=min(k, f.shape[0] - 1), axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(scores, idx, axis=1), axis=1)
        else:  # IP or cos_sim -> maximise inner product
            scores = qb @ f.T
            idx = np.argpartition(-scores, kth=min(k, f.shape[0] - 1), axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
        out[s:e] = np.take_along_axis(idx, order, axis=1)
    return out[:, 0] if k == 1 else out


def nearest_distance(queries, cell_features, mode="L2", chunk=4096):
    """Distance from each query to its single nearest cell (same metric as
    ``nearest_cell``). For L2 this is the Euclidean distance; for IP / cos_sim it
    is ``1 - similarity`` so that 'closer' is always a smaller number. Returns an
    ``(m,)`` float array (``+inf`` if there are no cells)."""
    m = queries.shape[0]
    if cell_features.shape[0] == 0:
        return np.full(m, np.inf, dtype=np.float64)
    q = np.ascontiguousarray(queries, dtype=np.float32)
    f = np.ascontiguousarray(cell_features, dtype=np.float32)
    if mode == "cos_sim":
        q, f = _normalize(q), _normalize(f)
    out = np.empty(m, dtype=np.float64)
    f_sq = np.einsum("ij,ij->i", f, f) if mode == "L2" else None
    q_sq = np.einsum("ij,ij->i", q, q) if mode == "L2" else None
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        qb = q[s:e]
        if mode == "L2":
            d2 = q_sq[s:e, None] + f_sq[None, :] - 2.0 * (qb @ f.T)
            out[s:e] = np.sqrt(np.clip(d2.min(axis=1), 0.0, None))
        else:  # IP / cos_sim -> 1 - max similarity
            out[s:e] = 1.0 - (qb @ f.T).max(axis=1)
    return out


# --------------------------------------------------------------------------- #
# Lineage / survivor-geometry signal (membership leaks through which synthetic
# cells SURVIVE near a record across rounds, not just their vote counts)
# --------------------------------------------------------------------------- #
def _mean_kernel(queries, points, h, mode="L2", chunk=2048):
    """Mean Gaussian kernel ``exp(-d^2/2h^2)`` from each query to a point set
    (a Parzen density estimate at each query). Returns ``(m,)``; 0 if no points."""
    m = queries.shape[0]
    if points.shape[0] == 0:
        return np.zeros(m, dtype=np.float64)
    q = np.ascontiguousarray(queries, dtype=np.float32)
    f = np.ascontiguousarray(points, dtype=np.float32)
    if mode == "cos_sim":
        q, f = _normalize(q), _normalize(f)
    f_sq = np.einsum("ij,ij->i", f, f)
    q_sq = np.einsum("ij,ij->i", q, q)
    out = np.empty(m, dtype=np.float64)
    inv = 1.0 / (2.0 * h * h)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        d2 = q_sq[s:e, None] + f_sq[None, :] - 2.0 * (q[s:e] @ f.T)
        out[s:e] = np.exp(-np.clip(d2, 0.0, None) * inv).mean(axis=1)
    return out


def lineage_density_records(query_embeddings, iterations, mode="L2",
                            bandwidth=None, reference_features=None):
    """Per-record geometric membership score from survivor proximity over rounds.

    In the kept/selected (rank) rounds, the cells carried in ``iterations`` are the
    synthetic samples that *survived* selection. A member seeded / kept a synthetic
    cell near itself, so across rounds its nearest survivor sits closer (and more
    persistently) than a non-member's. This signal is independent of the vote-count
    magnitude, so it is far less diluted by large private-set size N than the
    Poisson count LLR.

    Score = time-averaged Parzen density of survivor cells evaluated at the query.
    ``bandwidth`` defaults to the median nearest-survivor distance.

    If ``reference_features`` (in-distribution non-private points, i.e. D_ref) is
    given, the score is **calibrated** into a log density-ratio
    ``log(survivor_density) - log(reference_density)``. This removes the local
    data-density confound (dense regions look close to everything): it asks whether
    the query is closer to *survivors* than a typical non-member there, isolating
    the membership residual (members are over-represented among survivors).
    Returns an ``(m,)`` score (higher = more member-like).
    """
    m = query_embeddings.shape[0]
    if not iterations:
        return np.zeros(m, dtype=np.float64)
    # Bandwidth from nearest-survivor distances pooled over rounds.
    dists = np.full((len(iterations), m), np.nan)
    for i, it in enumerate(iterations):
        F = it["cell_features"]
        if F.shape[0] == 0:
            continue
        dists[i] = nearest_distance(query_embeddings, F, mode=mode)
    finite = dists[np.isfinite(dists)]
    if finite.size == 0:
        return np.zeros(m, dtype=np.float64)
    h = bandwidth if bandwidth else float(np.median(finite))
    h = max(h, 1e-6)

    # Survivor density: mean over rounds of the per-round Parzen density.
    dens = np.zeros(m, dtype=np.float64)
    n_used = 0
    for it in iterations:
        F = it["cell_features"]
        if F.shape[0] == 0:
            continue
        dens += _mean_kernel(query_embeddings, F, h, mode=mode)
        n_used += 1
    dens = dens / max(n_used, 1)

    if reference_features is None or len(reference_features) == 0:
        return dens
    ref_dens = _mean_kernel(query_embeddings, np.asarray(reference_features),
                            h, mode=mode)
    eps = 1e-12
    return np.log(dens + eps) - np.log(ref_dens + eps)


def _soft_votes(reference_features, cell_features, mode, tau, chunk=2048):
    """Soft (temperature-``tau``) analogue of hard nearest-cell binning.

    Each reference record distributes a unit of vote mass across cells by a softmax
    over (negative) distance instead of committing its whole vote to the single
    argmin cell.  This lowers the variance of ``q_j`` in the sparse cells (few or no
    hard reference votes) where a member's self-vote is most detectable.  Returns
    the per-cell summed mass ``(n_cells,)``.
    """
    q = np.ascontiguousarray(reference_features, dtype=np.float64)
    f = np.ascontiguousarray(cell_features, dtype=np.float64)
    if mode == "cos_sim":
        q, f = _normalize(q), _normalize(f)
    f_sq = np.einsum("ij,ij->i", f, f)
    mass = np.zeros(f.shape[0], dtype=np.float64)
    for s in range(0, q.shape[0], chunk):
        e = min(s + chunk, q.shape[0])
        qb = q[s:e]
        if mode == "L2":
            d = f_sq[None, :] - 2.0 * (qb @ f.T) + np.einsum("ij,ij->i", qb, qb)[:, None]
            d = np.maximum(d, 0.0)
        else:  # IP / cos_sim: larger inner product == closer
            d = -(qb @ f.T)
        if s == 0:
            scale = tau * (np.median(d) + 1e-12)
        w = d / (-scale)
        w -= w.max(axis=1, keepdims=True)
        np.exp(w, out=w)
        w /= w.sum(axis=1, keepdims=True)
        mass += w.sum(axis=0)
    return mass


def cell_occupancy(reference_features, cell_features, mode="L2", k=1, alpha=1.0,
                   soft_tau=0.0):
    """Estimate per-cell occupancy probability ``q_j`` from in-distribution,
    non-private reference data.

    ``q_j`` is the expected number of votes a single random in-distribution
    record contributes to cell ``j`` (with ``k``-NN voting, each record casts
    ``k`` votes). Laplace smoothing (``alpha``) keeps empty reference cells from
    forcing a zero/inf likelihood. Returns an array of length ``n_cells`` that
    sums to ``k`` (before smoothing).

    ``soft_tau > 0`` replaces the hard nearest-cell histogram of the reference set
    with a temperature-``tau`` soft assignment (see ``_soft_votes``); with a small
    ``alpha`` this carries the low-FPR power.  ``soft_tau == 0`` recovers the
    original hard binning.
    """
    n_cells = cell_features.shape[0]
    if reference_features.shape[0] == 0:
        return np.full(n_cells, k / max(n_cells, 1))
    r = reference_features.shape[0]
    if soft_tau > 0:
        votes = _soft_votes(reference_features, cell_features, mode, soft_tau)
    else:
        nn = nearest_cell(reference_features, cell_features, mode=mode, k=k)
        votes = np.bincount(np.ravel(nn), minlength=n_cells).astype(np.float64)
    # votes/r is mean votes-per-record into each cell; smooth and renormalise to k.
    q = (votes + alpha) / (r + alpha * n_cells)
    q *= k / q.sum()
    return q


def sample_mmd(X, Y, gamma=None, max_n=1500, seed=0):
    """Two-sample RBF-kernel MMD between point sets ``X`` and ``Y`` -- the actual
    distribution distance between the private (member) records and the auxiliary
    records in embedding space.

    ``MMD^2 = mean k(X,X) + mean k(Y,Y) - 2 mean k(X,Y)`` with the RBF kernel
    ``k(a,b) = exp(-gamma ||a-b||^2)``. ``gamma`` defaults to the inverse median
    squared pairwise distance over the pooled sample (the median heuristic). Each
    set is subsampled to ``max_n`` rows for tractability. Returns ``sqrt(max(.,0))``
    (a distance: 0 iff the two samples are indistinguishable under the kernel).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return float("nan")
    if X.shape[0] > max_n:
        X = X[rng.choice(X.shape[0], max_n, replace=False)]
    if Y.shape[0] > max_n:
        Y = Y[rng.choice(Y.shape[0], max_n, replace=False)]

    def _d2(A, B):
        return np.clip(np.einsum("ij,ij->i", A, A)[:, None]
                       + np.einsum("ij,ij->i", B, B)[None, :]
                       - 2.0 * (A @ B.T), 0.0, None)

    dxx, dyy, dxy = _d2(X, X), _d2(Y, Y), _d2(X, Y)
    if gamma is None:
        pooled = np.concatenate([dxx[np.triu_indices(len(X), 1)],
                                 dyy[np.triu_indices(len(Y), 1)], dxy.ravel()])
        med = float(np.median(pooled)) if pooled.size else 1.0
        gamma = 1.0 / med if med > 0 else 1.0
    mmd2 = (np.exp(-gamma * dxx).mean() + np.exp(-gamma * dyy).mean()
            - 2.0 * np.exp(-gamma * dxy).mean())
    return float(np.sqrt(max(float(mmd2), 0.0)))


def occupancy_mmd(p, q, cell_features, gamma=None):
    """Kernel MMD between two occupancy distributions ``p`` and ``q`` defined on the
    SAME synthetic-cell support, with an RBF kernel over the cell embeddings.

    ``MMD^2 = (p - q)^T K (p - q)``  with  ``K_ij = exp(-gamma * ||F_i - F_j||^2)``.
    Unlike cosine / TVD (which treat cells as exchangeable coordinates), this is
    geometry-aware: occupancy mass on *nearby* cells counts as similar, so it
    measures how far the private vote histogram sits from the aux-induced null in
    the embedding space the histogram actually lives in. ``gamma`` defaults to the
    inverse median squared pairwise cell distance (the median heuristic). Returns
    ``sqrt(max(MMD^2, 0))`` (a distance: larger = the two distributions are farther).
    """
    F = np.asarray(cell_features, dtype=np.float64)
    n = F.shape[0]
    if n == 0:
        return float("nan")
    sq = np.einsum("ij,ij->i", F, F)
    d2 = np.clip(sq[:, None] + sq[None, :] - 2.0 * (F @ F.T), 0.0, None)
    if gamma is None:
        iu = np.triu_indices(n, k=1)
        med = float(np.median(d2[iu])) if iu[0].size else 1.0
        gamma = 1.0 / med if med > 0 else 1.0
    K = np.exp(-gamma * d2)
    diff = np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)
    return float(np.sqrt(max(float(diff @ K @ diff), 0.0)))


# --------------------------------------------------------------------------- #
# Per-iteration log-likelihood ratios:  log P(obs | member) - log P(obs | non)
# --------------------------------------------------------------------------- #
def llr_pure(count, mu0, mu1, dispersion=1.0):
    """Pure-count regime. ``count`` is the integer vote count in the cell the
    query lands in. Under H0 (non-member) it is ``NB(mu0)``; under H1 (member) it
    is ``1 + NB(mu1)`` (x's own vote plus the others), where ``NB`` has
    ``var = dispersion*mean`` (``dispersion == 1`` -> Poisson).

    ``count == 0`` returns ``-inf`` -- the exact non-membership certificate: a
    member would necessarily have voted for this cell.
    """
    c = int(round(count))
    if c <= 0:
        return NEG_INF
    return float(_count_logpmf(c - 1, mu1, dispersion)
                 - _count_logpmf(c, mu0, dispersion))


def _obs_loglik(y, ks, sigma, threshold, censored=True):
    """log P(observed noised value ``y`` | true integer count ``ks``).

    censored=True  (text pipeline):  stores ``max(k + noise, threshold) - threshold``,
                   so ``y > 0`` is a Gaussian density and ``y == 0`` is the
                   left-censored mass ``P(k + noise <= threshold)``.
    censored=False (tabular pipeline, ``pe/dp/gaussian.py``):  stores
                   ``k + noise`` with no clipping, so every observed ``y`` (incl.
                   negative) is a plain Gaussian density.
    """
    ks = np.asarray(ks, dtype=np.float64)
    if not censored:
        return norm.logpdf(y, loc=ks, scale=sigma)
    if y > 0:
        return norm.logpdf(y + threshold, loc=ks, scale=sigma)
    return norm.logcdf((threshold - ks) / sigma)


def llr_noised(y, mu0, mu1, sigma, threshold=0.0, censored=True, dispersion=1.0):
    """Noised-count regime. Marginalise the unknown true count ``k`` over the
    null (``NB(mu0)``) and member-shifted (``1 + NB(mu1)``) priors, each seen
    through the (clipped or plain) Gaussian observation model (``dispersion == 1``
    -> Poisson priors, the original behaviour).
    """
    if sigma <= 0:
        return llr_pure(max(y, 0.0), mu0, mu1, dispersion=dispersion)
    hi = max(mu0, mu1, y) + 12.0 * np.sqrt(max(mu0, mu1, 1.0)) + 20.0
    ks = np.arange(0, int(np.ceil(hi)) + 1)
    obs = _obs_loglik(y, ks, sigma, threshold, censored=censored)
    log_pri0 = _count_logpmf(ks, mu0, dispersion)
    log_pri1 = np.full(ks.shape, NEG_INF)
    log_pri1[1:] = _count_logpmf(ks[1:] - 1, mu1, dispersion)
    logP0 = logsumexp(log_pri0 + obs)
    logP1 = logsumexp(log_pri1 + obs)
    return float(logP1 - logP0)


def llr_noised_vec(y, mu0, mu1, sigma, select_tau=None):
    """Vectorised, uncensored noised LLR over a batch of queries (each with its own
    ``y``, ``mu0``, ``mu1``), sharing one integer count grid. Equivalent to calling
    :func:`llr_noised` (``censored=False``) per element, but fast enough for the
    ablation. ``select_tau`` optionally conditions the observation on the cell being
    *selected* (``count >= tau``, the round's selection threshold), i.e. divides the
    Gaussian by its right-tail mass ``P(Y >= tau | k)`` -- a selection-bias-aware
    likelihood for the rank rounds (we only ever observe high-count survivors).
    Returns an ``(m,)`` array.
    """
    y = np.asarray(y, dtype=np.float64)
    mu0 = np.asarray(mu0, dtype=np.float64)
    mu1 = np.asarray(mu1, dtype=np.float64)
    if sigma <= 0:
        return np.array([llr_pure(max(yi, 0.0), m0, m1)
                         for yi, m0, m1 in zip(y, mu0, mu1)], dtype=np.float64)
    hi = max(float(mu0.max()), float(mu1.max()), float(y.max()))
    hi += 12.0 * np.sqrt(max(hi, 1.0)) + 20.0
    ks = np.arange(0, int(np.ceil(hi)) + 1)                      # (K,)
    obs = norm.logpdf(y[:, None], loc=ks[None, :], scale=sigma)  # (m, K)
    if select_tau is not None:
        obs = obs - norm.logsf((select_tau - ks[None, :]) / sigma)
    log_pri0 = poisson.logpmf(ks[None, :], mu0[:, None])         # (m, K)
    log_pri1 = np.full((y.shape[0], ks.shape[0]), NEG_INF)
    log_pri1[:, 1:] = poisson.logpmf(ks[None, 1:] - 1, mu1[:, None])
    logP0 = logsumexp(log_pri0 + obs, axis=1)
    logP1 = logsumexp(log_pri1 + obs, axis=1)
    out = logP1 - logP0
    # Guard the rare underflow case (both priors negligible -> -inf minus -inf).
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Per-record scoring across iterations
# --------------------------------------------------------------------------- #
def score_records(query_embeddings, iterations, mode="L2", regime="pure",
                  gen_k=1, sigma=0.0, threshold=0.0, ref_alpha=1.0,
                  censored=True, soft_tau=0.0, dispersion=1.0,
                  occupancy_cache=None):
    """Score a batch of candidate records (one class) against all iterations.

    ``query_embeddings`` : ``(m, d)`` embeddings of the candidates being tested.
    ``iterations``       : list of dicts, one per usable PE iteration, each with
                           ``cell_features`` ``(n, d)``, ``counts`` ``(n,)``
                           (clean for pure / noised for noised regime),
                           ``reference_features`` ``(r, d)`` (in-distribution,
                           non-private), and ``n_private`` (records in this class).

    Returns the aggregated membership score per record (sum of per-iteration
    LLRs; ``-inf`` if certified non-member in the pure regime).
    """
    m = query_embeddings.shape[0]
    total = np.zeros(m, dtype=np.float64)

    # Naive baseline: sum the raw vote count of the cell each query lands in,
    # with no density normalization.
    if regime == "raw":
        for it in iterations:
            cells = nearest_cell(query_embeddings, it["cell_features"],
                                 mode=mode, k=1)
            total += np.asarray(it["counts"], dtype=np.float64)[cells]
        return total

    for it_i, it in enumerate(iterations):
        F = it["cell_features"]
        counts = it["counts"]
        N = it["n_private"]
        if occupancy_cache is not None and it_i in occupancy_cache:
            q = occupancy_cache[it_i]
        else:
            q = cell_occupancy(it["reference_features"], F, mode=mode,
                               k=gen_k, alpha=ref_alpha, soft_tau=soft_tau)
            if occupancy_cache is not None:
                occupancy_cache[it_i] = q
        # With k-NN voting (gen_k>1) a member casts its +1 into EACH of its gen_k
        # nearest cells, so read all of them: the per-record LLR is the sum over
        # those cells, and an empty cell among them is still a non-membership
        # certificate.  gen_k==1 reduces exactly to the single-cell scorer.
        cells = nearest_cell(query_embeddings, F, mode=mode, k=gen_k)
        if gen_k == 1:
            cells = cells[:, None]
        for r in range(m):
            if total[r] == NEG_INF:
                continue
            ell_sum = 0.0
            certified = False
            for j in cells[r]:
                mu0 = N * q[j]
                mu1 = (N - 1) * q[j]
                if regime == "pure":
                    ell = llr_pure(counts[j], mu0, mu1, dispersion=dispersion)
                else:
                    ell = llr_noised(counts[j], mu0, mu1, sigma=sigma,
                                     threshold=threshold, censored=censored,
                                     dispersion=dispersion)
                if ell == NEG_INF:
                    certified = True
                    break
                ell_sum += ell
            total[r] = NEG_INF if certified else total[r] + ell_sum
    return total
