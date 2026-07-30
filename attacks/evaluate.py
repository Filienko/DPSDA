"""Standard membership-inference metrics: ROC AUC, TPR at low FPR, and
certificate accounting for the exact pure-count non-membership certificate.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _finite_scores(scores):
    """Replace ``-inf`` (certified non-members) with a value below all finite
    scores so ordering-based metrics treat them as the most confident negatives.
    """
    scores = np.asarray(scores, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    floor = (finite.min() - 1.0) if finite.size else -1.0
    out = scores.copy()
    out[~np.isfinite(out)] = floor
    return out


def tpr_at_fpr(labels, scores, target_fpr):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def evaluate(labels, scores, fprs=(0.001, 0.01, 0.05)):
    """Return a metrics dict for membership scores (higher = more likely member)."""
    labels = np.asarray(labels)
    raw = np.asarray(scores, dtype=np.float64)
    s = _finite_scores(raw)

    certified = ~np.isfinite(raw)  # -inf == certified non-member
    out = {
        "auc": float(roc_auc_score(labels, s)),
        "n_members": int(labels.sum()),
        "n_nonmembers": int((labels == 0).sum()),
        # A member certified as non-member is a certificate error -- must be 0
        # in the pure k=1 regime.
        "certified_total": int(certified.sum()),
        "certified_members_FALSE": int(certified[labels == 1].sum()),
        "certified_nonmembers": int(certified[labels == 0].sum()),
        "nonmember_certify_rate": float(
            certified[labels == 0].mean()) if (labels == 0).any() else 0.0,
    }
    for f in fprs:
        out[f"tpr@fpr={f}"] = tpr_at_fpr(labels, s, f)
    return out


def format_report(metrics, title=""):
    lines = [f"=== {title} ===" if title else "==="]
    lines.append(f"members={metrics['n_members']} "
                 f"nonmembers={metrics['n_nonmembers']}")
    lines.append(f"AUC = {metrics['auc']:.4f}")
    for k in sorted(metrics):
        if k.startswith("tpr@"):
            lines.append(f"{k} = {metrics[k]:.4f}")
    lines.append(f"certified non-members (correct) = {metrics['certified_nonmembers']} "
                 f"({metrics['nonmember_certify_rate']:.3f} of non-members)")
    lines.append(f"certified members (ERRORS, must be 0) = "
                 f"{metrics['certified_members_FALSE']}")
    return "\n".join(lines)
