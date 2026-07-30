"""Build a labelled membership-inference evaluation set for tabular PE.

Same design as ``aug-pe-baseline/attacks/audit_set.py``: members are the private
records actually fed to the generator; non-members are an in-distribution but
**non-private** holdout split (the test CSV). The only change is the data type ---
records are tabular feature rows (lists) keyed by integer label-id, instead of
text strings keyed by a ``label1\\tlabel2`` class string.
"""

import numpy as np

from pe.data import TabularCSV
from pe.constant.data import TABULAR_DATA_COLUMN_NAME, LABEL_ID_COLUMN_NAME


def _load_rows(csv_path, metadata_path):
    data = TabularCSV(csv_path=csv_path, metadata_path=metadata_path)
    rows = [list(r) for r in data.data_frame[TABULAR_DATA_COLUMN_NAME].tolist()]
    labels = list(data.data_frame[LABEL_ID_COLUMN_NAME].tolist())
    return rows, labels


# Fixed offset so the challenge/reference partition is a *separate* random stream
# from the member/non-member sampling, but still fully determined by ``seed``.
_REF_SPLIT_SALT = 987654321


def _ref_index_set(n, ref_holdout_frac, seed):
    """Deterministically reserve a ``ref_holdout_frac`` fraction of ``range(n)``
    for the reference/null model. Returns the set of reserved indices.

    Both ``build_audit_set`` (non-member challenges) and ``reference_rows_by_class``
    (D_ref) call this with the same ``n`` (rows of the same test CSV), ``seed`` and
    ``frac``, so they agree on the split and the two pools are disjoint: challenges
    are drawn from the complement, D_ref from this set.
    """
    if ref_holdout_frac <= 0 or n == 0:
        return set()
    rng = np.random.default_rng([seed, _REF_SPLIT_SALT])
    perm = rng.permutation(n)
    n_ref = int(round(ref_holdout_frac * n))
    return set(int(i) for i in perm[:n_ref])


def build_audit_set(member_csv, nonmember_csv, metadata_path,
                    n_members=500, n_nonmembers=500, seed=0, ref_holdout_frac=0.0):
    """Return ``(rows, classes, labels, sizes)`` with ``labels`` 1 = member, 0 = non.

    ``rows``    : list of feature lists (in ``feature_columns`` order).
    ``classes`` : integer label-ids (the PE histogram is computed per label-id).
    ``labels``  : 1 = member (private), 0 = non-member (holdout).
    ``sizes``   : dict of the full pool sizes (train/test/aux reservation) so the
                  caller can record exactly what the AUC was evaluated on.

    ``ref_holdout_frac`` > 0 reserves that fraction of the holdout CSV for the
    reference null model (see ``_ref_index_set``); non-member challenges are drawn
    only from the remaining, disjoint complement.
    """
    rng = np.random.default_rng(seed)
    m_rows, m_cls = _load_rows(member_csv, metadata_path)
    n_rows_all, n_cls_all = _load_rows(nonmember_csv, metadata_path)

    ref_set = _ref_index_set(len(n_rows_all), ref_holdout_frac, seed)
    member_set = set(map(tuple, m_rows))
    # Challenge non-members: not reserved for D_ref, and not coinciding with a
    # member (keep labels clean).
    keep = [i for i in range(len(n_rows_all))
            if i not in ref_set and tuple(n_rows_all[i]) not in member_set]
    n_rows = [n_rows_all[i] for i in keep]
    n_cls = [n_cls_all[i] for i in keep]

    m_idx = rng.choice(len(m_rows), size=min(n_members, len(m_rows)), replace=False)
    n_idx = rng.choice(len(n_rows), size=min(n_nonmembers, len(n_rows)), replace=False)

    rows = [m_rows[i] for i in m_idx] + [n_rows[i] for i in n_idx]
    classes = [m_cls[i] for i in m_idx] + [n_cls[i] for i in n_idx]
    labels = np.array([1] * len(m_idx) + [0] * len(n_idx), dtype=np.int64)
    sizes = {
        "train_total": len(m_rows),            # full private set (all are members)
        "test_total": len(n_rows_all),         # full holdout CSV
        "aux_reserved": len(ref_set),          # holdout rows reserved for D_ref
        "nonmember_pool": len(n_rows),         # holdout rows eligible as challenges
        "challenge_members": int(labels.sum()),
        "challenge_nonmembers": int((labels == 0).sum()),
    }
    return rows, np.asarray(classes), labels, sizes


def reference_rows_by_class(reference_csv, metadata_path, max_per_class=2000,
                            seed=0, ref_holdout_frac=0.0):
    """In-distribution non-private rows grouped by label-id, for the null model.

    ``ref_holdout_frac`` > 0 uses only the reserved reference partition (disjoint
    from the non-member challenges built by ``build_audit_set``)."""
    rows, labels = _load_rows(reference_csv, metadata_path)
    ref_set = _ref_index_set(len(rows), ref_holdout_frac, seed)
    rows, labels = np.asarray(rows, dtype=object), np.asarray(labels)
    rng = np.random.default_rng(seed)
    out = {}
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if ref_set:
            idx = np.array([i for i in idx if int(i) in ref_set], dtype=int)
        if idx.size == 0:
            continue
        if max_per_class > 0 and idx.size > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        out[int(cls)] = [list(r) for r in rows[idx]]
    return out
