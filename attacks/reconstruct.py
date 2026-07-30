"""Reconstruct per-iteration synthetic candidate sets + released vote counts from
a saved tabular-PE run.

This is the tabular analogue of ``aug-pe-baseline/attacks/reconstruct.py``. The
text pipeline scattered the signal across two files per iteration --- the vote
counts in ``{t}/count_class/{class}.csv`` and the candidate texts in
``{t-1}_all/samples.csv`` --- and had to re-embed the texts to recover the cell
geometry. The tabular pipeline (``pe/callback/common/save_checkpoints.py``)
instead persists *everything* in one place: each iteration's checkpoint
``{checkpoint}/{t:09d}/data_frame.pkl`` already contains, for every voted cell,
both its embedding (``PE.EMBEDDING.TabularEmbedding`` / ``PE.LOOKAHEAD_EMBEDDING``)
and its released counts (``PE.CLEAN_HISTOGRAM`` and the noised
``PE.DP_HISTOGRAM``). So reconstruction is just: load each checkpoint, keep the
rows that carry a histogram value (the candidate cells that were voted on that
round, kept by the ``keep_selected`` population), and group them by label-id.

Requires the run to have used ``lookahead_degree=0`` (so the indexed embedding is
the plain per-sample embedding, recomputable from the public synthetic row),
which the tabular examples do.
"""

import os
import numpy as np
import pandas as pd

from pe.constant.data import (
    CLEAN_HISTOGRAM_COLUMN_NAME,
    DP_HISTOGRAM_COLUMN_NAME,
    POST_PROCESSED_DP_HISTOGRAM_COLUMN_NAME,
    LOOKAHEAD_EMBEDDING_COLUMN_NAME,
    LABEL_ID_COLUMN_NAME,
    TABULAR_DATA_COLUMN_NAME,
)

EMBEDDING_COLUMN = "PE.EMBEDDING.TabularEmbedding"


def discover_iterations(checkpoint_folder):
    """Return sorted iteration indices ``t`` that have a ``data_frame.pkl``."""
    out = []
    if not os.path.isdir(checkpoint_folder):
        return out
    for name in os.listdir(checkpoint_folder):
        p = os.path.join(checkpoint_folder, name, "data_frame.pkl")
        if os.path.isfile(p):
            try:
                out.append(int(name))
            except ValueError:
                continue
    return sorted(out)


def _cell_embeddings(sub, embed_fn):
    """Per-cell embeddings: reuse the stored embedding if present, else recompute
    deterministically from the raw tabular row via ``embed_fn``."""
    col = EMBEDDING_COLUMN if EMBEDDING_COLUMN in sub.columns else LOOKAHEAD_EMBEDDING_COLUMN_NAME
    if col in sub.columns and sub[col].apply(lambda v: isinstance(v, (list, np.ndarray))).all():
        return np.stack(sub[col].values).astype(np.float32)
    return embed_fn(sub[TABULAR_DATA_COLUMN_NAME].tolist())


def reconstruct(checkpoint_folder, embed_fn, reference_features_by_class,
                n_private_by_class, start_t=1, max_iters=None, use_clean=False):
    """Build the per-iteration, per-class data consumed by ``score_records``.

    ``embed_fn``                   : ``rows -> (n, d)`` (only used as a fallback if
                                     a checkpoint lacks stored embeddings).
    ``reference_features_by_class``: ``{label_id: (r, d) array}`` in-distribution,
                                     non-private embeddings for the null model.
    ``n_private_by_class``         : ``{label_id: N}`` private record count per class
                                     (public dataset class sizes).
    ``use_clean``                  : ``False`` (default) reads the *released noised*
                                     counts ``PE.DP_HISTOGRAM`` --- what an attacker
                                     actually sees; ``True`` reads the clean
                                     histogram (pure-regime / audit upper bound).

    Returns ``{label_id: [iteration_dict, ...]}`` where each iteration_dict has
    ``cell_features``, ``counts``, ``reference_features`` and ``n_private``.
    """
    iters = [t for t in discover_iterations(checkpoint_folder) if t >= start_t]
    if max_iters is not None:
        iters = iters[:max_iters]

    by_class = {}
    for t in iters:
        df = pd.read_pickle(os.path.join(checkpoint_folder, f"{t:09d}", "data_frame.pkl"))
        if CLEAN_HISTOGRAM_COLUMN_NAME not in df.columns:
            continue  # initial / pop-without-keep_selected rounds carry no counts
        # Rows that actually carry a vote count are the candidate cells voted on
        # this round (kept by the keep_selected population).
        voted = df[df[CLEAN_HISTOGRAM_COLUMN_NAME].notna()]
        if len(voted) == 0:
            continue

        noised_col = (DP_HISTOGRAM_COLUMN_NAME if DP_HISTOGRAM_COLUMN_NAME in df.columns
                      else POST_PROCESSED_DP_HISTOGRAM_COLUMN_NAME)
        for cls, sub in voted.groupby(LABEL_ID_COLUMN_NAME):
            cls = int(cls)
            counts = (sub[CLEAN_HISTOGRAM_COLUMN_NAME] if use_clean
                      else sub[noised_col]).to_numpy(dtype=np.float64)
            cell_features = _cell_embeddings(sub, embed_fn)
            by_class.setdefault(cls, []).append({
                "cell_features": cell_features,
                "counts": counts,
                "reference_features": reference_features_by_class.get(
                    cls, np.empty((0, cell_features.shape[1]), dtype=np.float32)),
                "n_private": max(int(n_private_by_class.get(cls, 1)), 1),
            })
    return by_class
