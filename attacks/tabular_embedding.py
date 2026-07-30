"""Deterministic tabular embedding for the attack.

In the text attack the candidate / audit records were embedded with a learned
sentence encoder (``dpsda.feature_extractor.extract_features``). The tabular PE
pipeline instead uses :py:class:`pe.embedding.TabularEmbedding`, a *deterministic*
function of the raw feature row (min-max normalised numerics + weighted one-hot
categoricals). We reuse that exact embedding so the attacker's recomputed
nearest-neighbor assignment lands in the same space the histogram was built in.

``info`` (per-column categories / numeric bounds) is taken from the private data
exactly as the generator did (``TabularCSV.get_tab_info``); these bounds are
public knowledge about the dataset, not the private rows' values.
"""

import numpy as np

from pe.data import Data, TabularCSV
from pe.embedding import TabularEmbedding
from pe.constant.data import TABULAR_DATA_COLUMN_NAME, LABEL_ID_COLUMN_NAME


def load_private(csv_path, metadata_path):
    """Load the private dataset and return ``(TabularCSV, info)``."""
    priv = TabularCSV(csv_path=csv_path, metadata_path=metadata_path)
    return priv, priv.get_tab_info()


def make_embed_fn(priv_data, info):
    """Return ``embed(rows) -> (m, d) float32`` using the same TabularEmbedding
    the generator used. ``rows`` is an iterable of feature lists (in
    ``feature_columns`` order).
    """
    emb = TabularEmbedding(info=info)
    metadata = priv_data.metadata

    def embed(rows):
        rows = list(rows)
        if len(rows) == 0:
            # Determine dimensionality from a dummy single-row call.
            return np.empty((0, _emb_dim(emb, metadata)), dtype=np.float32)
        import pandas as pd
        df = pd.DataFrame(
            {
                TABULAR_DATA_COLUMN_NAME: [list(r) for r in rows],
                LABEL_ID_COLUMN_NAME: [0] * len(rows),
            }
        )
        d = Data(data_frame=df, metadata=metadata)
        d = emb.compute_embedding(d)
        return np.stack(d.data_frame[emb.column_name].values).astype(np.float32)

    return embed


def _emb_dim(emb, metadata):
    n_num = len(metadata["int_columns"]) + len(metadata["float_columns"])
    n_cat = sum(len(emb._info[c]["categories"]) for c in metadata["cat_columns"])
    return n_num + n_cat
