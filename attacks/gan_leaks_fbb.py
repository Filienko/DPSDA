"""GAN-Leaks "full black-box" (FBB) baseline, reimplemented directly against our
tabular embeddings -- a density/proximity attack that reads only the FINAL released
synthetic pool, no mechanism awareness (no vote counts, no selection history).

Ported from ``/home/daniilf/privacy/GAN-Leaks/attack_models/fbb.py`` (read-only
reference, not modified) + its scoring convention in
``attack_models/tools/eval_roc.py``: fit ``sklearn.neighbors.NearestNeighbors``
with ``K=5`` on the released synthetic samples ("generator output"), find each
query's K nearest neighbours, and use ONLY the distance to the single nearest
neighbour (``dist[:, 0]``) as the loss; the membership score is ``-dist[:, 0]``
(closer = more member-like). We drop the GAN-specific scaffolding (image loading,
argparse, latent-code recovery ``find_pred_z`` -- meaningless without a GAN) and
keep exactly the K-NN-distance attack model core.

Here "generator output" = the LAST round's released synthetic pool for each class
(the closest tabular analogue of "the GAN's generated.npz": the only thing an
attacker with FBB's threat model -- see only the final release -- would have).
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

K = 5


def fbb_scores(query_embeddings, gen_features, k=K):
    """GAN-Leaks FBB membership score: ``-distance to nearest generated sample``.

    ``query_embeddings``: (m, d) audit records (members + non-members).
    ``gen_features``:     (n, d) the released synthetic pool ("generator output").
    Returns an ``(m,)`` array, higher = more member-like (matches ``eval_roc.py``'s
    ``plot_roc(-pos_loss, -neg_loss)``).
    """
    k_eff = min(k, gen_features.shape[0])
    nn_obj = NearestNeighbors(n_neighbors=k_eff)
    nn_obj.fit(gen_features)
    dist, _idx = nn_obj.kneighbors(query_embeddings, k_eff)
    return -dist[:, 0]
