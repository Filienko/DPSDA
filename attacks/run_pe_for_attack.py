"""Generate a tabular PE run whose checkpoints an MIA can be scored against.

The ``example/tabular/*.py`` runners cannot be used for this directly. They pin the
dataset URLs to a layout that no longer exists, they use ``backend="torch"``, and
they attach ten callbacks -- two ``TabClassifier`` (TabICL) fits and two
``ComputeFID`` per iteration -- that dominate runtime while writing nothing an
attack reads. The attack only ever reads
``{exp_folder}/checkpoint/{t:09d}/data_frame.pkl``, which ``SaveCheckpoints``
writes on its own.

Everything that affects the *release* is copied verbatim from
``example/tabular/artificial_characters.py`` so the resulting numbers stay
comparable to the ones in ``running_summary.md``: the mutation-rate schedule, the
L2 nearest-neighbour histogram with ``lookahead_degree=0``, the
sample-then-rank ``CompositePopulation``, ``num_samples_schedule``, and epsilon.

Three deliberate differences, none of which change what is released:

* ``backend="sklearn"`` instead of ``"torch"`` -- identical exact search
  (``pe/histogram/nearest_neighbor_backend/sklearn.py``), and it drops the torch
  dependency.
* ``callbacks=[SaveCheckpoints(...)]``, ``loggers=[]`` -- metrics only.
* the dataset URLs point at the current nested layout,
  ``tabular/real/<slug>/<slug>_train.csv``. The flat
  ``tabular/<slug>_train.csv`` paths still hard-coded across
  ``example/tabular/`` (and all 32 files in ``example/tabular/variants/``) now
  return 404; ``example/tabular/dist_shift/make_shift_data.py`` is the only script
  in the repo already using the new one.

``SaveCheckpoints`` is imported from its defining module rather than from
``pe.callback``, because ``pe/callback/__init__.py`` pulls in ``ComputeFID`` ->
``import cleanfid.fid`` at module scope.

Logging stays enabled: ``attacks/strong_mia.py`` scrapes the noise multiplier out
of ``log.txt``.

Example
-------
    python -m attacks.run_pe_for_attack --dataset artificial-characters
    python -m attacks.run_pe_for_attack --dataset artificial-characters --natural iid
"""

import argparse
import os
import sys

import numpy as np

from pe.data import TabularCSV
from pe.embedding import TabularEmbedding
from pe.api import TabularAPI
from pe.histogram import NearestNeighbors
from pe.population import PEPopulation, CompositePopulation
from pe.runner import PE
from pe.callback.common.save_checkpoints import SaveCheckpoints
from pe.logging import setup_logging

BASE = ("https://raw.githubusercontent.com/toan-vt/cloud-data-store/"
        "refs/heads/main/tabular/real")
DATASETS = ("artificial-characters", "breast-cancer", "adult", "person-activity")
NATURAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "example", "tabular", "dist_shift", "data")


def dataset_paths(slug, natural=None):
    """``(train_csv, test_csv, metadata)`` for a dataset.

    ``natural=<tag>`` uses the committed matched/shifted splits under
    ``example/tabular/dist_shift/data/natural/`` instead of the remote CSVs.
    Those are artificial-characters only. The ``iid`` tag is the one where members
    and non-members come from the same PCA-1 band (mean embedding gap 0.015 vs
    0.42-0.77 for the shifted tags), so it is the setting in which a raised AUC
    cannot be the distribution confound of ``running_summary.md`` Exp B/C.
    """
    if natural:
        d = os.path.join(NATURAL_DIR, "natural")
        return (os.path.join(d, f"members_{natural}.csv"),
                os.path.join(d, f"aux_{natural}.csv"),
                os.path.join(NATURAL_DIR, "metadata.json"))
    return (f"{BASE}/{slug}/{slug}_train.csv",
            f"{BASE}/{slug}/{slug}_test.csv",
            f"{BASE}/{slug}/{slug}_metadata.json")


def make_inout_split(train_csv, metadata, out_dir, frac=0.5, seed=0):
    """Deduplicate the private CSV and split it into disjoint IN / OUT halves.

    The default audit protocol takes members from ``train.csv`` and non-members
    from ``test.csv``, which has two problems on these datasets. First, any
    train/test distribution difference is readable without touching the histogram
    at all -- ``running_summary.md`` Exp B/C measures that confound but never
    removes it. Second, on artificial-characters **72% of test rows exactly
    duplicate a training feature row**; ``audit_set.py:67-68`` drops them to keep
    the labels honest, which leaves only ~425 usable non-members out of 1533.

    Splitting one deduplicated pool at random fixes both: the two halves are
    identically distributed by construction, and no non-member can be a copy of a
    member. PE is then trained on the IN half only, so membership is exactly the
    split. Dedup happens *before* the split, otherwise a row appearing twice in
    the private data could put one copy on each side and produce a "non-member"
    that votes in the member's cell.

    Returns ``(members_csv, nonmembers_csv)``.
    """
    import pandas as pd
    df = pd.read_csv(train_csv)
    meta = TabularCSV(csv_path=train_csv, metadata_path=metadata).metadata
    feat = list(meta["feature_columns"])
    n_raw = len(df)
    df = df.drop_duplicates(subset=feat).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_in = int(round(frac * len(df)))
    os.makedirs(out_dir, exist_ok=True)
    members = os.path.join(out_dir, "audit_members.csv")
    nonmembers = os.path.join(out_dir, "audit_nonmembers.csv")
    df.iloc[perm[:n_in]].to_csv(members, index=False)
    df.iloc[perm[n_in:]].to_csv(nonmembers, index=False)
    print(f"in/out split : {n_raw} rows -> {len(df)} distinct -> "
          f"{n_in} members / {len(df) - n_in} non-members (seed {seed})")
    return members, nonmembers


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default="artificial-characters", choices=DATASETS)
    p.add_argument("--inout_split", action="store_true",
                   help="deduplicate the private CSV and train on a random half; "
                        "the held-out half becomes the non-member pool. Removes "
                        "the train/test distribution confound and the "
                        "member-duplicate contamination.")
    p.add_argument("--split_frac", type=float, default=0.5)
    p.add_argument("--natural", default="",
                   help="use example/tabular/dist_shift/data/natural/{members,aux}_<tag>.csv "
                        "(tags: iid q50 q35 q25 q18); artificial-characters only")
    p.add_argument("--epsilon", type=float, default=10.0)
    p.add_argument("--num_iterations", type=int, default=15)
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp_folder", default="")
    args = p.parse_args(argv)

    exp_folder = args.exp_folder or (
        f"results/tabular/dist_shift/natural_{args.natural}" if args.natural
        else f"results/tabular/{args.dataset}_composite_population")
    if args.seed:
        exp_folder = f"{exp_folder}_seed{args.seed}"
    os.makedirs(exp_folder, exist_ok=True)
    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))
    np.random.seed(args.seed)

    train_csv, _test_csv, metadata = dataset_paths(args.dataset, args.natural or None)
    print(f"dataset      : {args.dataset}{' (natural ' + args.natural + ')' if args.natural else ''}")
    if args.inout_split:
        train_csv, _ = make_inout_split(train_csv, metadata, exp_folder,
                                        frac=args.split_frac, seed=args.seed)
    print(f"private csv  : {train_csv}")
    print(f"exp_folder   : {exp_folder}")

    priv_data = TabularCSV(csv_path=train_csv, metadata_path=metadata)
    priv_info = priv_data.get_tab_info()
    n_private = len(priv_data.data_frame)
    print(f"private rows : {n_private}, classes: {len(priv_data.metadata.label_info)}")

    api = TabularAPI(
        info=priv_info,
        mutation_rate_init=0.5,
        mutation_rate_final=0.01,
        decay_type="polynomial",
        gamma=0.2,
        num_iterations=args.num_iterations,
    )
    embedding = TabularEmbedding(info=priv_info)
    histogram = NearestNeighbors(
        embedding=embedding, mode="L2", lookahead_degree=0, backend="sklearn",
    )
    # Rounds 1-4: multinomial resampling, survivors discarded (so those checkpoints
    # carry no counts -- only the parent multiplicity that strong_mia reads).
    population1 = PEPopulation(
        api=api, initial_variation_api_fold=0, next_variation_api_fold=1,
        keep_selected=False, selection_mode="sample", histogram_threshold=0,
    )
    # Rounds 5+: deterministic top-k on unclipped noisy counts, survivors kept.
    population2 = PEPopulation(
        api=api, initial_variation_api_fold=3, next_variation_api_fold=3,
        keep_selected=True, selection_mode="rank",
    )
    n_pop1 = min(5, args.num_iterations)
    population = CompositePopulation(
        populations=[population1] * n_pop1 + [population2] * (args.num_iterations - n_pop1))

    delta = 1.0 / n_private / np.log(n_private)
    pe_runner = PE(
        priv_data=priv_data,
        population=population,
        histogram=histogram,
        callbacks=[SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))],
        loggers=[],
    )
    pe_runner.run(
        num_samples_schedule=[args.num_samples] * args.num_iterations,
        delta=delta,
        epsilon=args.epsilon,
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )
    print(f"\ndone -> {exp_folder}/checkpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
