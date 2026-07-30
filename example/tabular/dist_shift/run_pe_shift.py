"""Run tabular Private Evolution at epsilon=10 on a covariate-shifted PRIVATE set.

This is example/tabular/artificial_characters.py with two changes:
  * the private train CSV is a local shifted variant (data/train_shift{tag}.csv),
    while the test CSV stays the fixed real holdout, and
  * paths are local (the upstream data store moved) and the exp_folder encodes
    the shift tag so each delta writes its own results/checkpoint tree.

Everything else -- iterations, sample schedule, composite population, callbacks,
epsilon=10 -- is identical to the base example, so the only independent variable
across runs is the private/aux distribution gap delta.

Usage:  python -m example.tabular.dist_shift.run_pe_shift --tag 1p0
"""

import argparse
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from pe.data import TabularCSV
from pe.logging import setup_logging
from pe.runner import PE
from pe.population import PEPopulation, CompositePopulation
from pe.api import TabularAPI
from pe.embedding import TabularEmbedding
from pe.histogram import NearestNeighbors
from pe.callback import (SaveCheckpoints, ComputeFID, TabClassifier,
                         SaveTabToCSV, ComputeTVD)
from pe.logger import CSVPrint, LogPrint
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

pd.options.mode.copy_on_write = True

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="artificial covariate-shift tag, e.g. 0p0, 3p0")
    ap.add_argument("--natural", help="natural-split tag, e.g. iid, q50, q18 "
                    "(uses data/natural/members_{tag}.csv + aux_{tag}.csv)")
    args = ap.parse_args()

    metadata = os.path.join(DATA, "metadata.json")
    if args.natural:
        nd = os.path.join(DATA, "natural")
        train_csv = os.path.join(nd, f"members_{args.natural}.csv")
        test_csv = os.path.join(nd, f"aux_{args.natural}.csv")
        exp_folder = f"results/tabular/dist_shift/natural_{args.natural}"
    else:
        train_csv = os.path.join(DATA, f"train_shift{args.tag}.csv")
        test_csv = os.path.join(DATA, "test.csv")
        exp_folder = f"results/tabular/dist_shift/artificial-characters_shift{args.tag}"
    assert os.path.exists(train_csv), train_csv
    load_dotenv()
    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    priv_data = TabularCSV(csv_path=train_csv, metadata_path=metadata)
    priv_info = priv_data.get_tab_info()
    test_data = TabularCSV(csv_path=test_csv, metadata_path=metadata)

    num_iterations = 15
    api = TabularAPI(info=priv_info, mutation_rate_init=0.5, mutation_rate_final=0.01,
                     decay_type="polynomial", gamma=0.2, num_iterations=num_iterations)
    embedding = TabularEmbedding(info=priv_info)
    histogram = NearestNeighbors(embedding=embedding, mode="L2",
                                 lookahead_degree=0, backend="torch")
    population1 = PEPopulation(api=api, initial_variation_api_fold=0,
                              next_variation_api_fold=1, keep_selected=False,
                              selection_mode="sample", histogram_threshold=0)
    population2 = PEPopulation(api=api, initial_variation_api_fold=3,
                              next_variation_api_fold=3, keep_selected=True,
                              selection_mode="rank")
    population = CompositePopulation(
        populations=[population1] * 5 + [population2] * (num_iterations - 5))

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    f = {VARIATION_API_FOLD_ID_COLUMN_NAME: -1}
    callbacks = [
        save_checkpoints,
        SaveTabToCSV(output_folder=os.path.join(exp_folder, "synthetic_tab")),
        ComputeFID(priv_data=priv_data, embedding=embedding, filter_criterion=f),
        TabClassifier(test_data=test_data, model_name="tabicl", filter_criterion=f),
        ComputeTVD(priv_data=priv_data, degree=1, filter_criterion=f),
        ComputeTVD(priv_data=priv_data, degree=2, filter_criterion=f),
        ComputeFID(priv_data=priv_data, embedding=embedding, filter_criterion=None),
        TabClassifier(test_data=test_data, model_name="tabicl", filter_criterion=None),
        ComputeTVD(priv_data=priv_data, degree=1, filter_criterion=None),
        ComputeTVD(priv_data=priv_data, degree=2, filter_criterion=None),
    ]

    num_private_samples = len(priv_data.data_frame)
    delta = 1.0 / num_private_samples / np.log(num_private_samples)

    pe_runner = PE(priv_data=priv_data, population=population, histogram=histogram,
                   callbacks=callbacks, loggers=[CSVPrint(output_folder=exp_folder),
                                                 LogPrint()])
    pe_runner.run(num_samples_schedule=[1000] * num_iterations, delta=delta,
                  epsilon=10.0, checkpoint_path=os.path.join(exp_folder, "checkpoint"))


if __name__ == "__main__":
    main()
