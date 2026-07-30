"""Build covariate-shifted copies of the artificial-characters PRIVATE train set.

The membership-inference study needs a single, interpretable knob for "how
different is the private (member) distribution from the auxiliary / holdout
(non-member + reference) distribution". We create that knob by applying a uniform
covariate shift of magnitude ``delta`` (in per-feature standard-deviation units)
to the private training features, while leaving the test/aux split untouched:

    x'_f = x_f + delta * sigma_f        (sigma_f = std of feature f on real train)

Integer columns are rounded back to integers so the shifted CSV still respects
the dataset's declared column types. delta = 0 reproduces the matched-distribution
baseline (the classical MIA assumption); larger delta drives the private data
progressively out of the aux distribution.

Downloads the real artificial-characters files once into ``data/`` and writes, for
each delta, ``data/train_shift{tag}.csv``. The fixed ``test.csv`` and
``metadata.json`` are copied through unchanged.
"""

import json
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BASE = ("https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/heads/"
        "main/tabular/real/artificial-characters/")

# delta (in sigma units)  ->  filename tag
DELTAS = {0.0: "0p0", 0.5: "0p5", 1.0: "1p0", 2.0: "2p0", 3.0: "3p0"}


def _download(name, dest):
    if not os.path.exists(dest):
        urllib.request.urlretrieve(BASE + name, dest)
        print(f"  downloaded {name}")
    return dest


def main():
    os.makedirs(DATA, exist_ok=True)
    train_path = _download("artificial-characters_train.csv",
                           os.path.join(DATA, "train.csv"))
    _download("artificial-characters_test.csv", os.path.join(DATA, "test.csv"))
    meta_path = _download("artificial-characters_metadata.json",
                          os.path.join(DATA, "metadata.json"))

    meta = json.load(open(meta_path))
    feat_cols = meta["int_columns"] + meta["float_columns"]
    int_cols = set(meta["int_columns"])

    train = pd.read_csv(train_path)
    sigma = {c: float(train[c].std(ddof=0)) for c in feat_cols}
    print("per-feature sigma:", {c: round(v, 3) for c, v in sigma.items()})

    for delta, tag in DELTAS.items():
        shifted = train.copy()
        for c in feat_cols:
            shifted[c] = shifted[c] + delta * sigma[c]
            if c in int_cols:
                shifted[c] = np.rint(shifted[c]).astype(int)
        out = os.path.join(DATA, f"train_shift{tag}.csv")
        shifted.to_csv(out, index=False)
        print(f"wrote {os.path.relpath(out, HERE)} (delta={delta} sigma)")


if __name__ == "__main__":
    main()
