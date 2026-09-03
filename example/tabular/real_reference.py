"""Real-data reference point for the utility/fidelity tables in utility_sweep.py.

Computes, per dataset, using ONLY real data (no synthetic release, no PE):
  - "real utility": TabClassifier (tabicl) trained on the REAL train split,
    evaluated on the REAL test split -- the same callback/methodology
    utility_sweep.py already uses for train-on-synthetic, just with real data
    swapped in for the synthetic side.
  - "real-vs-real fidelity": FID and 1-/2-way TVD between the REAL train split
    (as "private") and the REAL test split (as the "synthetic" side passed to
    ComputeFID/ComputeTVD) -- the noise floor these metrics would report even
    with a perfect generator, since train and test are two disjoint real
    samples from the same distribution.

Extends results/tabular/_utility_logs/utility_summary.json with a top-level
"real_reference" section, one entry per dataset, matching the existing per-row
schema (clf_acc, clf_f1, fid, tvd_1way, tvd_2way) so it drops directly into the
same comparison.
"""

import json
import os

from pe.data import TabularCSV
from pe.embedding import TabularEmbedding
from pe.callback import ComputeFID
from pe.callback import TabClassifier
from pe.callback import ComputeTVD

BASE = "https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/heads/main/tabular/real"
DATASETS = ["adult", "artificial-characters", "breast-cancer", "person-activity"]
OUT = "results/tabular/_utility_logs/utility_summary.json"


def _metric_value(items, name_substr):
    for it in items:
        if name_substr in it.name:
            v = it.value
            return float(v[0]) if isinstance(v, list) else float(v)
    return None


def compute_one(slug):
    priv_data = TabularCSV(csv_path=f"{BASE}/{slug}/{slug}_train.csv",
                           metadata_path=f"{BASE}/{slug}/{slug}_metadata.json")
    test_data = TabularCSV(csv_path=f"{BASE}/{slug}/{slug}_test.csv",
                           metadata_path=f"{BASE}/{slug}/{slug}_metadata.json")
    priv_info = priv_data.get_tab_info()
    embedding = TabularEmbedding(info=priv_info)

    # Real utility: train on real train, test on real test.
    clf = TabClassifier(test_data=test_data, model_name="tabicl", filter_criterion=None)
    clf_items = clf(priv_data)
    clf_acc = _metric_value(clf_items, "_test_acc")
    clf_f1 = _metric_value(clf_items, "_test_f1")

    # Real-vs-real fidelity: train split as "private", test split as the "synthetic" side.
    fid_cb = ComputeFID(priv_data=priv_data, embedding=embedding, filter_criterion=None)
    fid = _metric_value(fid_cb(test_data), "fid_")

    tvd1_cb = ComputeTVD(priv_data=priv_data, degree=1, filter_criterion=None)
    tvd1 = _metric_value(tvd1_cb(test_data), "1way-tvd")

    tvd2_cb = ComputeTVD(priv_data=priv_data, degree=2, filter_criterion=None)
    tvd2 = _metric_value(tvd2_cb(test_data), "2way-tvd")

    return {"clf_acc": clf_acc, "clf_f1": clf_f1, "fid": fid,
            "tvd_1way": tvd1, "tvd_2way": tvd2,
            "note": "train=real train split, eval side=real test split (disjoint real samples)"}


def main():
    real_reference = {}
    for slug in DATASETS:
        print(f"=== {slug} ===")
        r = compute_one(slug)
        real_reference[slug] = r
        print(f"  clf_acc={r['clf_acc']:.2f}  clf_f1={r['clf_f1']:.2f}  "
             f"fid={r['fid']:.4f}  tvd1={r['tvd_1way']:.4f}  tvd2={r['tvd_2way']:.4f}")

    with open(OUT) as f:
        summary = json.load(f)
    summary["real_reference"] = real_reference
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nUpdated {OUT} with real_reference section")


if __name__ == "__main__":
    main()
