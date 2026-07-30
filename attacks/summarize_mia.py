"""Collect every mia_report_{ref,noref}.json under results/tabular/ into one table.

For each dataset x privacy budget, prints the noised-regime AUC and TPR@1%FPR
both WITH an auxiliary reference set D_ref and WITHOUT it (uniform null), plus the
delta. Writes the combined grid to results/tabular/_mia_logs/mia_summary.json.
"""

import glob
import json
import os

ROOT = "results/tabular"

BUDGET_FROM_SUFFIX = {"": "eps10", "_nonoise": "inf"}
REGIME = "noised"  # the realistic attacker (released DP_HISTOGRAM)


def budget_of(run):
    suffix = run.split("_composite_population", 1)[1]
    return BUDGET_FROM_SUFFIX.get(suffix, suffix.lstrip("_") or "eps10")


def eps_of(budget):
    """Numeric epsilon for ordering (inf = no privacy)."""
    if budget in ("inf", "nonoise"):
        return float("inf")
    return float(budget.replace("eps", "").replace("p", "."))


def load(path):
    with open(path) as f:
        rep = json.load(f)
    ar = rep.get("regimes", {}).get(REGIME, {}).get("all_rounds", {})
    return (ar.get("auc"), ar.get("tpr@fpr=0.01"),
            rep.get("args", {}).get("noise_multiplier", 0.0),
            rep.get("data_sizes", {}))


def main():
    grid = {}
    for path in sorted(glob.glob(f"{ROOT}/*_composite_population*/mia_report_ref.json")):
        run = os.path.basename(os.path.dirname(path))
        slug = run.split("_composite_population", 1)[0]
        budget = budget_of(run)
        key = (slug, budget)
        auc_r, tpr_r, nm, sizes = load(path)
        noref_path = path.replace("mia_report_ref.json", "mia_report_noref.json")
        auc_n, tpr_n = (load(noref_path)[:2] if os.path.exists(noref_path) else (None, None))
        grid[key] = {"dataset": slug, "budget": budget, "noise_multiplier": nm,
                     "auc_ref": auc_r, "tpr_ref": tpr_r,
                     "auc_noref": auc_n, "tpr_noref": tpr_n,
                     # data the AUC was evaluated on (sizes are per-dataset, same
                     # across epsilons): train(members) / test / aux(D_ref) and the
                     # member/non-member challenge counts actually scored.
                     "n_train": sizes.get("train_total"),
                     "n_test": sizes.get("test_total"),
                     "n_aux": sizes.get("aux_used"),
                     "n_challenge_members": sizes.get("challenge_members"),
                     "n_challenge_nonmembers": sizes.get("challenge_nonmembers")}

    # Order weakest->strongest privacy within each dataset (inf, 100, 10, ... 0.25).
    rows = sorted(grid.values(), key=lambda r: (r["dataset"], -eps_of(r["budget"])))

    def f(x):
        return "  n/a " if x is None else f"{x:6.4f}"

    def d(a, b):
        return "   n/a " if a is None or b is None else f"{a - b:+7.4f}"

    hdr = (f"{'dataset':22} {'budget':8} {'sigma':>9} | "
           f"{'AUC ref':>8} {'AUC noref':>9} {'dAUC':>8} | "
           f"{'T@1 ref':>8} {'T@1 noref':>9} {'dT@1':>8}")
    print(f"noised regime (released DP_HISTOGRAM) -- with vs without auxiliary D_ref")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['dataset']:22} {r['budget']:8} {r['noise_multiplier']:9.4g} | "
              f"{f(r['auc_ref'])} {f(r['auc_noref'])} {d(r['auc_ref'], r['auc_noref'])} | "
              f"{f(r['tpr_ref'])} {f(r['tpr_noref'])} {d(r['tpr_ref'], r['tpr_noref'])}")

    # Per-dataset evaluation sizes (identical across epsilons within a dataset).
    print("\ndata the AUC is evaluated on (per dataset):")
    shdr = (f"{'dataset':22} {'train(memb)':>11} {'test':>7} {'aux(Dref)':>9} | "
            f"{'chal memb':>9} {'chal non':>9}")
    print(shdr)
    print("-" * len(shdr))
    seen = set()
    for r in rows:
        if r["dataset"] in seen:
            continue
        seen.add(r["dataset"])
        print(f"{r['dataset']:22} {str(r['n_train']):>11} {str(r['n_test']):>7} "
              f"{str(r['n_aux']):>9} | {str(r['n_challenge_members']):>9} "
              f"{str(r['n_challenge_nonmembers']):>9}")

    out = f"{ROOT}/_mia_logs/mia_summary.json"
    with open(out, "w") as fp:
        json.dump({"regime": REGIME, "rows": rows}, fp, indent=2, default=str)
    print(f"\nWrote {out}  ({len(rows)} dataset x budget cells)")


if __name__ == "__main__":
    main()
