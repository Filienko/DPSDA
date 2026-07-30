#!/usr/bin/env bash
# Run all four tabular examples at epsilon=1 and with no noise.
#
# Safe to launch while the epsilon=10 runs are going: every variant writes to its
# own results/tabular/<name>_composite_population_{eps1,nonoise}/ folder, so it
# never touches or resumes the epsilon=10 checkpoints. Runs sequentially to avoid
# thrashing the GPU against the in-flight epsilon=10 jobs.
#
# Usage:
#   bash example/tabular/run_eps1_and_nonoise.sh
# Optional: restrict to one GPU so the epsilon=10 jobs keep the other:
#   CUDA_VISIBLE_DEVICES=1 bash example/tabular/run_eps1_and_nonoise.sh

set -u
cd "$(dirname "$0")/../.." || exit 1   # repo root (DPSDA_tab)

PY=.venv/bin/python
LOGDIR=results/tabular/_variant_logs
mkdir -p "$LOGDIR"

# (Re)generate the variant scripts from the current examples.
"$PY" example/tabular/make_variants.py

DATASETS=(adult breast_cancer artificial_characters person_activity)
SETTINGS=(eps1 nonoise)

for setting in "${SETTINGS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    script="example/tabular/variants/${ds}_${setting}.py"
    log="$LOGDIR/${ds}_${setting}.log"
    echo "[$(date +%H:%M:%S)] running $script -> $log"
    "$PY" "$script" > "$log" 2>&1
    status=$?
    if [ $status -eq 0 ]; then
      echo "[$(date +%H:%M:%S)]   done ($script)"
    else
      echo "[$(date +%H:%M:%S)]   FAILED ($script, exit $status) -- see $log"
    fi
  done
done

echo "All variant runs finished."
