#!/usr/bin/env bash
# Run the histogram MIA / privacy audit on every completed tabular run
# (each dataset at every available privacy budget: eps=10, eps=1, no-noise).
#
# For each results/tabular/<dataset>_composite_population[_suffix]/ that has a
# checkpoint, we derive the dataset slug, read noise_multiplier from its log.txt,
# and run attacks.run_mia with --regime both (raw + pure + noised). Reports are
# written next to each run as mia_report_both.json and logged under _mia_logs/.
set -u
cd "$(dirname "$0")/.." || exit 1   # repo root (DPSDA_tab)

PY=.venv/bin/python
BASE="https://raw.githubusercontent.com/toan-vt/cloud-data-store/refs/heads/main/tabular"
LOGDIR=results/tabular/_mia_logs
mkdir -p "$LOGDIR"

for d in results/tabular/*_composite_population*/; do
  d=${d%/}
  cp="$d/checkpoint"
  last=$(ls "$cp" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  [ -n "$last" ] || { echo "skip (no checkpoint): $d"; continue; }

  run=$(basename "$d")                      # e.g. breast-cancer_composite_population_eps1
  slug=${run%%_composite_population*}        # e.g. breast-cancer
  suffix=${run#*_composite_population}       # "", "_eps1", "_nonoise"
  case "$suffix" in
    _eps1)    budget=eps1 ;;
    _nonoise) budget=nonoise ;;
    "")       budget=eps10 ;;
    *)        budget=${suffix#_} ;;
  esac

  nm=$(grep -hoE 'noise_multiplier=[0-9.eE+-]+' "$d/log.txt" 2>/dev/null | tail -1 | cut -d= -f2)
  nm=${nm:-0}

  # Two threat models: WITH an auxiliary reference set D_ref (test holdout -> the
  # Poisson occupancy null), and WITHOUT it (--no_ref -> uniform occupancy null).
  for ref in ref noref; do
    flag=""; out="$d/mia_report_ref.json"
    if [ "$ref" = "noref" ]; then flag="--no_ref"; out="$d/mia_report_noref.json"; fi
    log="$LOGDIR/${slug}_${budget}_${ref}.log"
    echo "[$(date +%H:%M:%S)] === $slug/$budget/$ref  noise_multiplier=$nm  (last ckpt $last) ==="
    "$PY" -m attacks.run_mia \
        --checkpoint_folder "$cp" \
        --train_csv "$BASE/${slug}_train.csv" \
        --test_csv  "$BASE/${slug}_test.csv" \
        --metadata  "$BASE/${slug}_metadata.json" \
        --noise_multiplier "$nm" \
        --regime both $flag \
        --ref_holdout_frac "${REF_HOLDOUT_FRAC:-0.5}" \
        --out_json "$out" \
        > "$log" 2>&1
    status=$?
    if [ $status -eq 0 ]; then
      echo "[$(date +%H:%M:%S)]   done -> $out"
    else
      echo "[$(date +%H:%M:%S)]   FAILED (exit $status) -- see $log"
    fi
  done
done

echo "All MIA runs finished."
