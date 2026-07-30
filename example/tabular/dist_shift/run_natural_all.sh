#!/usr/bin/env bash
# Full natural-shift sweep: for each variant tag, run PE at epsilon=10 on the
# member set, then run the histogram MIA (with and without an auxiliary D_ref)
# using aux_{tag} as the non-member + reference holdout. AUC / utility land in
# each run's results folder and _mia_logs/.
set -u
cd "$(dirname "$0")/../../.." || exit 1     # repo root
PY=.venv/bin/python
DATA=example/tabular/dist_shift/data
NAT=$DATA/natural
META=$DATA/metadata.json
ROOT=results/tabular/dist_shift
LOG=$ROOT/_logs
mkdir -p "$LOG"

TAGS=${TAGS:-"iid q50 q35 q25 q18"}
SKIP_PE=${SKIP_PE:-0}        # 1 = reuse existing checkpoints, only (re)run the MIA

for tag in $TAGS; do
  exp=$ROOT/natural_$tag
  if [ "$SKIP_PE" != 1 ]; then
    echo "[$(date +%H:%M:%S)] === PE natural_$tag ==="
    $PY -m example.tabular.dist_shift.run_pe_shift --natural "$tag" \
        > "$LOG/pe_$tag.log" 2>&1 && echo "  PE done" || { echo "  PE FAILED"; continue; }
  fi

  cp="$exp/checkpoint"
  # epsilon=10; read THIS run's OWN noise_multiplier (it depends on N via delta)
  # straight from the log the DP accountant wrote -- don't hardcode a constant.
  NM=$(grep -hoE 'noise_multiplier=[0-9.eE+-]+' "$exp/log.txt" 2>/dev/null | tail -1 | cut -d= -f2)
  NM=${NM:-1.7105092023506527}
  for ref in ref noref; do
    flag=""; out="$exp/mia_report_ref.json"
    [ "$ref" = noref ] && { flag="--no_ref"; out="$exp/mia_report_noref.json"; }
    echo "[$(date +%H:%M:%S)]   MIA $tag/$ref"
    $PY -m attacks.run_mia \
        --checkpoint_folder "$cp" \
        --train_csv "$NAT/members_$tag.csv" \
        --test_csv  "$NAT/aux_$tag.csv" \
        --metadata  "$META" \
        --noise_multiplier "$NM" --regime both $flag \
        --ref_holdout_frac 0.5 --out_json "$out" \
        > "$LOG/mia_${tag}_${ref}.log" 2>&1 \
        && echo "    -> $out" || echo "    MIA FAILED ($ref) see $LOG/mia_${tag}_${ref}.log"
  done
done
echo "[$(date +%H:%M:%S)] natural sweep complete."
