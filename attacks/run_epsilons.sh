#!/usr/bin/env bash
# Full privacy sweep, end to end.
#
#   STAGE 1  generate synthetic data for every dataset at every epsilon in
#            {0.25, 0.5, 1, 5, 10, 100, inf}.  Runs that already have their final
#            checkpoint (eps 1 / 10 / inf were generated earlier) are SKIPPED, so
#            only the missing budgets (0.25, 0.5, 5, 100) are actually generated.
#   STAGE 2  run the histogram MIA on every run, WITH and WITHOUT D_ref.
#   STAGE 3  print the combined with/without-D_ref summary across all epsilons.
#
# Usage:
#   bash attacks/run_epsilons.sh                       # everything, default GPU
#   CUDA_VISIBLE_DEVICES=1 bash attacks/run_epsilons.sh   # pin one GPU
#   GENERATE_ONLY=1 bash attacks/run_epsilons.sh       # stage 1 only
#   ATTACK_ONLY=1   bash attacks/run_epsilons.sh       # stages 2-3 only
#
# Generation is heavy (each run is 15-30 PE iterations + a tabicl classifier per
# round); the four datasets x three new budgets run SEQUENTIALLY to avoid GPU OOM.
set -u
cd "$(dirname "$0")/.." || exit 1   # repo root (DPSDA_tab)

PY=.venv/bin/python
LOGDIR=results/tabular/_variant_logs
mkdir -p "$LOGDIR" results/tabular/_mia_logs

# dataset file-base -> results slug (hyphenated, as used in exp_folder + CSV names)
declare -A SLUG=(
  [adult]=adult
  [breast_cancer]=breast-cancer
  [artificial_characters]=artificial-characters
  [person_activity]=person-activity
)
# eps tag -> exp_folder suffix (must match make_eps_variants.py)
declare -A SUFFIX=(
  [eps0p25]=_eps0p25 [eps0p5]=_eps0p5 [eps1]=_eps1 [eps5]=_eps5 [eps10]="" [eps100]=_eps100 [epsinf]=_nonoise
)
TAGS=(epsinf eps100 eps10 eps5 eps1 eps0p5 eps0p25)
DATASETS=(adult breast_cancer artificial_characters person_activity)

if [ "${ATTACK_ONLY:-0}" != "1" ]; then
  echo "##### STAGE 1: generate synthetic data #####"
  "$PY" example/tabular/make_eps_variants.py
  for base in "${DATASETS[@]}"; do
    slug=${SLUG[$base]}
    niter=$(grep -oE 'num_iterations *= *[0-9]+' example/tabular/$base.py | grep -oE '[0-9]+' | head -1)
    last=$(printf "%09d" $((niter - 1)))
    for tag in "${TAGS[@]}"; do
      folder=results/tabular/${slug}_composite_population${SUFFIX[$tag]}
      if [ -d "$folder/checkpoint/$last" ]; then
        echo "[skip] $slug/$tag already complete ($folder)"
        continue
      fi
      script=example/tabular/variants/${base}_${tag}.py
      log=$LOGDIR/${slug}_${tag}.log
      echo "[$(date +%H:%M:%S)] GENERATE $slug/$tag -> $folder"
      "$PY" "$script" > "$log" 2>&1
      if [ $? -eq 0 ]; then echo "[$(date +%H:%M:%S)]   done"; else echo "[$(date +%H:%M:%S)]   FAILED -- see $log"; fi
    done
  done
fi

if [ "${GENERATE_ONLY:-0}" = "1" ]; then
  echo "GENERATE_ONLY set -- skipping attack."
  exit 0
fi

echo "##### STAGE 2: attack every run (with + without D_ref) #####"
bash attacks/run_attack_all.sh

echo "##### STAGE 3: summary #####"
"$PY" attacks/summarize_mia.py
