#!/usr/bin/env bash

# Goal 5: empirical-value sensitivity study for MICE-CPO.
#
# This is a one-factor-at-a-time study on the paper's SafetyPointGoal1-v0
# sensitivity environment. Every configuration is run with the same six seeds.
# The paper uses 1e7 steps; this project uses 7000000 as a documented
# compute-limited deviation. With steps_per_epoch=20000, each run has 350
# epochs.

set -euo pipefail

ENV_ID="SafetyPointGoal1-v0"
DEVICE="cpu"
VECTOR_ENVS="16"
TORCH_THREADS="1"
TOTAL_STEPS="7000000"
PARALLEL="1"
# Keep this at 1 for the first verification. Change it to 0 to launch training.
DRY_RUN="0"

# The paper reports six random seeds but does not publish their values.
SEEDS=(0 1 2 3 4 5)

# Baseline values from MICE.yaml.
BASE_COST_LIMIT=25
BASE_K_KNN=10
BASE_INTRINSIC_FACTOR=5.0

# Paper-grounded threshold values: 0, 15, and 25 (Figure 6c/6d).
COST_LIMITS=(0 15 25)

# The paper states that k=10 is the default and studies its sensitivity.
# These wider values are the project's extension of that analysis.
K_VALUES=(1 5 10 20 40)

# Exploratory implementation-level sensitivity of the intrinsic-cost scale.
# This is not the paper's fixed-constant ablation (3, 5, 15), which requires
# replacing the memory-generated intrinsic cost in the algorithm source.
INTRINSIC_FACTORS=(0.0 1.0 3.0 5.0 10.0 15.0)

run_one() {
  local label="$1"
  local seed="$2"
  local cost_limit="$3"
  local k_knn="$4"
  local intrinsic_factor="$5"

  local -a command=(
    python -m experiments.train_mice
    --algo MICE
    --env-id "$ENV_ID"
    --device "$DEVICE"
    --parallel "$PARALLEL"
    --vector-env-nums "$VECTOR_ENVS"
    --torch-threads "$TORCH_THREADS"
    --total-steps "$TOTAL_STEPS"
    --seed "$seed"
    --algo_cfgs:cost_limit "$cost_limit"
    --algo_cfgs:k_knn "$k_knn"
    --algo_cfgs:intrinsic_factor "$intrinsic_factor"
  )

  echo "[$label] seed=$seed cost_limit=$cost_limit k_knn=$k_knn intrinsic_factor=$intrinsic_factor"
  printf '  '
  printf '%q ' "${command[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "0" ]]; then
    "${command[@]}"
  fi
}

run_configuration() {
  local label="$1"
  local cost_limit="$2"
  local k_knn="$3"
  local intrinsic_factor="$4"

  for seed in "${SEEDS[@]}"; do
    run_one "$label" "$seed" "$cost_limit" "$k_knn" "$intrinsic_factor"
  done
}

echo "Goal 5 MICE-CPO sensitivity study"
echo "env=$ENV_ID device=$DEVICE vector_envs=$VECTOR_ENVS parallel=$PARALLEL torch_threads=$TORCH_THREADS"
echo "total_steps=$TOTAL_STEPS seeds=${SEEDS[*]} dry_run=$DRY_RUN"
echo "Each run uses 350 epochs when TOTAL_STEPS=7000000."

# Baseline: run once as its own labelled configuration.
run_configuration "baseline" "$BASE_COST_LIMIT" "$BASE_K_KNN" "$BASE_INTRINSIC_FACTOR"

# One-factor-at-a-time threshold study.
for value in "${COST_LIMITS[@]}"; do
  [[ "$value" == "$BASE_COST_LIMIT" ]] && continue
  run_configuration "cost_limit_${value}" "$value" "$BASE_K_KNN" "$BASE_INTRINSIC_FACTOR"
done

# One-factor-at-a-time KNN study.
for value in "${K_VALUES[@]}"; do
  [[ "$value" == "$BASE_K_KNN" ]] && continue
  run_configuration "k_knn_${value}" "$BASE_COST_LIMIT" "$value" "$BASE_INTRINSIC_FACTOR"
done

# One-factor-at-a-time intrinsic-factor study.
for value in "${INTRINSIC_FACTORS[@]}"; do
  [[ "$value" == "$BASE_INTRINSIC_FACTOR" ]] && continue
  run_configuration "intrinsic_factor_${value}" "$BASE_COST_LIMIT" "$BASE_K_KNN" "$value"
done

echo "Done."