#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -n "${SINGULARITY_NAME:-}" ]; then
    _DEFAULT_RESULTS="/workspace/results"
else
    _DEFAULT_RESULTS="$(realpath -m "${SCRIPT_DIR}/../../results")"
fi
RESULTS_DIR="${RESULTS_DIR:-${_DEFAULT_RESULTS}}"

MODEL="${1:-qwen_math}"
MCMC_STEPS="${2:-10}"
TEMPERATURE="${3:-0.25}"
BLOCK_SIZE="${4:-128}"
TAU_AVG="${5:-1.5}"
TAU_MAX="${6:-3.0}"
BETA="${7:-1.0}"

mkdir -p "${RESULTS_DIR}"

run_shard() {
    local gpu=$1
    local batch_idx=$2
    local seed=$3
    echo "GPU${gpu}: batch_idx=${batch_idx} seed=${seed}"
    cd "${SCRIPT_DIR}/.."
    CUDA_VISIBLE_DEVICES=${gpu} python power_samp_entropy_math.py \
            --batch_idx="${batch_idx}" \
            --mcmc_steps="${MCMC_STEPS}" \
            --temperature="${TEMPERATURE}" \
            --block_size="${BLOCK_SIZE}" \
            --tau_avg="${TAU_AVG}" \
            --tau_max="${TAU_MAX}" \
            --beta="${BETA}" \
            --seed="${seed}" \
            --model="${MODEL}" \
            --save_str="${RESULTS_DIR}" \
        > "${RESULTS_DIR}/eg_gpu${gpu}_batch${batch_idx}_seed${seed}.log" 2>&1
}

# 5 shards × 8 seeds = 40 タスクを 4GPU で順次処理
PIDS=()
TASK=0
for BATCH_IDX in 0 1 2 3 4; do
    for SEED in 0 1 2 3 4 5 6 7; do
        GPU=$(( TASK % 4 ))
        run_shard "${GPU}" "${BATCH_IDX}" "${SEED}" &
        PIDS+=($!)
        TASK=$(( TASK + 1 ))
        if (( TASK % 4 == 0 )); then
            wait "${PIDS[@]}"
            PIDS=()
        fi
    done
done
wait "${PIDS[@]}"

echo "All tasks done. Results in ${RESULTS_DIR}"
