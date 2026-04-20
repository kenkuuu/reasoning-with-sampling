#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SIF="${REPO_DIR}/reasoning.sif"
RESULTS_DIR="${REPO_DIR}/results"

MODEL="${1:-qwen_math}"
MCMC_STEPS="${2:-10}"
TEMPERATURE="${3:-0.25}"

mkdir -p "${RESULTS_DIR}"

run_shard() {
    local gpu=$1
    local batch_idx=$2
    local seed=$3
    echo "GPU${gpu}: batch_idx=${batch_idx} seed=${seed}"
    CUDA_VISIBLE_DEVICES=${gpu} singularity exec --nv \
        --bind ~/.cache/huggingface:/root/.cache/huggingface \
        --bind "${REPO_DIR}/llm_experiments":/workspace/llm_experiments \
        --bind "${RESULTS_DIR}":/workspace/results \
        "${SIF}" \
        python /workspace/llm_experiments/power_samp_math.py \
            --batch_idx="${batch_idx}" \
            --mcmc_steps="${MCMC_STEPS}" \
            --temperature="${TEMPERATURE}" \
            --seed="${seed}" \
            --model="${MODEL}" \
            --save_str=/workspace/results \
        > "${RESULTS_DIR}/gpu${gpu}_batch${batch_idx}_seed${seed}.log" 2>&1
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
        # 4 タスクごとに待機（同時実行数を GPU 数に合わせる）
        if (( TASK % 4 == 0 )); then
            wait "${PIDS[@]}"
            PIDS=()
        fi
    done
done
wait "${PIDS[@]}"

echo "All tasks done. Results in ${RESULTS_DIR}"
