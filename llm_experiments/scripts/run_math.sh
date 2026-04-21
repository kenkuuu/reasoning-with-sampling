#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# RESULTS_DIR の解決:
#   コンテナ外: スクリプトの2階層上（プロジェクトルート）/results
#   Singularity コンテナ内: /workspace は読み取り専用のため,
#     $SINGULARITY_BIND から llm_experiments のホスト側パスを逆引きして results を導出
if [ -n "${SINGULARITY_NAME:-}" ]; then
    _HOST_LLM=$(printenv SINGULARITY_BIND 2>/dev/null | tr ',' '\n' | \
        awk -F: '$2=="/workspace/llm_experiments"{print $1}')
    if [ -n "${_HOST_LLM}" ]; then
        _DEFAULT_RESULTS="$(realpath -m "${_HOST_LLM}/../results")"
    else
        # bind 情報が取れない場合は $HOME 以下に書き込む
        _DEFAULT_RESULTS="${HOME}/reasoning-with-sampling/results"
    fi
else
    _DEFAULT_RESULTS="$(realpath -m "${SCRIPT_DIR}/../../results")"
fi
RESULTS_DIR="${RESULTS_DIR:-${_DEFAULT_RESULTS}}"

MODEL="${1:-qwen_math}"
MCMC_STEPS="${2:-10}"
TEMPERATURE="${3:-0.25}"

mkdir -p "${RESULTS_DIR}"

run_shard() {
    local gpu=$1
    local batch_idx=$2
    local seed=$3
    echo "GPU${gpu}: batch_idx=${batch_idx} seed=${seed}"
    CUDA_VISIBLE_DEVICES=${gpu} python "${SCRIPT_DIR}/../power_samp_math.py" \
            --batch_idx="${batch_idx}" \
            --mcmc_steps="${MCMC_STEPS}" \
            --temperature="${TEMPERATURE}" \
            --seed="${seed}" \
            --model="${MODEL}" \
            --save_str="${RESULTS_DIR}" \
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
