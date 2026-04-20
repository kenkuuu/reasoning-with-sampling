#!/bin/bash
#SBATCH --job-name=psamp_math
#SBATCH -t 0-23:59
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --array=0-39               # 5 shards × 8 seeds = 40 tasks

NUM_SHARDS=5
NUM_SEEDS=8
SEED=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
BATCH_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))

REPO_DIR=/home/kubota/reasoning-with-sampling
SIF=${REPO_DIR}/reasoning.sif
RESULTS_DIR=${REPO_DIR}/results

mkdir -p "${RESULTS_DIR}"

echo "Running shard BATCH_IDX=${BATCH_IDX} with SEED=${SEED} (task ${SLURM_ARRAY_TASK_ID})"

singularity exec --nv \
  --bind ~/.cache/huggingface:/root/.cache/huggingface \
  --bind "${RESULTS_DIR}":/workspace/results \
  "${SIF}" \
  python /workspace/llm_experiments/power_samp_math.py \
    --batch_idx="${BATCH_IDX}" \
    --mcmc_steps=10 \
    --temperature=0.25 \
    --seed="${SEED}" \
    --model=qwen_math \
    --save_str=/workspace/results
