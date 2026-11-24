#!/bin/bash
#SBATCH --job-name=sleep-ddp-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=haskell
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load pytorch/2.8.0

DATA_DIR=${DATA_DIR:-data/processed}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-0.001}
SEED=${SEED:-42}
RUN=${RUN:-ddp_gpu_${SLURM_JOB_ID}}
SCALE_LR=${SCALE_LR:-0}

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1

ARGS=(--data "$DATA_DIR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
      --lr "$LR" --threads 1 --seed "$SEED" --run-name "$RUN"
      --device cuda --backend nccl)
if [[ "$SCALE_LR" == "1" ]]; then
    ARGS+=(--scale-lr)
fi

echo "DDP GPU job $SLURM_JOB_ID on $SLURM_NODELIST: 2 GPUs, global batch=$BATCH_SIZE"
srun torch-exec torchrun --standalone --nproc_per_node=2 \
    -m src.train_ddp "${ARGS[@]}"
