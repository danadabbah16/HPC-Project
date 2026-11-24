#!/bin/bash
#SBATCH --job-name=sleep-ddp-cpu-2node
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

DATA_DIR=${DATA_DIR:-data/processed}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-0.001}
SEED=${SEED:-42}
RUN=${RUN:-ddp_cpu_2node_${SLURM_JOB_ID}}
WORKERS=${WORKERS:-2}
SCALE_LR=${SCALE_LR:-0}
MASTER_PORT=${MASTER_PORT:-29500}

if (( WORKERS < 2 || WORKERS > 4 || WORKERS % 2 != 0)); then
    echo "WORKERS must be 2 or 4 for this 2-node template" >&2
    exit 2
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT

ARGS=(--data "$DATA_DIR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
      --lr "$LR" --threads 1 --seed "$SEED" --run-name "$RUN"
      --backend gloo --master-port "$MASTER_PORT")
if [[ "$SCALE_LR" == "1" ]]; then
    ARGS+=(--scale-lr)
fi

echo "DDP 2-node CPU job $SLURM_JOB_ID on $SLURM_NODELIST: workers=$WORKERS"
srun --ntasks="$WORKERS" --ntasks-per-node=2 \
    python -m src.train_ddp "${ARGS[@]}"

