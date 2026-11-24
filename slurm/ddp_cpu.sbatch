#!/bin/bash
#SBATCH --job-name=sleep-ddp-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
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
RUN=${RUN:-ddp_cpu_${SLURM_JOB_ID}}
WORKERS=${WORKERS:-1}
SCALE_LR=${SCALE_LR:-0}
MASTER_PORT=${MASTER_PORT:-29500}

if (( WORKERS < 1 || WORKERS > 4 )); then
    echo "WORKERS must be 1, 2, or 4" >&2
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

echo "DDP CPU job $SLURM_JOB_ID on $SLURM_NODELIST: workers=$WORKERS, batch=$BATCH_SIZE"
srun --ntasks="$WORKERS" --ntasks-per-node="$WORKERS" \
    python -m src.train_ddp "${ARGS[@]}"

