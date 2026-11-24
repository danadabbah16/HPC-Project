# Distributed training and scaling

Owner: Nikita (Distributed Lead). Code: `src/train_ddp.py`, `slurm/ddp_*.sbatch`.

This note describes the PyTorch DistributedDataParallel (DDP) implementation,
how to run the scaling experiments, and how to use their measured results to
plan a larger EuroHPC allocation. The DDP entry point is separate from
`src/train.py`; single-process runs are unchanged.

## Summary

- DDP supports CPU with Gloo and CUDA with NCCL.
- CPU templates cover 1, 2 and 4 workers on one node, and 2 or 4 workers across
  two nodes.
- The default is strong scaling: `--batch-size 64` is the global batch, divided
  evenly among workers.
- The script records elapsed training time, speedup, parallel efficiency,
  throughput, validation/test metrics, and an estimate of communication time.
- The SciTech cluster was checked from its `rust` login host on 2026-11-24.
  Node resources and modules are listed below; DDP performance is still
  unmeasured.

## Cluster environment

The following inventory comes from `hostname`, `sinfo -o "%P %G %N"`, and
`module avail pytorch` on the cluster login shell:

| Node | Partitions | CPUs | Memory (MB) | GPUs |
|---|---|---:|---:|---|
| `fortran` | `cpu`, `gpu`, `interactive` | 512 | 756000 | 2 x `rtx6000bb` |
| `haskell` | `cpu`, `gpu`, `interactive` | 512 | 406528 | 2 x `rtx6000ada` |
| `julia` | `cpu`, `gpu`, `interactive` | 512 | 515002 | 1 x `rtx6000bb` |

The CPU partition reports all three nodes, 1536 CPUs total, `MaxNodes=UNLIMITED`,
and `MaxTime=7-00:00:00`. These are partition settings; account or QoS policy
may still constrain an individual user's job.

The `pytorch/2.8.0` module provides a CUDA 12.8 Apptainer container, sets
`PYTORCH_GPU_SIF`, and documents the `torch-exec` wrapper. It is intended for
GPU nodes. Its instructions specifically require `NCCL_P2P_DISABLE=1` for
multi-GPU DDP on `haskell`. The module list also included
`PyTorch/2.1.2-foss-2023a`; its CPU/GPU support has not been checked.

## Terms used in this note

| Term | Meaning |
|---|---|
| Worker / rank | One training process. Each process has a rank from 0 to `world_size - 1`. |
| World size | Total number of DDP training processes, possibly across multiple nodes. |
| DDP | PyTorch DistributedDataParallel; synchronizes model gradients between workers during training. |
| Gloo | PyTorch distributed backend used here for CPU training. |
| NCCL | NVIDIA's collective-communication backend used here for CUDA training. |
| Strong scaling | Keep the total dataset and global batch fixed while increasing workers. |
| Speedup | One-worker training time divided by the parallel training time. |
| Efficiency | Speedup divided by the number of workers. Ideal efficiency is 1.0 (100%). |
| Communication fraction | Estimated synchronization time divided by the measured backward-pass time in a short benchmark. |
| EuroHPC | European high-performance computing resources; a larger target allocation than the project cluster. |

## Implementation

`src/train_ddp.py` reads rank and world-size values from the process environment
provided by Slurm. For multi-node runs it uses the first hostname in
`SLURM_JOB_NODELIST` as the rendezvous address, unless `MASTER_ADDR` is already
set. `MASTER_PORT` defaults to 29500.

Training data is sharded using `DistributedSampler`; `set_epoch` reshuffles the
shards for each epoch. Each worker loads its own copy of each dataset split into
memory, so request sufficient memory per task. The current dataset implementation
loads the full split rather than streaming batches from disk.

The global batch size must be divisible by the number of workers. With the
default global batch of 64, each rank gets a local batch of 64, 32, or 16 for
1, 2, or 4 workers respectively. The default learning rate stays at 0.001.
`--scale-lr` is an optional weak-scaling variant that multiplies the learning
rate by the worker count; do not mix it with the default strong-scaling results.

Each rank uses seed `seed + rank`. Rank 0 evaluates the model, saves the best
checkpoint, and writes the result row. The CSV row includes the worker and node
counts, time, speedup, efficiency, throughput, accuracy, macro F1, and
communication estimate.

## Measuring communication

Before training, the script compares one backward pass with DDP synchronization
disabled (`no_sync`) against one normal DDP backward pass. Their elapsed-time
difference is stored as `comm_time_s`; the difference divided by the normal
backward-pass time is stored as `comm_fraction`.

This is a short estimate, not a profiler breakdown of all communication over
the full training run. It is most useful for comparing worker counts under the
same hardware and batch settings. A negative timing difference is clipped to
zero because the measurement is susceptible to noise.

## How to reproduce

Connect to the team's SciTech Slurm cluster and open the project checkout there.
The `rust` host shown above is the login host observed in this session. The
repository does not record the external SSH hostname, account name, VPN
requirement, or SSH-key setup; get those from the course/team cluster
instructions or administrator. The local branch must also be made available
in the cluster checkout before submitting jobs.

The CPU Slurm templates use the project's `.venv`; confirm that it exists on
the cluster and contains the required packages. The `pytorch/2.8.0` module is
for GPU nodes and should not be used as the CPU environment. The verified node
inventory and partition settings are listed above. Check resources and module
details before submitting:

```bash
sinfo -N -o "%N %P %c %m %G %t"
scontrol show partition cpu
module show pytorch/2.8.0
```

After confirming the CPU node count, memory request, Python environment, and
two-node allocation policy, run from the project root:

```bash
export DATA_DIR=data/processed
export EPOCHS=10
export BATCH_SIZE=64
export LR=0.001
export SEED=42

for w in 1 2 4; do
  WORKERS=$w RUN="ddp_cpu_w${w}_r1" sbatch slurm/ddp_cpu.sbatch
done

WORKERS=2 RUN=ddp_cpu_2node_w2_r1 sbatch slurm/ddp_cpu_2node.sbatch
WORKERS=4 RUN=ddp_cpu_2node_w4_r1 sbatch slurm/ddp_cpu_2node.sbatch
```

Submit at least two or three repetitions per configuration to estimate timing
variation. Confirm that the runs use the canonical 153-recording Dataset A and
the same train/validation/test splits as the 1-worker reference. The CPU
templates request 12 GB of memory; adjust only after checking cluster policy
and actual per-task memory requirements.

The GPU template uses two GPUs on `haskell`, loads the cluster's
`pytorch/2.8.0` module, and launches two DDP workers with `torchrun`. The module
sets `PYTORCH_GPU_SIF` and provides `torch-exec`; the script also sets the
documented `NCCL_P2P_DISABLE=1` workaround. GPU training still needs a short
smoke test to verify container imports, Slurm GPU binding, and DDP startup
before a full 10-epoch run.

Plot available DDP CSV rows with:

```bash
python scripts/plot_ddp.py --input results/results.csv --output results/plots
```

## Results to collect

The following table is intentionally blank until cluster jobs complete. Use
repeated-run means (and report the spread) for timing and efficiency.

| Nodes | Workers | Train time (s) | Speedup | Efficiency | Throughput (samples/s) | Test accuracy | Test macro F1 | Communication fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | — | — | — | — | — | — | — |
| 1 | 2 | — | — | — | — | — | — | — |
| 1 | 4 | — | — | — | — | — | — | — |
| 2 | 2 | — | — | — | — | — | — | — |
| 2 | 4 | — | — | — | — | — | — | — |

Use the single-worker result from the same dataset and device class as the
speedup reference. The script defaults to the existing Dataset A baseline
training time (495.44 s); pass `--baseline-train-time` when a better matched
reference has been measured.

## EuroHPC scale-up plan

The project model and dataset are relatively small, while every worker currently
loads its own complete copy of the dataset. Scaling may therefore be limited by
data memory and communication latency, especially for CPU Gloo. A larger
allocation should be justified by observed efficiency rather than extrapolated
from the ideal speedup line.

Use the measured scaling results to estimate a target:

```text
speedup(workers) = single_worker_time / measured_time(workers)
efficiency(workers) = speedup(workers) / workers
estimated_core_hours = target_runtime_seconds * allocated_cores / 3600
estimated_nodes = ceil(target_workers / usable_workers_per_node)
```

### Staged request

1. **Validation:** use 1 node for the 1/2/4-worker comparison and short smoke
   tests before long runs.
2. **Inter-node check:** use 2 nodes at 2 and 4 workers to measure network
   synchronization overhead and confirm multi-node launch works.
3. **Scale-up decision:** consider a 4-8 node pilot only if measured efficiency
   remains useful at 2 nodes and memory/resource requirements are known. If
   efficiency falls sharply, investigate data loading and communication before
   requesting more nodes.
4. **Production estimate:** calculate node count and core/GPU hours from the
   target dataset, epoch count, measured throughput, and measured efficiency.
   Include a clearly stated allowance for queueing, setup, and repeat runs.

Before making a numerical EuroHPC request, record available CPUs/GPUs per node,
the interconnect, permitted node counts, memory limits, and the actual target
workload. The SciTech CPU/GPU node counts and advertised resources are now
recorded, but the interconnect, effective per-user limits, and all scaling
results remain unknown, so no node-hour estimate is claimed.

## References

- PyTorch documentation, [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
- PyTorch documentation, [DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler).
