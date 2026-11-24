# Performance: bottleneck analysis, optimization, technology exploration

Owner: Salome (Performance Lead). Code: `src/profile_train.py`, `src/benchmark_io.py`,
`src/data/to_hdf5.py`, `src/data/preprocess_dask.py`, `slurm/profile.sbatch`,
`slurm/loader_workers.sh`, `slurm/io_benchmark.sbatch`, `slurm/preprocess_dask.sbatch`.

## Problem this section answers

Training speed is limited somewhere in the stack (I/O, data loading, CPU compute, or
GPU transfer). We profile first, then change one thing at a time, and always compare
accuracy / macro F1 against Nini’s `baseline_1cpu` row in `results/results.csv`.

## Dependencies already in the repo

| Piece | Who | Status |
|---|---|---|
| EDF → filtered 30 s epochs → `.npz` + subject splits | Dana | Done (`docs/DATA.md`) |
| 1D CNN + train/eval + class weights + shared CSV | Nini | Done (`src/train.py`, `src/models/`) |
| CUDA / AMP / CPU thread Slurm templates | Gabriel | On `gpu-amp` (this branch is based on it) |
| DDP / multi-node | Nikita | Not merged yet — not required for Salome’s jobs |

## 1. Bottleneck analysis (PyTorch Profiler)

Ran on the SciTech cluster (`slurm/profile.sbatch`), 1 CPU thread, `loader_workers=0`,
full processed data. Artifacts: `results/profiler/summary_20261123_014917.txt` (+ trace/ops).

**Finding:** once Dana’s `.npz` data is already in memory, training time is mostly
**compute**, not disk or the DataLoader. Top self-CPU time:

| Setting | Top op | Share of self CPU | Notes |
|---|---|---|---|
| 1 CPU, workers=0 | `Optimizer.step#Adam.step` | **54.4%** | Loss (`aten::cross_entropy_loss`) **35.0%**; convolutions a few %; DataLoader **&lt;1%** |
| 1 CPU, workers=4 | not re-profiled | — | Loader sweep (below) already showed workers do not help with 1 train thread |

So the expensive I/O is the **one-time** EDF parse (~165 s sequential). Re-doing that every
training job would dominate; saving once is the main I/O win. During training itself, the
Adam step and loss dominate.

## 2. Main optimization: preprocess once → fast format

Measured with `slurm/io_benchmark.sbatch` → `results/io_benchmark.csv` (23 Nov 2026).

| Path | What | Train-split load (or projected) | Speedup vs projected EDF |
|---|---|---|---|
| Re-parse EDF | MNE read + filter + epoch | projected from 8 nights | 1× (baseline) |
| `.npz` per night (Dana) | default store | **7.2 s** for 133,319 epochs | **6.5×** |
| Single HDF5 (Salome) | `sleep_edf.h5` | **9.7 s** for same split | **4.8×** |

**Keep `.npz` as the default.** HDF5 works and is still much faster than EDF, but was
slightly slower than the `.npz` directory load in this run.

Training accuracy with the saved formats stays in line with `baseline_1cpu`
(test acc **0.719**, macro F1 **0.684**, seed 42).

## 3. Technology exploration: Dask (+ HDF5)

| Technology | Role | Measured result |
|---|---|---|
| **Dask.distributed** | Parallel preprocess (4 workers) | **63.3 s** (`data/processed_dask/meta.json`) vs Dana’s sequential **164.9 s** → about **2.6×** wall-clock |
| **HDF5 (h5py)** | Single-file store | 4.8× vs projected EDF (see above); optional |
| PyTorch Profiler | Bottleneck diagnosis | Adam ~54%, loss ~35%, DataLoader &lt;1% |
| DataLoader `num_workers` | Prefetch | No clear throughput win with 1 train thread (section 4) |

## 4. DataLoader worker sensitivity

Ran `bash slurm/loader_workers.sh` with train **threads = 1**, batch 64, lr 1e-3, epochs 10,
seed 42. Rows in `results/results.csv`:

| Run | loader_workers | throughput (samples/s) | test acc | test macro F1 |
|---|---|---|---|---|
| baseline_1cpu | 0 | 2690.9 | 0.719 | 0.684 |
| loader_w0 | 0 | **3057.0** | 0.719 | 0.684 |
| loader_w1 | 1 | 2673.5 | 0.711 | 0.675 |
| loader_w2 | 2 | 2643.7 | 0.711 | 0.675 |
| loader_w4 | 4 | 2554.8 | 0.711 | 0.675 |
| loader_w8 | 8 | 2863.4 | 0.711 | 0.675 |

**Conclusion:** with one training thread, extra DataLoader workers do **not** clearly speed
things up (and can add process overhead). Accuracy stays close to the baseline.

```bash
python scripts/plot_salome.py
```

## 5. Technology matrix (for the report)

| Technology | Used for | Parallelism | Where it runs | Metric improved | Accuracy impact | Kept? |
|---|---|---|---|---|---|---|
| Compressed `.npz` (Dana) | Persist epochs once | I/O reduction | Shared FS | **6.5×** vs re-reading EDF | None (same tensors) | **Yes (default)** |
| HDF5 pack | Single-file reload | I/O reduction | Shared FS | **4.8×** vs EDF | None | Optional (npz faster here) |
| Dask.distributed | Parallel preprocess | Multi-process | CPU node | **63.3 s** vs 164.9 s | None (same algorithm) | **Yes for prep jobs** |
| DataLoader workers | Prefetch batches | Multi-process | Train job | no clear gain @ 1 train thread | ~unchanged | **No** (keep 0 with 1 thread) |
| PyTorch Profiler | Diagnosis only | — | Train job | — | — | **Yes** (methodology) |
| CUDA / AMP | Train compute | GPU | Gabriel | throughput | Check vs baseline | Gabriel’s section |
| PyTorch DDP | Multi-worker train | Multi-process / multi-node | Nikita | throughput, efficiency | Check vs baseline | Nikita’s section |

## 6. Reproducibility knobs Salome records

| Item | Value |
|---|---|
| Random seed | `42` (same as Dana’s split and Nini’s baseline) |
| Batch size / lr / epochs | `64` / `1e-3` / `10` |
| Train threads for loader study | `1` |
| Packages | `env/requirements-cpu.txt` + `env/requirements-perf.txt` (venv on cluster) |
| Results schema | `results/results.csv`, `results/io_benchmark.csv`, `results/profiler/` |
| Cluster | SciTech IE HPC; prep/profile/loader jobs on CPU partition (e.g. node `fortran`) |
| Key artifacts | profiler `summary_20261123_014917.txt`; Dask `preprocess_seconds: 63.3` |

## 7. Report subsection checklist

- [x] Bottleneck: Adam ~54%, loss ~35%, DataLoader &lt;1% once data is in RAM
- [x] Optimization: EDF vs `.npz` (6.5×) vs HDF5 (4.8×)
- [x] Technology exploration: Dask 63.3 s vs 164.9 s; matrix filled
- [ ] Loader-worker plot (`python scripts/plot_salome.py`) + commit figures
- [ ] Help write Conclusions (what to keep for EuroHPC scale-up)
