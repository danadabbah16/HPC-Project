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

```bash
# Local smoke test
python -m src.profile_train --data data/sample --loader-workers 0

# Cluster, same 1-CPU setup as the baseline
sbatch slurm/profile.sbatch
# With DataLoader workers:
WORKERS=4 sbatch --cpus-per-task=5 slurm/profile.sbatch
```

Outputs land in `results/profiler/`:

- `trace_*.json` — open in [Perfetto](https://ui.perfetto.dev) or `chrome://tracing`
- `summary_*.txt` — top ops table
- `ops_*.csv` — machine-readable top ops for the report

**Expected finding (hypothesis):** with `loader_workers=0` and data already in RAM
(after Dana’s `.npz` load), the hot path is CPU convolution / GEMM, not disk I/O.
The expensive disk work is the *one-time* EDF parse (~165 s sequential, see `docs/DATA.md`).
If someone re-ran preprocess every training job, that would dominate — which is why
the main optimization is “save once, reload fast”.

Fill in after the job:

| Setting | Top op | Share of self CPU | Notes |
|---|---|---|---|
| 1 CPU, workers=0 | _TBD_ | _TBD_ | |
| 1 CPU, workers=4 | _TBD_ | _TBD_ | |

## 2. Main optimization: preprocess once → fast format

| Path | What it does | Script |
|---|---|---|
| EDF every time | MNE read + filter + epoch (slow) | `src.data.preprocess` |
| `.npz` per night (Dana) | Already the project default | `data/processed/*.npz` |
| Single HDF5 (Salome) | One sequential read per split | `python -m src.data.to_hdf5` |

```bash
sbatch slurm/io_benchmark.sbatch
# or locally on sample (skips EDF):
python -m src.data.to_hdf5 --data data/sample
python -m src.benchmark_io --processed data/sample --skip-edf
```

Report the speedup of `.npz` / HDF5 vs projected full-train EDF cost from
`results/io_benchmark.csv`. Training accuracy must stay within noise of `baseline_1cpu`
(same seed 42, same epochs / batch / lr).

## 3. Technology exploration: Dask (+ HDF5)

| Technology | Role in this project | Why we tried it |
|---|---|---|
| **Dask.distributed** | Parallelise Dana’s per-recording preprocess | Embarrassingly parallel over 153 nights; compare to 164.9 s sequential |
| **HDF5 (h5py)** | Optional single-file training store | Fewer filesystem ops than 100+ `.npz` opens |
| PyTorch Profiler | Bottleneck diagnosis | Required for the report |
| DataLoader `num_workers` | Overlap host prep with compute | Sensitivity experiment below |

```bash
sbatch slurm/preprocess_dask.sbatch          # default 4 workers
N_WORKERS=8 sbatch --cpus-per-task=8 slurm/preprocess_dask.sbatch
```

Record `preprocess_seconds` from `data/processed_dask/meta.json` vs Dana’s 164.9 s.

## 4. DataLoader worker sensitivity

```bash
bash slurm/loader_workers.sh
# submits loader_w0 .. loader_w8, each appending to results/results.csv
```

Keep `THREADS=1` for the training loop so this is not confused with Gabriel’s
CPU-thread scaling. Plot with:

```bash
python scripts/plot_salome.py
```

## 5. Technology matrix (for the report)

| Technology | Used for | Parallelism | Where it runs | Metric improved | Accuracy impact | Kept? |
|---|---|---|---|---|---|---|
| Compressed `.npz` (Dana) | Persist epochs once | I/O reduction | Shared FS | data_load_s, preprocess avoided | None (same tensors) | **Yes (default)** |
| HDF5 pack | Single-file reload | I/O reduction | Shared FS | data_load_s | None if bit-identical load | Optional |
| Dask.distributed | Parallel preprocess | Multi-process | CPU node | preprocess wall time | None (same algorithm) | Yes for prep jobs |
| DataLoader workers | Prefetch batches | Multi-process | Train job | throughput / epoch time | Should be ~unchanged (seed 42) | Tune after plot |
| PyTorch Profiler | Diagnosis only | — | Train job | — | — | Yes (methodology) |
| CUDA / AMP | Train compute | GPU | Gabriel | throughput | Check vs baseline | Gabriel’s section |
| PyTorch DDP | Multi-worker train | Multi-process / multi-node | Nikita | throughput, efficiency | Check vs baseline | Nikita’s section |

## 6. Reproducibility knobs Salome records

| Item | Value |
|---|---|
| Random seed | `42` (same as Dana’s split and Nini’s baseline) |
| Batch size / lr / epochs | `64` / `1e-3` / `10` unless noted |
| Train threads for loader study | `1` |
| Packages | `env/requirements-cpu.txt` + `env/requirements-perf.txt` |
| Results schema | append-only `results/results.csv` (+ `results/io_benchmark.csv`) |
| Cluster modules (fill on site) | e.g. `pytorch/2.8.0` — copy from Gabriel’s methodology |

## 7. Report subsection checklist

- [ ] Bottleneck: paste top profiler ops + 1–2 sentences on what dominates
- [ ] Optimization: table of EDF vs `.npz` vs HDF5 times and speedup
- [ ] Technology exploration: Dask speedup + technology matrix table above
- [ ] Loader-worker plot + note that test macro F1 stays near baseline
- [ ] Help write Conclusions (what to keep for EuroHPC scale-up)
