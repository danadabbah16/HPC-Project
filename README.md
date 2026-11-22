# HPC-Project — Sleep-EDF stage classification

1D CNN sleep-stage classifier on PhysioNet Sleep-EDF, used as a vehicle for HPC
experiments (CPU threads, GPU/AMP, DDP, I/O optimization, profiling).

## Repo layout

```
src/           # Python package (data, model, train, profile, benchmarks)
slurm/         # Slurm job scripts and launch helpers
env/           # Pinned requirements
data/sample/   # Tiny committed sample (full data is git-ignored)
results/       # Shared CSV + logs + figures (generated)
docs/          # Design notes per role
scripts/       # Download + plotting helpers
```

## Roles

| Role | Lead | Start here |
|---|---|---|
| Data | Dana | `docs/DATA.md`, `scripts/download_data.sh`, `src/data/` |
| ML baseline | Nini | `src/models/`, `src/train.py`, `slurm/train_cpu.sbatch` |
| Neuro / validation | Jinin | (multi-channel / κ / interpretability — branch TBD) |
| GPU | Gabriel | `slurm/gpu_*.slurm`, `--amp` in `src/train.py` |
| Distributed | Nikita | DDP + EuroHPC plan (branch TBD) |
| Performance | Salome | `docs/PERFORMANCE.md`, `REPRODUCE.md` |

## Quick start (laptop, sample data)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements-cpu.txt
pip install -r env/requirements-perf.txt   # Salome extras: h5py, dask, pandas

python -m src.train --data data/sample --epochs 2 --run-name smoke_sample
python -m src.profile_train --data data/sample
```

## Cluster baseline (everyone compares to this)

Nini’s 1-CPU run is the reference row in `results/results.csv`
(`baseline_1cpu`: test acc ≈ 0.719, macro F1 ≈ 0.684, seed 42).

```bash
# After data/processed exists (see docs/DATA.md):
DATA_DIR=data/processed EPOCHS=10 RUN=baseline_1cpu sbatch slurm/train_cpu.sbatch
```

## Salome — Performance Lead commands

See **`REPRODUCE.md`** and **`docs/PERFORMANCE.md`** for the full checklist.

```bash
sbatch slurm/profile.sbatch                 # PyTorch Profiler
sbatch slurm/io_benchmark.sbatch            # EDF vs npz vs HDF5
sbatch slurm/preprocess_dask.sbatch         # Dask parallel preprocess
bash  slurm/loader_workers.sh               # workers 0,1,2,4,8
python scripts/plot_salome.py               # figures → results/figures/
```

## Shared results format

Every training job **appends one row** to `results/results.csv` (time, throughput,
accuracy, macro F1, resources). Do not overwrite other people’s rows. Plot with the
shared colours in `scripts/plot_style.py`.

## Seeds and defaults

| Knob | Default |
|---|---|
| Subject split seed | 42 |
| Train seed | 42 |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Epochs (full runs) | 10 |
| Channel / filter | EEG Fpz-Cz, 0.3–35 Hz (see `docs/DATA.md`) |
