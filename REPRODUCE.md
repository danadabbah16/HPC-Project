# Reproduce (Salome / Performance Lead + shared setup)

This file records how to recreate Salome’s experiments and how to keep results
comparable with the rest of the team.

## 0. Environment

On the cluster (example — adjust to the modules Gabriel records):

```bash
git clone <repo-url> HPC-Project && cd HPC-Project
git checkout performance

python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements-cpu.txt
pip install -r env/requirements-perf.txt
```

Or with the site PyTorch module / Apptainer image (see Gabriel’s GPU scripts):

```bash
module purge
module load pytorch/2.8.0
# pip install h5py dask[distributed] pandas into a user env if not in the image
```

Record the exact versions you actually ran:

```bash
python -c "import torch,numpy,sklearn,mne,h5py,dask; print(torch.__version__, numpy.__version__)"
pip freeze > results/logs/pip_freeze_$(date +%Y%m%d).txt
```

## 1. Data (Dana — once per machine)

```bash
# Login node (needs network):
bash scripts/download_data.sh data/raw

# Compute node:
sbatch slurm/prepare_sleep_edf.sbatch
# → data/processed/*.npz + splits.json  (or use existing processed data)
```

Subject split seed **42**. Do not rebuild splits with another seed if you want
matching accuracy.

## 2. Reference baseline (Nini — already in CSV)

`results/results.csv` contains:

| run | device | threads | test_accuracy | test_macro_f1 |
|---|---|---|---|---|
| baseline_1cpu | cpu | 1 | 0.719 | 0.6841 |

Re-run only if you need to confirm your environment:

```bash
DATA_DIR=data/processed EPOCHS=10 RUN=baseline_1cpu_check sbatch slurm/train_cpu.sbatch
```

## 3. Salome experiments (submit yourself; append to shared CSV)

### 3a. Profile (bottleneck)

```bash
sbatch slurm/profile.sbatch
WORKERS=4 sbatch --cpus-per-task=5 slurm/profile.sbatch
```

Artifacts: `results/profiler/trace_*.json`, `summary_*.txt`, `ops_*.csv`.

### 3b. I/O optimization (EDF vs `.npz` vs HDF5)

```bash
sbatch slurm/io_benchmark.sbatch
```

Artifacts: `results/io_benchmark.csv`. Optional training on HDF5:

```bash
DATA_FORMAT=hdf5 RUN=hdf5_loader_w0 LOADER_WORKERS=0 \
  sbatch slurm/train_loader.sbatch
```

### 3c. Dask parallel preprocess

```bash
sbatch slurm/preprocess_dask.sbatch
# Compare meta.json preprocess_seconds to Dana’s 164.9 s in docs/DATA.md
```

### 3d. DataLoader worker sensitivity

```bash
bash slurm/loader_workers.sh
# Jobs: loader_w0, loader_w1, loader_w2, loader_w4, loader_w8
```

Fixed settings: seed 42, batch 64, lr 1e-3, epochs 10, **train threads = 1**.

## 4. Plots

```bash
source .venv/bin/activate
python scripts/plot_salome.py --results results/results.csv --io results/io_benchmark.csv
# → results/figures/loader_workers.png, io_benchmark.png, ...
```

## 5. Job settings template (copy into the report)

| Field | Value used |
|---|---|
| Cluster / partition | _fill in_ |
| Node hostname(s) | from job `.out` |
| CPUs requested | see each `.sbatch` |
| Memory | see each `.sbatch` |
| Modules / container | _fill in_ |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | set to train threads |
| Seed | 42 |
| Code commit | `git rev-parse HEAD` |

## 6. What “success” looks like for Salome

1. Profiler trace identifies the dominant op(s).
2. I/O table shows large speedup of `.npz`/HDF5 vs re-parsing EDF.
3. Dask preprocess wall time &lt; sequential 164.9 s (efficiency noted).
4. Loader-worker curve in `results/figures/`; macro F1 stays close to baseline.
5. Technology matrix filled in `docs/PERFORMANCE.md`.
6. Own Results subsection + help on Conclusions; ~50 s slide on bottleneck/optimization.
