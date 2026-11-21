# Data pipeline

Owner: Dana (Data Lead). Code: `scripts/download_data.sh`, `src/data/`.

## How to get the data

The data is not in the repo (`/data/` is git-ignored). Run these on the cluster, inside the repo:

```bash
bash scripts/download_data.sh data/raw            # login node, ~8 GB. Add "10" for 10 subjects only
srun -c 4 --mem=8G python -m src.data.preprocess --raw data/raw/sleep-cassette --out data/processed
python -m src.data.make_splits --data data/processed --seed 42 --sample data/sample
```

Then, in Python:

```python
from src.data.dataset import SleepEDFDataset
train = SleepEDFDataset("data/processed", "train")   # also "val", "test"
x, y = train[0]                   # x: float32 (1, 3000), y: int64 in 0..4
weights = train.class_weights()   # for nn.CrossEntropyLoss(weight=...)
```

`data/sample` is a tiny version committed to the repo for quick tests: 4 recordings, 150 random epochs each (2 train, 1 val, 1 test).

## Source

Sleep-EDF Expanded v1.0.0 from PhysioNet (`physionet.org/files/sleep-edfx/1.0.0/`), sleep-cassette study only: 153 whole-night recordings from 78 healthy subjects. Most subjects have two nights; three have one. The sleep-telemetry study (subjects on a sleep drug) is not used. All downloaded files are verified against PhysioNet's SHA-256 checksums.

## Choices made

| Choice | What we do | Why |
|---|---|---|
| Channel | `EEG Fpz-Cz` only, 100 Hz | The usual single-channel choice for this dataset; keeps the baseline simple. Pz-Oz and EOG are in the files and can be added (`--channel`, or extend `load_recording`). |
| Filter | Band-pass 0.3–35 Hz (MNE FIR, zero-phase) | Removes slow drift and high-frequency noise; keeps delta waves, spindles and K-complexes. |
| Epochs | 30 s = 3000 samples, non-overlapping | Matches how the experts scored the recordings. |
| Label mapping | W→0, S1→1 (N1), S2→2 (N2), S3 and S4→3 (N3), R→4 (REM) | The recordings were scored with the older R&K rules; merging S3 and S4 into N3 follows the current AASM standard. |
| Dropped epochs | "Movement time" and "?" (unscored) | They are not sleep stages. |
| Wake trimming | Keep only 30 min of wake before the first and after the last sleep epoch | Recordings last about 20 hours, mostly daytime wake, which would otherwise swamp the dataset. |
| Normalisation | z-score per recording | Removes amplitude differences between subjects and nights. |
| Split | By subject, 70/15/15, seed 42 | Both nights of a subject always land in the same split, so the model is never tested on a person it has trained on. |
| Storage | One compressed `.npz` per recording (`x`, `y`), plus `meta.json` and `splits.json` | Read once from EDF, then load fast. `meta.json` has per-recording counts. |

## Resulting dataset

| Stage | Epochs | Share |
|---|---|---|
| Wake | 65,941 | 33.7% |
| N1 | 21,522 | 11.0% |
| N2 | 69,132 | 35.4% |
| N3 | 13,039 | 6.7% |
| REM | 25,835 | 13.2% |
| **Total** | **195,469** | |

| Split | Subjects | Recordings | Epochs | Share of data |
|---|---|---|---|---|
| Train | 54 | 105 | 133,319 | 68.2% |
| Val | 12 | 24 | 34,286 | 17.5% |
| Test | 12 | 24 | 27,864 | 14.3% |

Class counts per split, with each class's share of that split:

| Stage | Train | Val | Test |
|---|---|---|---|
| Wake | 44,721 (33.5%) | 12,911 (37.7%) | 8,309 (29.8%) |
| N1 | 15,241 (11.4%) | 3,219 (9.4%) | 3,062 (11.0%) |
| N2 | 48,305 (36.2%) | 10,858 (31.7%) | 9,969 (35.8%) |
| N3 | 7,716 (5.8%) | 2,575 (7.5%) | 2,748 (9.9%) |
| REM | 17,336 (13.0%) | 4,723 (13.8%) | 3,776 (13.6%) |
| **Total** | **133,319** | **34,286** | **27,864** |

The split is by subject, not stratified by class, so the class shares differ a little between splits. With the fixed seed (42), everyone gets exactly these numbers.

Inverse-frequency class weights from `class_weights()` on the training set (W, N1, N2, N3, REM): 0.38 / 1.11 / 0.35 / 2.19 / 0.97.

The classes are imbalanced: N3 is the rarest (6.7%) and N1 the second rarest (11.0%), so macro F1 and per-class results matter more than plain accuracy.

## Timing

Preprocessing all 153 recordings from EDF took 164.9 s as one Slurm job (single process). This is the "read from EDF" cost that the saved `.npz` files avoid, and the baseline for any parallel preprocessing (e.g. Dask).

## References

- Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE Transactions on Biomedical Engineering, 47(9):1185–1194, 2000.
- Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. Circulation, 101(23):e215–e220, 2000.
