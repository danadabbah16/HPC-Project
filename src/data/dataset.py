"""PyTorch Dataset for the preprocessed Sleep-EDF epochs.

    from src.data.dataset import SleepEDFDataset, load_split
    train = SleepEDFDataset("data/processed", "train")
    x, y = train[0]          # x: float32 tensor (1, 3000), y: int64 scalar
    weights = train.class_weights()   # for nn.CrossEntropyLoss(weight=...)

    # Single-file HDF5 store (Salome): build with `python -m src.data.to_hdf5`
    train = load_split("data/processed", "train", fmt="hdf5")

Use "data/sample" instead of "data/processed" for quick tests.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


class SleepEDFDataset(Dataset):
    """Load a split from Dana's per-recording compressed .npz files into RAM."""

    def __init__(self, data_dir, split="train"):
        data_dir = Path(data_dir)
        recs = json.loads((data_dir / "splits.json").read_text())[split]
        xs, ys = [], []
        for rec in recs:
            d = np.load(data_dir / f"{rec}.npz")
            xs.append(d["x"])
            ys.append(d["y"])
        # Whole split held in RAM: ~12 kB per epoch, about 2.5 GB for all of train.
        self.x = torch.from_numpy(np.concatenate(xs)).unsqueeze(1)   # (N, 1, 3000)
        self.y = torch.from_numpy(np.concatenate(ys))                # (N,)
        self.recordings = recs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def class_counts(self):
        return torch.bincount(self.y, minlength=len(STAGE_NAMES))

    def class_weights(self):
        """Inverse-frequency weights, normalised to mean 1."""
        counts = self.class_counts().float().clamp(min=1)
        w = counts.sum() / (len(counts) * counts)
        return w / w.mean()


class SleepEDFHDF5Dataset(Dataset):
    """Load a split from one HDF5 file written by `src.data.to_hdf5`.

    Data is still copied into RAM once at construction so training cost matches
    the .npz path; the win is fewer open() calls and a single sequential read.
    """

    def __init__(self, hdf5_path, split="train"):
        import h5py

        hdf5_path = Path(hdf5_path)
        with h5py.File(hdf5_path, "r") as f:
            g = f[split]
            self.x = torch.from_numpy(g["x"][:]).unsqueeze(1)  # (N, 1, 3000)
            self.y = torch.from_numpy(g["y"][:])
            self.recordings = [r.decode() if isinstance(r, bytes) else r for r in g["recordings"][:]]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def class_counts(self):
        return torch.bincount(self.y, minlength=len(STAGE_NAMES))

    def class_weights(self):
        counts = self.class_counts().float().clamp(min=1)
        w = counts.sum() / (len(counts) * counts)
        return w / w.mean()


def load_split(data_dir, split="train", fmt="npz"):
    """Factory used by train / profile / benchmark scripts."""
    data_dir = Path(data_dir)
    if fmt == "npz":
        return SleepEDFDataset(data_dir, split)
    if fmt == "hdf5":
        path = data_dir if data_dir.suffix == ".h5" else data_dir / "sleep_edf.h5"
        return SleepEDFHDF5Dataset(path, split)
    raise ValueError(f"unknown data format: {fmt}")
