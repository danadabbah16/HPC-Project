"""PyTorch Dataset for the preprocessed Sleep-EDF epochs.

    from src.data.dataset import SleepEDFDataset
    train = SleepEDFDataset("data/processed", "train")
    x, y = train[0]          # x: float32 tensor (1, 3000), y: int64 scalar
    weights = train.class_weights()   # for nn.CrossEntropyLoss(weight=...)

Use "data/sample" instead of "data/processed" for quick tests.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


class SleepEDFDataset(Dataset):
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
