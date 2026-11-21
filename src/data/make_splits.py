"""Split recordings into train / val / test BY SUBJECT and build a small sample.

Both nights of a subject always land in the same split, so the model is never
tested on a person it has already seen (no data leakage).

Usage:
    python -m src.data.make_splits --data data/processed --seed 42
    python -m src.data.make_splits --data data/processed --sample data/sample
"""
import argparse
import json
from pathlib import Path

import numpy as np


def make_splits(data_dir, val_frac=0.15, test_frac=0.15, seed=42):
    recs = sorted(p.stem for p in Path(data_dir).glob("SC*.npz"))
    subjects = sorted({r[3:5] for r in recs})
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_test = max(1, round(len(subjects) * test_frac))
    n_val = max(1, round(len(subjects) * val_frac))
    groups = {
        "test": subjects[:n_test],
        "val": subjects[n_test:n_test + n_val],
        "train": subjects[n_test + n_val:],
    }
    splits = {name: [r for r in recs if r[3:5] in subj] for name, subj in groups.items()}
    return splits, groups


def make_sample(data_dir, sample_dir, splits, n_recs=None, n_epochs=150, seed=42):
    """Tiny dataset for quick tests: a few recordings, 150 random epochs each."""
    data_dir, sample_dir = Path(data_dir), Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_recs = n_recs or {"train": 2, "val": 1, "test": 1}
    sample_splits = {}
    for name, n in n_recs.items():
        recs = splits[name]
        sample_splits[name] = recs[:n]
        for rec in recs[:n]:
            d = np.load(data_dir / f"{rec}.npz")
            idx = np.sort(rng.choice(len(d["y"]), min(n_epochs, len(d["y"])), replace=False))
            np.savez_compressed(sample_dir / f"{rec}.npz", x=d["x"][idx], y=d["y"][idx])
    (sample_dir / "splits.json").write_text(json.dumps(sample_splits, indent=2))
    return sample_splits


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="data/processed")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample", default=None, help="also write a small sample dataset here")
    args = p.parse_args()

    splits, groups = make_splits(args.data, args.val_frac, args.test_frac, args.seed)
    assert not set(groups["train"]) & set(groups["val"])
    assert not set(groups["train"]) & set(groups["test"])
    assert not set(groups["val"]) & set(groups["test"])

    out = Path(args.data) / "splits.json"
    out.write_text(json.dumps(splits, indent=2))
    for name in ("train", "val", "test"):
        print(f"{name:5s}: {len(groups[name]):2d} subjects, {len(splits[name]):3d} recordings")
    print(f"Wrote {out} (seed={args.seed})")

    if args.sample:
        s = make_sample(args.data, args.sample, splits, seed=args.seed)
        print(f"Wrote sample dataset to {args.sample}: {s}")


if __name__ == "__main__":
    main()
