"""Pack Dana's per-recording .npz files into one HDF5 store for faster reloads.

Usage:
    python -m src.data.to_hdf5 --data data/processed --out data/processed/sleep_edf.h5
    python -m src.data.to_hdf5 --data data/sample --out data/sample/sleep_edf.h5
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np


def pack_split(data_dir, split, out_group):
    recs = json.loads((data_dir / "splits.json").read_text())[split]
    xs, ys = [], []
    for rec in recs:
        d = np.load(data_dir / f"{rec}.npz")
        xs.append(d["x"])
        ys.append(d["y"])
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    out_group.create_dataset("x", data=x, compression="gzip", compression_opts=4)
    out_group.create_dataset("y", data=y, compression="gzip", compression_opts=4)
    out_group.create_dataset("recordings", data=np.array(recs, dtype="S"))
    out_group.attrs["n_epochs"] = len(y)
    return len(y)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default=None, help="defaults to <data>/sleep_edf.h5")
    args = p.parse_args()

    data_dir = Path(args.data)
    out = Path(args.out) if args.out else data_dir / "sleep_edf.h5"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with h5py.File(out, "w") as f:
        f.attrs["source"] = str(data_dir)
        for split in ("train", "val", "test"):
            n = pack_split(data_dir, split, f.create_group(split))
            print(f"{split:5s}: {n:6d} epochs")
    elapsed = time.perf_counter() - t0
    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {out} ({size_mb:.1f} MB) in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
