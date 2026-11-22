"""Compare I/O paths: re-read EDF vs load .npz vs load HDF5 (Salome).

The main optimization claim: preprocess once to a fast format, then never
re-parse EDF during training. This script measures that claim.

Usage:
    # Needs data/raw (EDF) and data/processed (.npz). Optional: build HDF5 first.
    python -m src.data.to_hdf5 --data data/processed
    python -m src.benchmark_io --raw data/raw/sleep-cassette --processed data/processed

    # Sample-only (skips EDF timing if raw is missing):
    python -m src.benchmark_io --processed data/sample --skip-edf
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data.dataset import load_split
from src.data.preprocess import find_pairs, load_recording


def time_edf(raw_dir, n_recordings=8, channel="EEG Fpz-Cz"):
    """Time reading + filtering + epoching a few EDFs (stand-in for 'every epoch')."""
    pairs = find_pairs(raw_dir)[:n_recordings]
    if not pairs:
        raise FileNotFoundError(f"no EDF pairs in {raw_dir}")
    t0 = time.perf_counter()
    n_epochs = 0
    for rec, psg, hyp in pairs:
        x, y = load_recording(psg, hyp, channel)
        n_epochs += len(y)
    elapsed = time.perf_counter() - t0
    return {
        "path": "edf_reparse",
        "n_recordings": len(pairs),
        "n_epochs": n_epochs,
        "seconds": round(elapsed, 3),
        "epochs_per_s": round(n_epochs / elapsed, 1) if elapsed else 0.0,
    }


def time_npz_dir(data_dir, split="train"):
    t0 = time.perf_counter()
    ds = load_split(data_dir, split, fmt="npz")
    elapsed = time.perf_counter() - t0
    return {
        "path": "npz_dir",
        "n_recordings": len(ds.recordings),
        "n_epochs": len(ds),
        "seconds": round(elapsed, 3),
        "epochs_per_s": round(len(ds) / elapsed, 1) if elapsed else 0.0,
    }


def time_hdf5(data_dir, split="train"):
    data_dir = Path(data_dir)
    h5 = data_dir if data_dir.suffix == ".h5" else data_dir / "sleep_edf.h5"
    if not h5.exists():
        return None
    t0 = time.perf_counter()
    ds = load_split(data_dir, split, fmt="hdf5")
    elapsed = time.perf_counter() - t0
    return {
        "path": "hdf5",
        "n_recordings": len(ds.recordings),
        "n_epochs": len(ds),
        "seconds": round(elapsed, 3),
        "epochs_per_s": round(len(ds) / elapsed, 1) if elapsed else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", default="data/raw/sleep-cassette")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--n-edf", type=int, default=8, help="how many EDFs to time (scaled later in the report)")
    p.add_argument("--skip-edf", action="store_true")
    p.add_argument("--results", default="results/io_benchmark.csv")
    args = p.parse_args()

    rows = []
    if not args.skip_edf and Path(args.raw).exists():
        row = time_edf(args.raw, n_recordings=args.n_edf)
        rows.append(row)
        print(f"EDF  ({row['n_recordings']} nights): {row['seconds']:.2f} s  "
              f"({row['epochs_per_s']:.0f} epochs/s)")
    else:
        print("Skipping EDF timing (--skip-edf or raw dir missing)")

    row = time_npz_dir(args.processed, "train")
    rows.append(row)
    print(f"NPZ  train split: {row['seconds']:.2f} s  ({row['epochs_per_s']:.0f} epochs/s)")

    row = time_hdf5(args.processed, "train")
    if row:
        rows.append(row)
        print(f"HDF5 train split: {row['seconds']:.2f} s  ({row['epochs_per_s']:.0f} epochs/s)")
    else:
        print("No sleep_edf.h5 yet — run: python -m src.data.to_hdf5 --data", args.processed)

    # Relative speedup vs EDF (if measured), else vs npz as baseline note.
    base = next((r for r in rows if r["path"] == "edf_reparse"), None)
    if base:
        # Extrapolate EDF cost to the same epoch count as the train split.
        train_epochs = next(r["n_epochs"] for r in rows if r["path"] == "npz_dir")
        edf_per_epoch = base["seconds"] / max(base["n_epochs"], 1)
        edf_projected = edf_per_epoch * train_epochs
        print(f"\nProjected EDF cost for {train_epochs} train epochs: {edf_projected:.1f} s")
        for r in rows:
            if r["path"] == "edf_reparse":
                continue
            r["speedup_vs_edf_projected"] = round(edf_projected / max(r["seconds"], 1e-9), 1)
            print(f"  {r['path']}: {r['speedup_vs_edf_projected']}× vs projected EDF")

    for r in rows:
        r["date"] = datetime.now().isoformat(timespec="seconds")
        r["processed"] = str(args.processed)

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    new_file = not out.exists()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nAppended {len(rows)} rows to {out}")

    # Also dump a JSON snapshot for the report.
    snap = out.with_suffix(".json")
    snap.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
