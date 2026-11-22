"""Parallel Sleep-EDF preprocessing with Dask (same outputs as src.data.preprocess).

Compares wall-clock time against the sequential baseline in docs/DATA.md (~165 s).

Usage:
    python -m src.data.preprocess_dask --raw data/raw/sleep-cassette --out data/processed_dask --n-workers 4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dask.distributed import Client, LocalCluster

# Workers are separate processes; make sure the repo root is importable.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("PYTHONPATH", _REPO_ROOT)

from src.data.preprocess import CHANNEL, STAGE_NAMES, find_pairs, load_recording


def _process_one(rec, psg, hyp, out_dir, channel):
    """Worker task: one recording -> one .npz + summary dict."""
    x, y = load_recording(psg, hyp, channel)
    out = Path(out_dir)
    np.savez_compressed(out / f"{rec}.npz", x=x, y=y)
    counts = np.bincount(y, minlength=5).tolist()
    return rec, {"subject": rec[3:5], "n_epochs": len(y), "counts": counts}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", default="data/raw/sleep-cassette")
    p.add_argument("--out", default="data/processed_dask")
    p.add_argument("--channel", default=CHANNEL)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--threads-per-worker", type=int, default=1)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(args.raw)
    print(f"Found {len(pairs)} recordings; launching {args.n_workers} Dask workers")

    t0 = time.perf_counter()
    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        processes=True,
        silence_logs=40,
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    futures = [
        client.submit(_process_one, rec, str(psg), str(hyp), str(out), args.channel)
        for rec, psg, hyp in pairs
    ]
    results = client.gather(futures)
    client.close()
    cluster.close()
    elapsed = time.perf_counter() - t0

    summary = {rec: info for rec, info in results}
    for rec, info in sorted(summary.items()):
        print(f"{rec}: {info['n_epochs']:5d} epochs  {dict(zip(STAGE_NAMES, info['counts']))}")

    total = np.sum([s["counts"] for s in summary.values()], axis=0)
    meta = {
        "channel": args.channel,
        "n_recordings": len(summary),
        "n_subjects": len({s["subject"] for s in summary.values()}),
        "n_epochs": int(total.sum()),
        "class_counts": dict(zip(STAGE_NAMES, total.tolist())),
        "preprocess_seconds": round(elapsed, 1),
        "n_workers": args.n_workers,
        "backend": "dask",
        "recordings": summary,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nTotal: {meta['n_epochs']} epochs in {meta['preprocess_seconds']} s "
          f"({args.n_workers} workers)")
    print("Class counts:", meta["class_counts"])


if __name__ == "__main__":
    main()
