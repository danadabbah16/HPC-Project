"""Profile one training epoch with the PyTorch Profiler (Salome / Performance Lead).

Finds whether time is spent in data loading, host→device copy, or compute.
Writes a Chrome trace + a short CSV summary under results/profiler/.

Quick local test:
    python -m src.profile_train --data data/sample --epochs 1

Cluster (1 CPU, matches baseline setup):
    sbatch slurm/profile.sbatch
"""
import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader

from src.data.dataset import load_split
from src.models.cnn import SleepCNN
from src.train import set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="data/sample")
    p.add_argument("--data-format", default="npz", choices=["npz", "hdf5"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--loader-workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wait", type=int, default=1, help="profiler schedule: warm-up batches to skip")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--active", type=int, default=3, help="batches to record")
    p.add_argument("--out", default="results/profiler")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_ds = load_split(args.data, "train", fmt=args.data_format)
    loader_kw = dict(num_workers=args.loader_workers)
    if args.loader_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)

    model = SleepCNN().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=train_ds.class_weights().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print(f"Profiling on {device} | threads={args.threads} | "
          f"loader_workers={args.loader_workers} | batches wait/warmup/active="
          f"{args.wait}/{args.warmup}/{args.active}")

    model.train()
    sched = schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=1)
    t0 = time.perf_counter()
    with profile(
        activities=activities,
        schedule=sched,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            prof.step()
            if step >= args.wait + args.warmup + args.active - 1:
                break
    wall = time.perf_counter() - t0

    trace_path = out / f"trace_{stamp}.json"
    prof.export_chrome_trace(str(trace_path))

    key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=key, row_limit=20)
    (out / f"summary_{stamp}.txt").write_text(table)
    print(table)
    print(f"\nWall time for profiled loop: {wall:.2f} s")
    print(f"Chrome trace: {trace_path}")
    print("Open in chrome://tracing or https://ui.perfetto.dev")

    # Compact CSV of top ops for the shared results folder / report tables.
    rows = []
    for evt in prof.key_averages():
        rows.append({
            "name": evt.key,
            "cpu_time_total_us": round(evt.cpu_time_total, 1),
            "self_cpu_time_total_us": round(evt.self_cpu_time_total, 1),
            "cuda_time_total_us": round(getattr(evt, "cuda_time_total", 0.0) or 0.0, 1),
            "cpu_memory_usage": evt.cpu_memory_usage or 0,
            "device": device.type,
            "loader_workers": args.loader_workers,
            "threads": args.threads,
            "batch_size": args.batch_size,
            "data_format": args.data_format,
            "wall_s": round(wall, 3),
            "date": datetime.now().isoformat(timespec="seconds"),
        })
    rows.sort(key=lambda r: r["self_cpu_time_total_us"], reverse=True)
    csv_path = out / f"ops_{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows[:50])
    print(f"Top ops CSV: {csv_path}")


if __name__ == "__main__":
    main()
