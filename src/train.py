"""Train the sleep-stage CNN and log speed + accuracy.

Quick test on the laptop:
    python -m src.train --data data/sample --epochs 2

Baseline on the cluster (1 CPU core):
    srun -c 1 python -m src.train --data data/processed --run-name baseline_1cpu
"""
import argparse
import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import STAGE_NAMES, SleepEDFDataset
from src.models.cnn import SleepCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed", help="folder with the .npz files and splits.json")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                   help="CPU threads for PyTorch (defaults to the cores Slurm gave you)")
    p.add_argument("--loader-workers", type=int, default=0, help="DataLoader worker processes")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="test")
    p.add_argument("--results", default="results/results.csv", help="CSV that every run appends one row to")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return accuracy, macro F1 and per-class F1 on one split."""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.argmax(dim=1).cpu())
        labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    per_class = f1_score(labels, preds, average=None, labels=range(len(STAGE_NAMES)), zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", labels=range(len(STAGE_NAMES)), zero_division=0),
        "per_class_f1": per_class,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device} | threads: {args.threads} | batch size: {args.batch_size}")

    # --- Data (Dana's pipeline) ---
    t0 = time.perf_counter()
    train_ds = SleepEDFDataset(args.data, "train")
    val_ds = SleepEDFDataset(args.data, "val")
    test_ds = SleepEDFDataset(args.data, "test")
    load_time = time.perf_counter() - t0
    print(f"Loaded {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test epochs in {load_time:.1f} s")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.loader_workers)
    val_loader = DataLoader(val_ds, batch_size=256, num_workers=args.loader_workers)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=args.loader_workers)

    # --- Model, loss, optimiser ---
    model = SleepCNN().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=train_ds.class_weights().to(device))  # rare stages count more
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training ---
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    best_path = Path(args.results).parent / f"model_{args.run_name}.pt"
    best_f1, epoch_times = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        if device.type == "cuda":
            torch.cuda.synchronize()  # make sure the GPU has finished before stopping the clock
        epoch_time = time.perf_counter() - t0
        epoch_times.append(epoch_time)

        val = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d} | loss {total_loss / len(train_ds):.3f} | "
              f"val acc {val['accuracy']:.3f} | val macro F1 {val['macro_f1']:.3f} | "
              f"{epoch_time:.1f} s ({len(train_ds) / epoch_time:.0f} samples/s)")
        if val["macro_f1"] > best_f1:  # keep the best model by validation macro F1
            best_f1 = val["macro_f1"]
            torch.save(model.state_dict(), best_path)

    # --- Final results on the best model ---
    model.load_state_dict(torch.load(best_path, map_location=device))
    val = evaluate(model, val_loader, device)
    test = evaluate(model, test_loader, device)
    print("\nBest model, test set:")
    print(f"  accuracy {test['accuracy']:.3f} | macro F1 {test['macro_f1']:.3f}")
    for name, f1 in zip(STAGE_NAMES, test["per_class_f1"]):
        print(f"  {name:>3}: F1 {f1:.3f}")

    train_time = sum(epoch_times)
    row = {
        "run": args.run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": device.type,
        "nodes": 1,
        "workers": 1,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "data_load_s": round(load_time, 2),
        "train_time_s": round(train_time, 2),
        "time_per_epoch_s": round(train_time / args.epochs, 2),
        "throughput_samples_s": round(len(train_ds) * args.epochs / train_time, 1),
        "val_accuracy": round(val["accuracy"], 4),
        "val_macro_f1": round(val["macro_f1"], 4),
        "test_accuracy": round(test["accuracy"], 4),
        "test_macro_f1": round(test["macro_f1"], 4),
        **{f"test_f1_{n}": round(f, 4) for n, f in zip(STAGE_NAMES, test["per_class_f1"])},
    }
    new_file = not Path(args.results).exists()
    with open(args.results, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nSaved results to {args.results} and model to {best_path}")


if __name__ == "__main__":
    main()
