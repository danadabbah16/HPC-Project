"""Distributed training entry point for the sleep-stage CNN.

Designed for Slurm + PyTorch DDP (Gloo for CPU, NCCL for GPU).
The existing single-process src.train entry point remains unchanged.
"""
import argparse
import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics import accuracy_score, f1_score

from src.data.dataset import STAGE_NAMES, SleepEDFDataset
from src.models.cnn import SleepCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64,
                   help="Global batch size across all DDP ranks")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scale-lr", action="store_true",
                   help="Scale lr as base_lr * world_size (optional weak-scaling variant)")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--loader-workers", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="ddp")
    p.add_argument("--results", default="results/results.csv")
    p.add_argument("--baseline-train-time", type=float, default=495.44,
                   help="Same-dataset 1-worker train time used for speedup")
    p.add_argument("--backend", choices=["auto", "gloo", "nccl"], default="auto")
    p.add_argument("--master-port", type=int, default=29500)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_distributed(args):
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))

    if world_size > 1:
        master_addr = os.environ.get("MASTER_ADDR")
        if not master_addr:
            nodelist = os.environ.get("SLURM_JOB_NODELIST")
            if nodelist:
                import subprocess
                hosts = subprocess.run(
                    ["bash", "-c", "scontrol show hostnames \"$1\"", "_", nodelist],
                    capture_output=True, text=True, check=True,
                ).stdout.split()
                master_addr = hosts[0]
        master_addr = master_addr or "127.0.0.1"
        os.environ["MASTER_ADDR"] = master_addr
        os.environ.setdefault("MASTER_PORT", str(args.master_port))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if world_size > 1:
        if args.backend == "auto":
            backend = "nccl" if device.type == "cuda" else "gloo"
        else:
            backend = args.backend
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)

    return rank, world_size, local_rank, device


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.argmax(dim=1).cpu())
        labels.append(y)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    per_class = f1_score(labels, preds, average=None, labels=range(len(STAGE_NAMES)), zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", labels=range(len(STAGE_NAMES)), zero_division=0),
        "per_class_f1": per_class,
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_communication(model, batch, loss_fn, device):
    """Estimate all-reduce cost by comparing normal backward with DDP no_sync."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return 0.0, 0.0

    x, y = batch
    x, y = x.to(device), y.to(device)
    model.train()
    buffers = {name: value.detach().clone() for name, value in model.module.named_buffers()}

    try:
        model.module.zero_grad(set_to_none=True)
        synchronize(device)
        t0 = time.perf_counter()
        with model.no_sync():
            loss = loss_fn(model(x), y)
            loss.backward()
        synchronize(device)
        no_sync_s = time.perf_counter() - t0

        model.module.zero_grad(set_to_none=True)
        synchronize(device)
        t0 = time.perf_counter()
        loss = loss_fn(model(x), y)
        loss.backward()
        synchronize(device)
        sync_s = time.perf_counter() - t0
    finally:
        model.module.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, value in model.module.named_buffers():
                value.copy_(buffers[name])

    comm_s = max(sync_s - no_sync_s, 0.0)
    fraction = comm_s / sync_s if sync_s > 0 else 0.0
    return comm_s, fraction


def append_result_csv(path, row):
    """Append a row and extend an existing CSV header exactly once, under an advisory lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = list(row.keys())

    with open(path, "a+", newline="", encoding="utf-8") as f:
        if fcntl is not None:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except (AttributeError, OSError):
                pass

        f.seek(0)
        content = f.read()
        if content:
            import io
            reader = csv.DictReader(io.StringIO(content))
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader)
            fieldnames = existing_fields + [name for name in extra_fields if name not in existing_fields]
            if fieldnames != existing_fields:
                f.seek(0)
                f.truncate()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow(existing_row)
                writer.writerow(row)
            else:
                f.seek(0, os.SEEK_END)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
        else:
            writer = csv.DictWriter(f, fieldnames=extra_fields)
            writer.writeheader()
            writer.writerow(row)

        if fcntl is not None:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (AttributeError, OSError):
                pass


def main():
    args = parse_args()
    rank, world_size, local_rank, device = resolve_distributed(args)
    set_seed(args.seed + rank)
    torch.set_num_threads(args.threads)

    if dist.is_initialized():
        backend = dist.get_backend()
    else:
        backend = "none"
    effective_lr = args.lr * world_size if args.scale_lr else args.lr
    print(
        f"Rank {rank}/{world_size} | local rank {local_rank} | device {device} | "
        f"backend {backend} | threads {args.threads} | lr {effective_lr}"
    )

    t0 = time.perf_counter()
    train_ds = SleepEDFDataset(args.data, "train")
    val_ds = SleepEDFDataset(args.data, "val")
    test_ds = SleepEDFDataset(args.data, "test")
    load_time = time.perf_counter() - t0

    if rank == 0:
        print(f"Loaded {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test epochs "
              f"in {load_time:.1f} s")

    if args.batch_size % world_size != 0:
        raise ValueError(
            f"--batch-size ({args.batch_size}) must be divisible by world_size ({world_size})"
        )
    per_rank_batch = args.batch_size // world_size

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if dist.is_initialized()
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=per_rank_batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.loader_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=args.loader_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=args.loader_workers, pin_memory=device.type == "cuda")

    model = SleepCNN().to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    loss_fn = torch.nn.CrossEntropyLoss(weight=train_ds.class_weights().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)

    if dist.is_initialized():
        dist.barrier()

    comm_time_s = 0.0
    comm_fraction = 0.0
    if world_size > 1:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        try:
            first_batch = next(iter(train_loader))
            comm_time_s, comm_fraction = benchmark_communication(
                model, first_batch, loss_fn, device
            )
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device)

    if rank == 0:
        print(
            f"Global batch {args.batch_size} | per-rank batch {per_rank_batch} | "
            f"communication benchmark: {comm_time_s:.6f} s ({comm_fraction:.3%})"
        )

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    best_path = Path(args.results).parent / f"model_{args.run_name}.pt"
    best_f1 = -1.0
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        synchronize(device)
        t0 = time.perf_counter()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        synchronize(device)
        epoch_time = time.perf_counter() - t0
        epoch_times.append(epoch_time)

        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            val = evaluate(model.module if dist.is_initialized() else model, val_loader, device)
            print(
                f"Epoch {epoch:2d} | loss {total_loss / len(train_ds):.3f} | "
                f"val acc {val['accuracy']:.3f} | val macro F1 {val['macro_f1']:.3f} | "
                f"{epoch_time:.1f} s ({len(train_ds) / epoch_time:.0f} global samples/s)"
            )
            if val["macro_f1"] > best_f1:
                best_f1 = val["macro_f1"]
                state = model.module.state_dict() if dist.is_initialized() else model.state_dict()
                torch.save(state, best_path)

        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        model_for_load = SleepCNN().to(device)
        model_for_load.load_state_dict(torch.load(best_path, map_location=device))
        val = evaluate(model_for_load, val_loader, device)
        test = evaluate(model_for_load, test_loader, device)
        print("\nBest model, test set:")
        print(f"  accuracy {test['accuracy']:.3f} | macro F1 {test['macro_f1']:.3f}")
        for name, f1 in zip(STAGE_NAMES, test["per_class_f1"]):
            print(f"  {name:>3}: F1 {f1:.3f}")

        train_time = sum(epoch_times)
        speedup = args.baseline_train_time / train_time if train_time > 0 else 0.0
        efficiency = speedup / world_size if world_size > 0 else 0.0
        nodes = int(os.environ.get("SLURM_NNODES", "1"))
        row = {
            "run": args.run_name,
            "date": datetime.now().isoformat(timespec="seconds"),
            "device": device.type,
            "nodes": nodes,
            "workers": world_size,
            "world_size": world_size,
            "threads": args.threads,
            "batch_size": args.batch_size,
            "lr": effective_lr,
            "epochs": args.epochs,
            "seed": args.seed,
            "data_load_s": round(load_time, 2),
            "train_time_s": round(train_time, 2),
            "time_per_epoch_s": round(train_time / args.epochs, 2),
            "throughput_samples_s": round(len(train_ds) * args.epochs / train_time, 1),
            "speedup": round(speedup, 4),
            "efficiency": round(efficiency, 4),
            "comm_time_s": round(comm_time_s, 6),
            "comm_fraction": round(comm_fraction, 6),
            "val_accuracy": round(val["accuracy"], 4),
            "val_macro_f1": round(val["macro_f1"], 4),
            "test_accuracy": round(test["accuracy"], 4),
            "test_macro_f1": round(test["macro_f1"], 4),
            **{f"test_f1_{n}": round(f, 4) for n, f in zip(STAGE_NAMES, test["per_class_f1"])},
        }
        append_result_csv(args.results, row)
        print(f"\nSaved results to {args.results} and model to {best_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
