"""Plot DDP scaling rows from the shared results CSV.

Usage:
    python scripts/plot_ddp.py --input results/results.csv --output results/plots
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ddp_rows(rows):
    return [r for r in rows if r.get("world_size") or r.get("run", "").startswith("ddp")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/results.csv")
    p.add_argument("--output", default="results/plots")
    args = p.parse_args()

    rows = ddp_rows(read_rows(args.input))
    if not rows:
        raise SystemExit("No DDP rows found. Run distributed jobs before plotting.")

    by_workers = {}
    for row in rows:
        workers = int(row.get("world_size") or row.get("workers") or 1)
        by_workers.setdefault(workers, []).append(row)
    workers = sorted(by_workers)

    speedups = [sum(float(r["speedup"]) for r in by_workers[w]) / len(by_workers[w]) for w in workers]
    efficiencies = [sum(float(r["efficiency"]) for r in by_workers[w]) / len(by_workers[w]) for w in workers]
    comm_fractions = [
        sum(float(r["comm_fraction"]) for r in by_workers[w] if r.get("comm_fraction"))
        / sum(1 for r in by_workers[w] if r.get("comm_fraction"))
        if any(r.get("comm_fraction") for r in by_workers[w])
        else 0.0
        for w in workers
    ]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    plt.plot(workers, speedups, "o-", label="DDP speedup")
    plt.plot(workers, workers, "--", label="Ideal")
    plt.xlabel("Workers")
    plt.ylabel("Speedup")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "ddp_speedup.svg")

    plt.figure(figsize=(7, 4.5))
    plt.plot(workers, efficiencies, "o-", color="#4C78A8")
    plt.axhline(1.0, linestyle="--", color="#888", label="Ideal")
    plt.xlabel("Workers")
    plt.ylabel("Parallel efficiency")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "ddp_efficiency.svg")

    plt.figure(figsize=(7, 4.5))
    plt.plot(workers, comm_fractions, "o-", color="#E45756")
    plt.xlabel("Workers")
    plt.ylabel("Communication fraction")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "ddp_comm_fraction.svg")


if __name__ == "__main__":
    main()
