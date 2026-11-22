"""Plots for Salome's Performance Lead experiments.

Reads the shared results CSV (+ optional io_benchmark.csv) and writes figures
under results/figures/ with the shared colour scheme.

    python scripts/plot_salome.py --results results/results.csv
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import COLOURS, apply_style, savefig


def plot_loader_workers(df, out_dir):
    sub = df[df["run"].astype(str).str.startswith("loader_w")].copy()
    if sub.empty:
        print("No loader_w* rows yet — skip loader plot")
        return
    if "loader_workers" in sub.columns:
        sub["w"] = sub["loader_workers"].astype(float)
    else:
        sub["w"] = sub["run"].str.replace("loader_w", "", regex=False).astype(int)
    sub = sub.sort_values("w")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(sub["w"], sub["throughput_samples_s"], "o-", color=COLOURS["loader"],
             label="throughput")
    ax2.plot(sub["w"], sub["test_macro_f1"], "s--", color=COLOURS["baseline"],
             label="macro F1")
    ax1.set_xlabel("DataLoader workers")
    ax1.set_ylabel("Throughput (samples/s)")
    ax2.set_ylabel("Test macro F1")
    ax1.set_title("DataLoader worker sensitivity")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="best")
    savefig(fig, out_dir / "loader_workers.png")
    plt.close(fig)


def plot_io(io_csv, out_dir):
    path = Path(io_csv)
    if not path.exists():
        print(f"No {path} yet — skip I/O plot")
        return
    df = pd.read_csv(path)
    # Keep the latest row per path.
    df = df.drop_duplicates(subset=["path"], keep="last")
    colour_map = {"edf_reparse": COLOURS["io_edf"], "npz_dir": COLOURS["io_npz"], "hdf5": COLOURS["io_hdf5"]}
    colours = [colour_map.get(p, COLOURS["cpu"]) for p in df["path"]]
    fig, ax = plt.subplots()
    ax.bar(df["path"], df["seconds"], color=colours)
    ax.set_ylabel("Load time (s)")
    ax.set_title("I/O path comparison (lower is better)")
    savefig(fig, out_dir / "io_benchmark.png")
    plt.close(fig)


def plot_baseline_vs_optimized(df, out_dir):
    """Compare baseline_1cpu vs a fast-format / loader run if present."""
    names = ["baseline_1cpu"]
    for candidate in ("loader_w4", "hdf5_loader_w0", "npz_fast"):
        if (df["run"] == candidate).any():
            names.append(candidate)
    sub = df[df["run"].isin(names)].drop_duplicates("run", keep="last")
    if len(sub) < 2:
        print("Need baseline + at least one optimized run for comparison plot")
        return
    fig, ax = plt.subplots()
    ax.bar(sub["run"], sub["throughput_samples_s"], color=[COLOURS["baseline"], COLOURS["loader"]][:len(sub)])
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title("Baseline vs optimized training throughput")
    plt.xticks(rotation=20, ha="right")
    savefig(fig, out_dir / "baseline_vs_optimized.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/results.csv")
    p.add_argument("--io", default="results/io_benchmark.csv")
    p.add_argument("--out", default="results/figures")
    args = p.parse_args()

    apply_style()
    out_dir = Path(args.out)
    df = pd.read_csv(args.results)
    plot_loader_workers(df, out_dir)
    plot_baseline_vs_optimized(df, out_dir)
    plot_io(args.io, out_dir)


if __name__ == "__main__":
    main()
