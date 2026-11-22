"""Plot the batch size sensitivity experiment.

Reads results/results.csv and saves results/plots/batch_size.png.

    python scripts/plot_batch_size.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path("results/results.csv")
OUT = Path("results/plots/batch_size.png")

# Shared group style: same colours for "fixed" and "scaled" in every plot
COLORS = {"fixed": "#2a78d6", "scaled": "#eb6834"}
MARKERS = {"fixed": "o", "scaled": "s"}
LABELS = {"fixed": "Fixed learning rate (0.001)", "scaled": "Scaled learning rate (0.001 × batch/64)"}


def load():
    with open(RESULTS) as f:
        rows = list(csv.DictReader(f))
    series = {"fixed": [], "scaled": []}
    for r in rows:
        bs = int(r["batch_size"])
        point = (bs, float(r["time_per_epoch_s"]), float(r["test_macro_f1"]))
        if r["run"] == "baseline_1cpu":
            # Batch 64 with lr 0.001 is the starting point of both lines
            series["fixed"].append(point)
            series["scaled"].append(point)
        elif r["run"].startswith("bs") and r["run"].endswith("_lr_fixed"):
            series["fixed"].append(point)
        elif r["run"].startswith("bs") and r["run"].endswith("_lr_scaled"):
            series["scaled"].append(point)
    return {k: sorted(v) for k, v in series.items()}


def main():
    series = load()
    fig, (ax_time, ax_f1) = plt.subplots(1, 2, figsize=(10, 4))

    # Speed: the learning rate does not change the amount of work, so the fixed and
    # scaled runs of one batch size are two timings of the same thing. Show their
    # average, with a bar for the range between them (noise from sharing the node).
    times = {}
    for pts in series.values():
        for bs, t, _ in pts:
            times.setdefault(bs, set()).add(t)
    bs_t = sorted(times)
    mean = [sum(times[b]) / len(times[b]) for b in bs_t]
    low = [m - min(times[b]) for b, m in zip(bs_t, mean)]
    high = [max(times[b]) - m for b, m in zip(bs_t, mean)]
    ax_time.errorbar(bs_t, mean, yerr=[low, high], color="#2a78d6", marker="o",
                     linewidth=2, markersize=7, capsize=4, elinewidth=1.5)

    for name, pts in series.items():
        bs = [p[0] for p in pts]
        ax_f1.plot(bs, [p[2] for p in pts], color=COLORS[name], marker=MARKERS[name],
                   linewidth=2, markersize=7, label=LABELS[name])

    all_bs = sorted({p[0] for pts in series.values() for p in pts})
    for ax in (ax_time, ax_f1):
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_bs)
        ax.set_xticklabels([str(b) for b in all_bs])
        ax.set_xlabel("Batch size")
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    ax_time.set_ylabel("Time per epoch (s)")
    ax_time.set_title("Training speed, 1 CPU core\n(bars: spread between the two runs)",
                      loc="left", fontsize=11)
    ax_time.set_ylim(0, None)

    ax_f1.set_ylabel("Test macro F1")
    ax_f1.set_title("Accuracy on the test set", loc="left", fontsize=11)
    ax_f1.set_ylim(0.60, 0.75)
    ax_f1.legend(loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()