"""Create Gabriel's experiment plots without third-party plotting packages."""
import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "gabriel_scaling_results.csv"
OUTPUT = ROOT / "docs" / "plots"


def read_results():
    with RESULTS.open(newline="") as f:
        return list(csv.DictReader(f))


def chart(title, ylabel, series, labels, width=760, height=430):
    left, right, top, bottom = 90, 30, 55, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [v for _, values in series for v in values]
    ymin = min(0.0, min(all_values))
    ymax = max(all_values)
    if ymax == ymin:
        ymax = ymin + 1
    ymax *= 1.12

    def sx(i, n):
        return left + i * plot_w / max(1, n - 1)

    def sy(v):
        return top + plot_h * (1 - (v - ymin) / (ymax - ymin))

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold">{html.escape(title)}</text>',
    ]

    for tick in range(6):
        v = ymin + (ymax - ymin) * tick / 5
        y = sy(v)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d1d5db" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{v:.3g}</text>')
    parts.append(f'<text x="24" y="{height/2}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 24 {height/2})">{html.escape(ylabel)}</text>')

    names = None
    for s_idx, (name, values) in enumerate(series):
        points = " ".join(f"{sx(i, len(values)):.1f},{sy(v):.1f}" for i, v in enumerate(values))
        color = colors[s_idx % len(colors)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for i, v in enumerate(values):
            parts.append(f'<circle cx="{sx(i, len(values)):.1f}" cy="{sy(v):.1f}" r="5" fill="{color}"/>')
            parts.append(f'<text x="{sx(i, len(values)):.1f}" y="{sy(v)-9:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{v:.3g}</text>')
        legend_x = left + s_idx * 150
        parts.append(f'<line x1="{legend_x}" y1="{height-22}" x2="{legend_x+25}" y2="{height-22}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+32}" y="{height-18}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>')

    for i, label in enumerate(labels):
        x = sx(i, len(labels))
        parts.append(f'<text x="{x:.1f}" y="{height-48}" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(str(label))}</text>')

    parts.append("</svg>")
    return "".join(parts)


def write_plot(filename, title, ylabel, series, labels):
    svg = chart(title, ylabel, series, labels)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / filename).write_text(svg)


def main():
    rows = read_results()
    cpu = [r for r in rows if r["device"] == "cpu"]
    threads = [int(r["threads"]) for r in cpu]
    throughput = [float(r["throughput_samples_s"]) for r in cpu]
    speedup = [float(r["speedup"]) for r in cpu]
    efficiency = [float(r["efficiency"]) * 100 for r in cpu]

    write_plot(
        "cpu_thread_scaling_throughput.svg",
        "CPU Thread Scaling: Throughput",
        "Samples/s",
        [("PyTorch CPU", throughput)],
        threads,
    )
    write_plot(
        "cpu_thread_scaling_speedup.svg",
        "CPU Thread Scaling: Speedup",
        "Speedup",
        [("Speedup vs 1 thread", speedup), ("Ideal linear", threads)],
        threads,
    )
    write_plot(
        "cpu_thread_scaling_efficiency.svg",
        "CPU Thread Scaling: Efficiency",
        "Efficiency (%)",
        [("Parallel efficiency", efficiency)],
        threads,
    )

    gpu = [r for r in rows if r["device"] == "cuda"]
    hardware = [("CPU 1", float(cpu[0]["throughput_samples_s"])),
                ("CPU 16", float(cpu[-1]["throughput_samples_s"]))]
    hardware.extend((r["run"], float(r["throughput_samples_s"])) for r in gpu)
    labels = [name for name, _ in hardware]
    values = [value for _, value in hardware]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "cpu_gpu_throughput.svg").write_text(
        bar_chart("CPU and GPU Throughput", "Samples/s", labels, values)
    )

    accuracy = [float(r["test_accuracy"]) for r in rows]
    macro_f1 = [float(r["test_macro_f1"]) for r in rows]
    labels = [r["run"] for r in rows]
    (OUTPUT / "accuracy_macro_f1.svg").write_text(
        bar_chart("Accuracy and Macro F1", "Score", labels, accuracy, macro_f1)
    )


def bar_chart(title, ylabel, labels, *datasets):
    width, height = 920, 470
    left, right, top, bottom = 100, 40, 55, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    flat = [v for dataset in datasets for v in dataset]
    ymax = max(flat) * 1.12
    colors = ["#2563eb", "#dc2626"]
    names = ["Accuracy", "Macro F1"] if len(datasets) == 2 else [ylabel]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        v = ymax * tick / 5
        y = top + plot_h * (1 - tick / 5)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d1d5db"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{v:.2f}</text>')
    group_w = plot_w / len(labels)
    bar_w = group_w * 0.7 / max(1, len(datasets))
    for group, label in enumerate(labels):
        for data_idx, dataset in enumerate(datasets):
            value = dataset[group]
            x = left + group * group_w + group_w * 0.15 + data_idx * bar_w
            bar_h = plot_h * value / ymax
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{colors[data_idx]}"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.3f}</text>')
        parts.append(f'<text x="{left+(group+0.5)*group_w:.1f}" y="{height-70}" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(label)}</text>')
    for idx, name in enumerate(names):
        lx = left + idx * 170
        parts.append(f'<rect x="{lx}" y="{height-28}" width="24" height="12" fill="{colors[idx]}"/>')
        parts.append(f'<text x="{lx+32}" y="{height-18}" font-family="sans-serif" font-size="12">{name}</text>')
    parts.append("</svg>")
    return "".join(parts)


if __name__ == "__main__":
    main()
