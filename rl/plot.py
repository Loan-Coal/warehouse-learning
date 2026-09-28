"""Charts. Needs matplotlib (in requirements.txt); nothing else in rl/ imports this file.

    python -m rl.plot checkpoints/myalgo                       # learning curve -> checkpoints/myalgo/learning_curve.png
    python -m rl.plot checkpoints/a checkpoints/b              # several runs on one chart per metric

rl.bench draws its own chart with bench_chart().
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # files only, no window
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated reference palette), never cycled past the list.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CURVE_METRICS = ("deliveries", "blocked", "mean_return")
BENCH_METRICS = ("deliveries", "value", "blocked", "invalid")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("runs", nargs="+", metavar="FOLDER", help="folders with a train_log.csv")
    parser.add_argument("--window", type=int, default=50, help="moving-average window in episodes")
    parser.add_argument("--out", help="output PNG (default: FOLDER/learning_curve.png of the first run)")
    args = parser.parse_args()
    logs = {}
    for folder in args.runs:
        path = Path(folder) / "train_log.csv"
        if not path.is_file():
            parser.error(f"{path} not found: train with --train --save {folder}")
        with open(path, encoding="utf-8") as f:
            logs[Path(folder).name] = list(csv.DictReader(f))
    if len(logs) > len(SERIES):
        parser.error(f"at most {len(SERIES)} runs per chart")
    out = Path(args.out) if args.out else Path(args.runs[0]) / "learning_curve.png"
    learning_curve(logs, out, args.window)
    print(f"wrote {out}")


def learning_curve(logs, out, window):
    """logs: {run name: train_log rows}. One panel per metric, moving average per run."""
    fig, axes = plt.subplots(1, len(CURVE_METRICS), figsize=(5 * len(CURVE_METRICS), 3.6))
    for ax, metric in zip(axes, CURVE_METRICS):
        for color, (name, rows) in zip(SERIES, logs.items()):
            values = [float(r[metric]) for r in rows]
            ax.plot(range(len(values)), values, color=color, alpha=0.15, linewidth=1)
            ax.plot(range(len(values)), _moving_average(values, window), color=color, linewidth=2, label=name)
        _style(ax, metric.replace("_", " "), "episode")
    if len(logs) > 1:
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=len(logs),
                   frameon=False, labelcolor=INK)
    fig.suptitle(f"Training ({window}-episode moving average)", color=INK, x=0.01, ha="left")
    _save(fig, out)


def bench_chart(rows, out):
    """rows: rl.bench result rows. One panel per metric: mean per scenario, one bar per algorithm."""
    algorithms = list(dict.fromkeys(r["algorithm"] for r in rows))[:len(SERIES)]
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    fig, axes = plt.subplots(1, len(BENCH_METRICS), figsize=(4.2 * len(BENCH_METRICS), 3.8))
    width = 0.8 / len(algorithms)
    for ax, metric in zip(axes, BENCH_METRICS):
        for a, (algorithm, color) in enumerate(zip(algorithms, SERIES)):
            means = []
            for scenario in scenarios:
                values = [float(r[metric]) for r in rows if r["algorithm"] == algorithm and r["scenario"] == scenario]
                means.append(sum(values) / len(values) if values else 0.0)
            xs = [s + (a - (len(algorithms) - 1) / 2) * width for s in range(len(scenarios))]
            # White edges leave a gap between neighbouring bars.
            ax.bar(xs, means, width, color=color, edgecolor="white", linewidth=2, label=algorithm)
        ax.set_xticks(range(len(scenarios)), scenarios)
        _style(ax, f"{metric} (mean per episode)", None)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=len(algorithms),
               frameon=False, labelcolor=INK)
    fig.suptitle("Benchmark", color=INK, x=0.01, ha="left")
    _save(fig, out)


def _moving_average(values, window):
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v - (values[i - window] if i >= window else 0.0)
        out.append(total / min(i + 1, window))
    return out


def _style(ax, title, xlabel):
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)


def _save(fig, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
