"""Charts. Needs matplotlib (in requirements.txt); nothing else in rl/ imports this file.

    python -m rl.plot checkpoints/myalgo                       # learning curve -> checkpoints/myalgo/learning_curve.png
    python -m rl.plot checkpoints/a checkpoints/b              # several runs on one chart per metric
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
CURVE_METRICS = ("deliveries", "invalid", "mean_return")


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
    """logs: {run name: train_log rows}. One panel per metric, moving average per run.

    Training rows are drawn faint: they include exploration and any teacher the algorithm
    mixes in, so they say little about the policy itself. Rows written with mode=eval come
    from held-out episodes with all of that off, and are drawn on top. A log without a mode
    column is all training rows, so older runs plot exactly as before.
    """
    fig, axes = plt.subplots(1, len(CURVE_METRICS), figsize=(5 * len(CURVE_METRICS), 3.6))
    any_eval = False
    for ax, metric in zip(axes, CURVE_METRICS):
        for color, (name, rows) in zip(SERIES, logs.items()):
            train = [r for r in rows if r.get("mode", "train") != "eval"]
            if train:
                xs = [int(r["episode"]) for r in train]
                ys = [float(r[metric]) for r in train]
                ax.plot(xs, ys, color=color, alpha=0.15, linewidth=1)
                ax.plot(xs, _moving_average(ys, window), color=color, linewidth=2, alpha=0.45,
                        label=f"{name} (train)")
            ex, ey = _eval_series(rows, metric)
            if ex:
                any_eval = True
                ax.plot(ex, ey, color=color, linewidth=2.4, marker="o", markersize=3.5,
                        label=f"{name} (eval)")
        _style(ax, metric.replace("_", " "), "episode")
    if len(logs) > 1 or any_eval:
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2,
                   frameon=False, labelcolor=INK)
    fig.suptitle(f"Training ({window}-episode moving average)", color=INK, x=0.01, ha="left")
    _save(fig, out)


def _eval_series(rows, metric):
    """(episodes, mean value) for the mode=eval rows, which come in groups sharing an episode."""
    by_episode = {}
    for r in rows:
        if r.get("mode") == "eval":
            by_episode.setdefault(int(r["episode"]), []).append(float(r[metric]))
    episodes = sorted(by_episode)
    return episodes, [sum(by_episode[e]) / len(by_episode[e]) for e in episodes]


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
