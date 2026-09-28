"""Benchmark: every algorithm on every scenario, same seeds, no learning.

    python -m rl.bench greedy myalgo=checkpoints/myalgo --scenarios basic mixed hard --episodes 50

Each entry is a registered policy name, or name=CHECKPOINT_FOLDER for a trained model
(a colleague's folder works as-is). Prints a table, and writes
results/bench.csv (one row per episode) and results/bench.png (one chart per metric).
"""
import argparse
import csv
import statistics
from pathlib import Path

from env.scenarios import SCENARIOS
from rl.cli import layout_for, make_parser, parse, build
from rl.loop import run_episode

METRICS = ("deliveries", "value", "blocked", "invalid", "mean_return")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("entries", nargs="+", metavar="POLICY[=CHECKPOINT]")
    parser.add_argument("--scenarios", nargs="+", default=["basic"], help=f"from: {', '.join(SCENARIOS)}")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000, help="evaluation seeds: seed + k (unlike training's 0 + k)")
    parser.add_argument("--out", default="results/bench", help="writes OUT.csv and OUT.png")
    args = parser.parse_args()
    unknown = [s for s in args.scenarios if s not in SCENARIOS]
    if unknown or args.episodes < 1:
        parser.error(f"--episodes must be at least 1; scenarios must be in: {', '.join(SCENARIOS)}")

    rows = []
    for entry in args.entries:
        for scenario in args.scenarios:
            try:
                rows += evaluate(entry, scenario, args.episodes, args.seed)
            except ValueError as err:
                print(f"skipped {entry} on {scenario}: {err}")
    if not rows:
        parser.error("nothing could be evaluated")
    print_table(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "scenario", "episode", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out.with_suffix('.csv')}")
    try:
        from rl.plot import bench_chart
    except ImportError:
        print("matplotlib is not installed (pip install -r requirements.txt): no chart")
        return
    bench_chart(rows, out.with_suffix(".png"))
    print(f"wrote {out.with_suffix('.png')}")


def evaluate(entry, scenario, episodes, seed):
    """Per-episode result rows for one algorithm on one scenario."""
    name, _, checkpoint = entry.partition("=")
    argv = ["--scenario", scenario, "--seed", str(seed)]
    if name:
        argv += ["--policy", name]
    if checkpoint:
        argv += ["--checkpoint", checkpoint]
    # Only the scenario sets the env, never the checkpoint's training settings: a fair comparison.
    args = parse(make_parser(parser_class=_Raising), argv, env_from_checkpoint=False)
    env, policy = build(args)
    rows = []
    for k in range(episodes):
        layout = layout_for(args, k) if args.new_map else None
        stats = run_episode(env, policy, args.seed + k, train=False, layout=layout)
        rows.append({"algorithm": entry, "scenario": scenario, "episode": k,
                     **{m: round(stats[m], 4) for m in METRICS}})
    return rows


class _Raising(argparse.ArgumentParser):
    """Errors become ValueError, so one bad entry does not stop the whole benchmark."""

    def error(self, message):
        raise ValueError(message)


def print_table(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["algorithm"], row["scenario"]), []).append(row)
    width = max(len(a) for a, _ in groups) + 2
    print(f"\n{'algorithm':<{width}}{'scenario':<11}" + "".join(f"{m:>18}" for m in METRICS))
    for (algorithm, scenario), group in groups.items():
        cells = []
        for m in METRICS:
            values = [r[m] for r in group]
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{statistics.mean(values):>10.2f} +- {spread:<5.2f}")
        print(f"{algorithm:<{width}}{scenario:<11}" + "".join(cells))
    print("(mean +- std over episodes; value = priority-weighted deliveries)\n")


if __name__ == "__main__":
    main()
