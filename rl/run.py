"""Train, evaluate or watch a policy in the terminal.

    python -m rl.run                                              # one rendered episode, greedy, default map
    python -m rl.run --maze --seed 3 --max-steps 200              # a random maze of racks
    python -m rl.run --scenario hard                              # a preset of flags (env/scenarios.py)
    python -m rl.run --policy myalgo --train --episodes 5000 --no-render --save checkpoints/myalgo --snapshot-every 500
    python -m rl.run --checkpoint checkpoints/myalgo --episodes 100 --no-render    # evaluate, same settings as training
    python -m rl.plot checkpoints/myalgo                          # learning curve of that training run

Training with --save writes FOLDER/train_log.csv (one row per episode) and, with
--snapshot-every N, a replayable checkpoint FOLDER/ep_000500/ every N episodes.
isaac/demo.py takes the same flags, so both show the same episode.
"""
import csv
import time
from pathlib import Path

from rl.checkpoint import save_checkpoint
from rl.cli import build_or_exit, env_args, layout_for, make_parser, parse
from rl.loop import run_episode

LOG_FIELDS = ("episode", "deliveries", "value", "blocked", "invalid", "mean_return")


def main():
    parser = make_parser(__doc__)
    parser.add_argument("--train", action="store_true", help="call the policy's update() after each step")
    parser.add_argument("--save", metavar="FOLDER", help="with --train: checkpoint + train_log.csv go here")
    parser.add_argument("--snapshot-every", type=int, default=0, metavar="N",
                        help="with --save: also keep a checkpoint every N episodes, to replay later")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=1, metavar="N", help="print every N-th episode")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.15, help="seconds between rendered steps")
    args = parse(parser)
    if args.episodes < 1 or args.print_every < 1 or args.snapshot_every < 0:
        parser.error("--episodes and --print-every must be at least 1, --snapshot-every at least 0")
    if args.save and not args.train:
        parser.error("--save only makes sense with --train")
    if args.snapshot_every and not args.save:
        parser.error("--snapshot-every needs --save")
    env, policy = build_or_exit(parser, args)

    def show(*_):
        print(env.render() + "\n")
        time.sleep(args.delay)

    log = TrainLog(args.save) if args.save else None
    for k in range(args.episodes):
        try:
            layout = layout_for(args, k) if args.new_map else None
        except ValueError as err:
            parser.error(str(err))
        stats = run_episode(env, policy, args.seed + k, args.train, None if args.no_render else show, layout)
        if log:
            log.write(k, stats)
        if (k + 1) % args.print_every == 0 or k == args.episodes - 1:
            print(f"episode {k}: deliveries {stats['deliveries']}  value {stats['value']:.1f}  "
                  f"blocked {stats['blocked']}  invalid {stats['invalid']}  mean return {stats['mean_return']:.2f}")
        if args.snapshot_every and (k + 1) % args.snapshot_every == 0:
            save_checkpoint(policy, Path(args.save) / f"ep_{k + 1:06d}", args.policy, env.n_robots,
                            env_args(args), episode=k + 1)
    if log:
        log.close()
    if args.save:
        save_checkpoint(policy, args.save, args.policy, env.n_robots, env_args(args), episode=args.episodes)
        print(f"saved checkpoint to {args.save}")


class TrainLog:
    """FOLDER/train_log.csv: one row per training episode, for rl.plot."""

    def __init__(self, folder):
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.file = open(Path(folder) / "train_log.csv", "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(LOG_FIELDS)

    def write(self, episode, stats):
        self.writer.writerow([episode] + [round(stats[k], 4) for k in LOG_FIELDS[1:]])

    def close(self):
        self.file.close()


if __name__ == "__main__":
    main()
