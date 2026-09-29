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
import argparse
import csv
import time
from pathlib import Path

from env.scenarios import SCENARIOS
from rl.checkpoint import save_checkpoint
from rl.cli import build_or_exit, env_args, layout_for, make_env, make_parser, parse, train_seeds
from rl.loop import run_episode

# Append-only: older logs without the later columns still read back, and the first six
# columns keep their meaning and order for anything that reads them by position.
LOG_FIELDS = ("episode", "deliveries", "value", "blocked", "invalid", "mean_return",
              "mode", "epsilon", "p_expert", "loss")


def build_parser():
    """rl.run's own flags on top of the shared ones. Separate from main() so tests can use it."""
    parser = make_parser(__doc__)
    parser.add_argument("--train", action="store_true", help="call the policy's update() after each step")
    parser.add_argument("--save", metavar="FOLDER", help="with --train: checkpoint + train_log.csv go here")
    parser.add_argument("--snapshot-every", type=int, default=0, metavar="N",
                        help="with --save: also keep a checkpoint every N episodes, to replay later")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=1, metavar="N", help="print every N-th episode")
    parser.add_argument("--eval-every", type=int, default=0, metavar="N",
                        help="with --train: every N episodes, play --eval-episodes without learning,\n"
                             "exploration or a teacher. This is the curve that shows what the net learned")
    parser.add_argument("--eval-episodes", type=int, default=5, metavar="K")
    parser.add_argument("--eval-seed", type=int, default=90_000,
                        help="evaluation episodes use eval-seed + j, held out from training's seed + k")
    parser.add_argument("--eval-scenario", help="evaluate on a different preset than training")
    parser.add_argument("--eval-robots", type=int, default=None, metavar="N",
                        help="evaluate with a different fleet size than training. A policy whose\n"
                             "net does not depend on the robot count transfers; one built by\n"
                             "independent() cannot")
    parser.add_argument("--eval-size", type=int, nargs=2, default=None, metavar=("W", "H"),
                        help="evaluate on a different maze size than training")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.15, help="seconds between rendered steps")
    return parser


def check_args(parser, args):
    """Refuse flag combinations that would quietly produce a meaningless measurement."""
    if args.episodes < 1 or args.print_every < 1 or args.snapshot_every < 0:
        parser.error("--episodes and --print-every must be at least 1, --snapshot-every at least 0")
    if args.save and not args.train:
        parser.error("--save only makes sense with --train")
    if args.snapshot_every and not args.save:
        parser.error("--snapshot-every needs --save")
    if args.eval_every and not args.train:
        parser.error("--eval-every needs --train (without it every episode is already an evaluation)")
    if args.eval_every < 0 or args.eval_episodes < 1:
        parser.error("--eval-every must be at least 0 and --eval-episodes at least 1")
    if args.eval_scenario and args.eval_scenario not in SCENARIOS:
        parser.error(f"unknown --eval-scenario '{args.eval_scenario}'; in env/scenarios.py: "
                     f"{', '.join(SCENARIOS)}")
    if args.eval_every:
        # Held out means held out. Without this check a long --new-map run walks its layout seeds
        # straight into the evaluation range and quietly starts training on the eval mazes.
        train_lo, train_hi = train_seeds(args, args.episodes)
        eval_lo, eval_hi = args.eval_seed, args.eval_seed + args.eval_episodes - 1
        if train_lo <= eval_hi and eval_lo <= train_hi:
            parser.error(f"training maps (seeds {train_lo}-{train_hi}) overlap the evaluation maps "
                         f"(seeds {eval_lo}-{eval_hi}): raise --eval-seed, or bound training's "
                         f"range with --maze-pool")


def main():
    parser = build_parser()
    args = parse(parser)
    check_args(parser, args)
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
            log.write(k, dict(stats, **policy.stats()))
        if (k + 1) % args.print_every == 0 or k == args.episodes - 1:
            print(f"episode {k}: deliveries {stats['deliveries']}  value {stats['value']:.1f}  "
                  f"blocked {stats['blocked']}  invalid {stats['invalid']}  mean return {stats['mean_return']:.2f}")
        if args.eval_every and (k + 1) % args.eval_every == 0:
            evaluate(env, policy, args, k + 1, log)
        if args.snapshot_every and (k + 1) % args.snapshot_every == 0:
            save_checkpoint(policy, Path(args.save) / f"ep_{k + 1:06d}", args.policy, env.n_robots,
                            env_args(args), episode=k + 1, policy_opts=args.policy_opts)
    if log:
        log.close()
    if args.save:
        save_checkpoint(policy, args.save, args.policy, env.n_robots, env_args(args),
                        episode=args.episodes, policy_opts=args.policy_opts)
        print(f"saved checkpoint to {args.save}")


def evaluate(env, policy, args, episode, log):
    """Play held-out episodes with learning, exploration and any teacher switched off.

    run_episode(train=False) sets policy.training = False and never calls update(), so this
    measures the policy alone.

    Evaluation gets its OWN env and an explicit map per episode. It used to be handed the
    training env with layout=None, which means "keep whatever map is installed" -- so under
    --new-map it measured the policy on the maze it had just finished training on, and the eval
    map moved at every eval point, which made the curve incomparable with itself. Its own env is
    also what lets evaluation use a different fleet or maze size than training, which is how the
    generalisation numbers are measured.
    """
    settings = eval_settings(args)
    eval_env = make_env(settings, env.spec.obs_features, layout_for(settings, 0))
    rows = [run_episode(eval_env, policy, args.eval_seed + j, train=False,
                        layout=layout_for(settings, j))
            for j in range(args.eval_episodes)]
    if log:
        for row in rows:
            # No policy.stats() here: epsilon/p_expert/loss describe the training schedule and
            # would read as if they applied to these episodes.
            log.write(episode, row, mode="eval")
    mean = {m: sum(r[m] for r in rows) / len(rows) for m in ("deliveries", "invalid", "mean_return")}
    print(f"  eval @ {episode}: deliveries {mean['deliveries']:.1f}  "
          f"invalid {mean['invalid']:.0f}  mean return {mean['mean_return']:.2f}")


def eval_settings(args):
    """The env flags as evaluation sees them: held-out maps, optionally a different size or fleet.

    new_map is forced on whenever the map is random at all, so the evaluation mazes come from
    --eval-seed rather than from training's seed. For a run that trained on one fixed maze that
    turns evaluation into the more interesting question of whether the policy generalises off it.
    """
    settings = argparse.Namespace(**vars(args))
    for key, value in SCENARIOS.get(args.eval_scenario or "", {}).items():
        setattr(settings, key, value)
    settings.seed = args.eval_seed
    settings.new_map = settings.map != "default"
    settings.maze_pool = 0      # a pool would cycle; every eval episode gets its own map
    settings.robots = args.eval_robots or settings.robots
    settings.size = args.eval_size or settings.size
    return settings


class TrainLog:
    """FOLDER/train_log.csv: one row per episode, for rl.plot. mode is "train" or "eval"."""

    def __init__(self, folder):
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.file = open(Path(folder) / "train_log.csv", "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(LOG_FIELDS)

    def write(self, episode, stats, mode="train"):
        row = dict(stats, mode=mode)
        self.writer.writerow([episode] + [_cell(row.get(k)) for k in LOG_FIELDS[1:]])
        self.file.flush()   # so rl.plot works on a run that is still going

    def close(self):
        self.file.close()


def _cell(value):
    """A metric the policy did not report becomes an empty cell, not a crash."""
    if value is None:
        return ""
    return round(value, 4) if isinstance(value, float) else value


if __name__ == "__main__":
    main()
