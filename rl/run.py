"""Train, evaluate or watch a policy in the terminal.

    python -m rl.run                                              # one rendered episode, greedy, default map
    python -m rl.run --maze --seed 3 --max-steps 200              # a random maze of racks
    python -m rl.run --policy myalgo --train --episodes 5000 --no-render --save checkpoints/myalgo
    python -m rl.run --policy myalgo --checkpoint checkpoints/myalgo --episodes 100 --no-render

isaac/demo.py takes the same flags, so both show the same episode.
"""
import argparse
import time

from env.layout import DEFAULT_MAP, maze_layout, random_layout
from env.warehouse import Warehouse
from rl.checkpoint import load_checkpoint, save_checkpoint
from rl.loop import run_episode
from rl.policies import POLICIES


def add_common_args(parser):
    """Flags shared with isaac/demo.py: which env and which policy."""
    parser.add_argument("--policy", default="greedy", help=f"one of: {', '.join(POLICIES)}")
    parser.add_argument("--checkpoint", metavar="FOLDER", help="load trained weights before running")
    parser.add_argument("--seed", type=int, default=0, help="episode k uses seed + k")
    parser.add_argument("--robots", type=int, default=2)
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--random-map", action="store_true", help="random warehouse from --seed")
    maps.add_argument("--maze", action="store_true", help="random maze of racks from --seed")
    parser.add_argument("--size", type=int, nargs=2, default=(11, 9), metavar=("W", "H"),
                        help="maze width and height, both odd (default 11 9)")
    parser.add_argument("--max-steps", type=int, default=100, help="episode length")


def build(parser, args):
    """(env, policy) from the common flags; bad flags end in a parser error, not a traceback."""
    if args.policy not in POLICIES:
        parser.error(f"unknown policy '{args.policy}'; registered in rl/policies/__init__.py: "
                     f"{', '.join(POLICIES)}")
    try:
        if args.maze:
            layout = maze_layout(args.seed, *args.size, n_robots=args.robots)
        elif args.random_map:
            layout = random_layout(args.seed, n_robots=args.robots)
        else:
            layout = DEFAULT_MAP
        env = Warehouse(layout, n_robots=args.robots, max_steps=args.max_steps)
        policy = POLICIES[args.policy](env.spec, args.seed)
        if args.checkpoint:
            load_checkpoint(policy, args.checkpoint, args.policy)
    except ValueError as err:
        parser.error(str(err))
    return env, policy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--train", action="store_true", help="call the policy's update() after each step")
    parser.add_argument("--save", metavar="FOLDER", help="with --train: write a checkpoint at the end")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.15, help="seconds between rendered steps")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.save and not args.train:
        parser.error("--save only makes sense with --train")
    env, policy = build(parser, args)

    def show(*_):
        print(env.render() + "\n")
        time.sleep(args.delay)

    for k in range(args.episodes):
        stats = run_episode(env, policy, args.seed + k, args.train, None if args.no_render else show)
        print(f"episode {k}: deliveries {stats['deliveries']}  blocked {stats['blocked']}  "
              f"invalid {stats['invalid']}  mean return {stats['mean_return']:.2f}")
    if args.save:
        save_checkpoint(policy, args.save, args.policy, env.n_robots)
        print(f"saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
