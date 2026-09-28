"""Run episodes of the warehouse with one policy per robot, in the terminal.

    python -m rl.run                                   # one rendered episode, default map
    python -m rl.run --random-map --seed 3             # a random warehouse
    python -m rl.run --maze --seed 3 --max-steps 200   # a random maze of racks
    python -m rl.run --episodes 500 --no-render        # the future training loop
"""
import argparse
import time

from env.layout import DEFAULT_MAP, maze_layout, random_layout
from env.warehouse import Warehouse
from rl.policies.greedy import GreedyPolicy


def make_policies(env, seed):
    # One shared object in every slot = parameter sharing. For independent learners,
    # build a separate instance per robot instead. Swapping algorithms happens here only.
    return [GreedyPolicy(env.grid, seed=seed)] * env.n_robots


def run_episode(env, policies, seed, render, delay):
    obs = env.reset(seed)
    for policy in {id(p): p for p in policies}.values():   # once per distinct object
        if hasattr(policy, "reset"):
            policy.reset()
    returns = [0.0] * env.n_robots
    done = False
    while not done:
        actions = [p.act(o) for p, o in zip(policies, obs)]
        next_obs, rewards, done, _ = env.step(actions)
        for i, p in enumerate(policies):
            if hasattr(p, "update"):
                p.update(obs[i], actions[i], rewards[i], next_obs[i], done)
            returns[i] += rewards[i]
        obs = next_obs
        if render:
            print(env.render() + "\n")
            time.sleep(delay)
    return dict(env.totals, mean_return=sum(returns) / len(returns))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="episode k uses seed + k")
    parser.add_argument("--robots", type=int, default=2)
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--random-map", action="store_true", help="random warehouse from --seed")
    maps.add_argument("--maze", action="store_true", help="random maze of racks from --seed")
    parser.add_argument("--size", type=int, nargs=2, default=(11, 9), metavar=("W", "H"),
                        help="maze width and height, both odd (default 11 9)")
    parser.add_argument("--max-steps", type=int, default=100, help="episode length")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.15, help="seconds between rendered steps")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    try:
        if args.maze:
            layout = maze_layout(args.seed, *args.size, n_robots=args.robots)
        elif args.random_map:
            layout = random_layout(args.seed, n_robots=args.robots)
        else:
            layout = DEFAULT_MAP
        env = Warehouse(layout, n_robots=args.robots, max_steps=args.max_steps)
    except ValueError as err:
        parser.error(str(err))
    policies = make_policies(env, args.seed)

    for k in range(args.episodes):
        stats = run_episode(env, policies, args.seed + k, not args.no_render, args.delay)
        print(f"episode {k}: deliveries {stats['deliveries']}  blocked {stats['blocked']}  "
              f"invalid {stats['invalid']}  mean return {stats['mean_return']:.2f}")


if __name__ == "__main__":
    main()
