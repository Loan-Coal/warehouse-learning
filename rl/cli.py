"""The flags shared by rl.run, rl.bench and isaac.demo, and building (env, policy) from them.

Where a setting comes from, strongest first:
    1. a flag you type            --humans 3
    2. the scenario               --scenario hard       (env/scenarios.py)
    3. the checkpoint             --checkpoint FOLDER   (the flags it was trained with, from meta.json)
    4. the default below
So `--checkpoint checkpoints/myalgo` alone replays a model with the settings it was trained on.
"""
import argparse

from env.layout import DEFAULT_MAP, maze_layout, random_layout
from env.scenarios import SCENARIOS
from env.warehouse import Warehouse
from rl.checkpoint import load_checkpoint, read_meta
from rl.policies import POLICIES

# Flags that describe the env; a checkpoint records them so it can be replayed as trained.
ENV_KEYS = ("seed", "robots", "map", "size", "max_steps", "new_map",
            "mixed_crates", "mixed_robots", "priority", "humans")


def make_parser(description=None, parser_class=argparse.ArgumentParser):
    parser = parser_class(description=description, formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(parser)
    return parser


def add_common_args(parser):
    parser.add_argument("--policy", help=f"one of: {', '.join(POLICIES)} (default greedy)")
    parser.add_argument("--checkpoint", metavar="FOLDER", help="load trained weights (and their settings)")
    parser.add_argument("--scenario", help=f"preset of the flags below: {', '.join(SCENARIOS)}")
    parser.add_argument("--seed", type=int, default=0, help="episode k uses seed + k")
    parser.add_argument("--robots", type=int, default=2)
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--random-map", dest="map", action="store_const", const="random",
                      help="random warehouse from --seed")
    maps.add_argument("--maze", dest="map", action="store_const", const="maze",
                      help="random maze of racks from --seed")
    parser.set_defaults(map="default")
    parser.add_argument("--size", type=int, nargs=2, default=(11, 9), metavar=("W", "H"),
                        help="maze width and height, both odd (default 11 9)")
    parser.add_argument("--max-steps", type=int, default=100, help="episode length")
    parser.add_argument("--new-map", action="store_true",
                        help="with --maze or --random-map: a new map every episode (seed + k)")
    factors = parser.add_argument_group("complexity factors (all off by default)")
    factors.add_argument("--mixed-crates", action="store_true", help="crates get a weight and volume 1-3")
    factors.add_argument("--mixed-robots", action="store_true",
                         help="robots differ: small fast ones, large slow ones that carry more")
    factors.add_argument("--priority", action="store_true",
                         help="jobs have a priority 1-3; delivery value decays with job age")
    factors.add_argument("--humans", type=int, default=0, help="humans walking in the aisles")


def parse(parser, argv=None, env_from_checkpoint=True):
    """parser.parse_args, with defaults filled from --checkpoint and --scenario.

    env_from_checkpoint=False takes only the policy name from the checkpoint (rl.bench: every
    entry must play exactly the same env, whatever it was trained on).
    """
    pre, _ = parser.parse_known_args(argv)
    defaults = {}
    if pre.checkpoint:
        try:
            meta = read_meta(pre.checkpoint)
        except ValueError as err:
            parser.error(str(err))
        if env_from_checkpoint:
            defaults.update(meta.get("env_args", {}))
        defaults["policy"] = meta["policy"]
    if pre.scenario:
        if pre.scenario not in SCENARIOS:
            parser.error(f"unknown scenario '{pre.scenario}'; in env/scenarios.py: {', '.join(SCENARIOS)}")
        defaults.update(SCENARIOS[pre.scenario])
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.policy = args.policy or "greedy"
    return args


def env_args(args):
    """The env flags as a plain dict (stored in checkpoints and benchmark results)."""
    return {key: getattr(args, key) for key in ENV_KEYS}


def layout_for(args, episode):
    """The map for episode k: a fixed map, or a new one per episode with --new-map."""
    seed = args.seed + episode if args.new_map else args.seed
    if args.map == "maze":
        return maze_layout(seed, *args.size, n_robots=args.robots)
    if args.map == "random":
        return random_layout(seed, n_robots=args.robots)
    if args.new_map:
        raise ValueError("--new-map needs --maze or --random-map")
    return DEFAULT_MAP


def build(args):
    """(env, policy) from parsed args. Raises ValueError with a readable message on bad settings."""
    if args.policy not in POLICIES:
        raise ValueError(f"unknown policy '{args.policy}'; registered in rl/policies/__init__.py: "
                         f"{', '.join(POLICIES)}")
    make = POLICIES[args.policy]
    env = Warehouse(layout_for(args, 0), n_robots=args.robots, max_steps=args.max_steps, obs=make.OBS,
                    mixed_crates=args.mixed_crates, mixed_robots=args.mixed_robots,
                    priority=args.priority, humans=args.humans)
    policy = make(env.spec, args.seed)
    if args.checkpoint:
        load_checkpoint(policy, args.checkpoint, args.policy)
    return env, policy


def build_or_exit(parser, args):
    """build(), turning bad settings into a parser error instead of a traceback."""
    try:
        return build(args)
    except ValueError as err:
        parser.error(str(err))
