"""Training with several envs at once. --envs 1 is the same loop with a single slot.

rl/loop.py stays the one-episode loop that rl.bench, isaac/demo.py and every evaluation use; this
is the training path, and it writes the train_log.csv columns.

THE ENVS RUN IN LOCKSTEP ON A SHARED MAP, deliberately. A policy gets one `spec`, and both things
it works out for itself -- action legality and target distance -- are functions of that grid, so
per-slot mazes would mean a spec per slot: an interface change every colleague absorbs. Episodes
all run to the same step limit, so the slots end together and the map changes for all at once.
Cost: a rollout sees one maze, so ~3 consecutive updates are correlated by map, out of thousands.
Map randomisation is not lost -- every round draws a new maze, and starts, humans, crates and
jobs are redrawn per episode per slot.
"""
import time

from rl.cli import layout_for, make_env
from rl.loop import _as_ints
from rl.metrics import format_rates
from rl.vecenv import VecWarehouse


def train_vec(args, env, policy, log, evaluate, save=None, show=None):
    """Train with args.envs envs in lockstep.

    evaluate(episode) is called at the eval points, save(folder, episode) at the snapshots, and
    show() after every step (a single env only: the caller refuses to render several).
    """
    envs = [env] + [make_env(args, env.spec.obs_features, layout_for(args, 0))
                    for _ in range(args.envs - 1)]

    def episode(n):        # a seed per episode, a map per round of --envs episodes
        return args.seed + n, layout_for(args, n // args.envs)

    vec = VecWarehouse(envs, episode)
    obs = vec.reset()
    _begin_round(policy, envs, args.train)
    started = time.perf_counter()
    deadline = started + 60 * args.max_minutes if args.max_minutes else None
    while vec.finished < args.episodes and (deadline is None or time.perf_counter() < deadline):
        actions = [_as_ints(one) for one in policy.act_vec(obs)]
        nxt, rewards, dones, ended, stats = vec.step(actions)
        if args.train:
            # A finished slot must learn from the state it ENDED on: nxt is another episode.
            policy.update_vec(obs, actions, rewards,
                              [e if e is not None else o for e, o in zip(ended, nxt)], dones)
        if show:
            show()
        obs = nxt
        if not stats:
            continue
        before = vec.finished - len(stats)
        _begin_round(policy, envs, args.train)      # lockstep: the map changed for every slot
        for offset, row in enumerate(stats):
            if log:
                log.write(before + offset + 1, dict(row, **policy.stats()))
        _report(args, vec, stats, started)
        if _crossed(before, vec.finished, args.eval_every):
            evaluate(vec.finished)
            # evaluate() plays on its own env through run_episode, which leaves policy.spec on the
            # evaluation map and policy.training False; the next round must not inherit either.
            _begin_round(policy, envs, args.train)
        if save and _crossed(before, vec.finished, args.snapshot_every):
            save(f"{args.save}/ep_{vec.finished:06d}", vec.finished)
    return vec


def _begin_round(policy, envs, training):
    """Point the policy at the training map and mode, and start its episode state afresh."""
    policy.spec = envs[0].spec
    policy.training = training
    policy.reset()


def _crossed(before, after, every):
    """True when a multiple of `every` falls in (before, after]; N episodes land at once."""
    return bool(every) and after // every > before // every


def _report(args, vec, stats, started):
    if not _crossed(vec.finished - len(stats), vec.finished, args.print_every):
        return
    mean = {m: sum(r[m] for r in stats) / len(stats)
            for m in ("deliveries", "value", "blocked", "invalid", "mean_return")}
    rate = vec.steps / max(time.perf_counter() - started, 1e-9)
    print(f"episode {vec.finished}: deliveries {mean['deliveries']:.1f}  value {mean['value']:.1f}  "
          f"blocked {mean['blocked']:.1f}  invalid {mean['invalid']:.1f}  "
          f"mean return {mean['mean_return']:.2f}  {format_rates(stats)}  "
          f"[{vec.steps} steps, {rate:.0f}/s]")
