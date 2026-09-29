"""Training with several envs at once. --envs 1 keeps the plain loop, so nothing changes by default.

rl/loop.py stays the one-episode loop that rl.bench, isaac/demo.py and every evaluation use. This
is only the training path taken by --envs N, and it writes the same train_log.csv columns.

THE ENVS RUN IN LOCKSTEP ON A SHARED MAP, deliberately. A policy gets one `spec`, and both things
it has to work out for itself -- action legality and target distance -- are functions of that one
grid. Per-slot mazes would mean a spec per slot: an interface change every colleague's file would
absorb. Since every episode runs to the same step limit and the env never terminates, the slots
end together and the map changes for all of them at once.

The cost: a rollout shorter than an episode sees one maze, so a few consecutive updates are
correlated by map -- three per maze out of thousands, which a small-step method tolerates. Map
randomisation itself is not lost; every round draws a new maze, and robot starts, humans, crates
and the job sequence are redrawn every episode per slot.
"""
import time

from rl.checkpoint import save_checkpoint
from rl.cli import env_args, layout_for, make_env
from rl.loop import _as_ints
from rl.vecenv import VecWarehouse


def train_vec(args, env, policy, log, evaluate):
    """Train with args.envs envs in lockstep. evaluate(episode) is called at the eval points."""
    envs = [env] + [make_env(args, env.spec.obs_features, layout_for(args, 0))
                    for _ in range(args.envs - 1)]

    def episode(n):
        """Env seed and map for the n-th episode played: a seed each, a map per round."""
        return args.seed + n, layout_for(args, n // args.envs)

    vec = VecWarehouse(envs, episode)
    obs = vec.reset()
    policy.spec = envs[0].spec
    policy.training = args.train
    policy.reset()
    started = time.perf_counter()
    while _keep_going(args, vec):
        actions = [_as_ints(one) for one in policy.act_vec(obs)]
        nxt, rewards, dones, ended, stats = vec.step(actions)
        if args.train:
            # A slot that ran out of time must learn from the state it ENDED on: nxt has already
            # moved to an unrelated episode, and the env never terminates, so that return is real.
            policy.update_vec(obs, actions, rewards,
                              [e if e is not None else o for e, o in zip(ended, nxt)], dones)
        obs = nxt
        if not stats:
            continue
        before = vec.finished - len(stats)
        policy.spec = envs[0].spec      # lockstep: the map has just changed for every slot
        policy.reset()
        for offset, row in enumerate(stats):
            if log:
                log.write(before + offset + 1, dict(row, **policy.stats()))
        _report(args, vec, stats, started)
        if _crossed(before, vec.finished, args.eval_every):
            evaluate(vec.finished)
            # evaluate() plays on its own env, on its own held-out map, through run_episode --
            # which sets policy.spec and policy.training and does not put them back. Leaving
            # them is not a crash: the policy simply builds its action masks from the evaluation
            # maze for a whole training round, and plays hundreds of illegal moves it is fined
            # for. It showed up only as a spike in the invalid column of the learning curve.
            policy.spec = envs[0].spec
            policy.training = args.train
            policy.reset()
        if _crossed(before, vec.finished, args.snapshot_every):
            save_checkpoint(policy, f"{args.save}/ep_{vec.finished:06d}", args.policy,
                            env.n_robots, env_args(args), episode=vec.finished,
                            policy_opts=args.policy_opts)
    return vec


def _keep_going(args, vec):
    return vec.finished < args.episodes and (not args.total_steps or vec.steps < args.total_steps)


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
          f"mean return {mean['mean_return']:.2f}  [{vec.steps} steps, {rate:.0f}/s]")
