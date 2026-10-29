"""Where a centralised Q-table stops working: 2, 3, 4 robots on the default 7x6 map.

    python -m rl.qtable_report --robots 2 3 4 --episodes 100

For each n it loads checkpoints/q_<n> (central_q) and checkpoints/ppo_<n>, plays them and greedy
without learning on the same held-out seeds (900000000 + k, never reached by training), and
prints a markdown table, also written to results/qtable_table.md. A missing checkpoint shows "—".
"""
import argparse
import statistics
from pathlib import Path

from env.layout import DEFAULT_MAP
from env.warehouse import N_ACTIONS, Warehouse
from rl.checkpoint import load_checkpoint, read_meta
from rl.loop import run_episode
from rl.policies import POLICIES

SEED = 900_000_000
AISLE_CELLS, HEADINGS, TASK_SITUATIONS = 34, 4, 17     # default map: one robot's (position, heading, task)
DASH = "—"
COLUMNS = ("Robots", "Actions jointes", "États théoriques", "États visités", "Mémoire Q (Mo)",
           "Livraisons Q-table", "Livraisons glouton", "Livraisons PPO", "États inconnus en éval (%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--robots", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoints", default="checkpoints", help="holds q_<n>/ and ppo_<n>/")
    parser.add_argument("--max-steps", type=int, default=100, help="episode length (rl.run's default)")
    parser.add_argument("--out", default="results/qtable_table.md")
    args = parser.parse_args()
    if args.episodes < 1 or min(args.robots) < 1:
        parser.error("--episodes and --robots must be at least 1")

    rows = [report_row(n, args) for n in args.robots]
    table = "\n".join(["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS),
                       *("| " + " | ".join(row) + " |" for row in rows)])
    note = (f"\n\n{args.episodes} épisodes d'évaluation par ligne, carte par défaut 7x6, graines "
            f"{SEED}+k, sans apprentissage. Livraisons : moyenne ± écart-type par épisode. États "
            f"théoriques : ({AISLE_CELLS}*{HEADINGS}*{TASK_SITUATIONS})^n.\n")
    print(table + note)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + note, encoding="utf-8")
    print(f"wrote {out}")


def report_row(n, args):
    folder = Path(args.checkpoints)
    q_policy, q_env = load(folder / f"q_{n}", n, args.max_steps)
    ppo_policy, ppo_env = load(folder / f"ppo_{n}", n, args.max_steps)
    greedy_env = make_env(POLICIES["greedy"].OBS, n, args.max_steps)
    greedy = POLICIES["greedy"](greedy_env.spec, 0)

    unknown = [0, 0]        # [unknown state steps, steps]
    if q_policy is not None:
        stats = q_policy.stats()
        visited, memory = f"{stats['states']:,}".replace(",", " "), f"{stats['table_mb']:.1f}"

        def count(obs, *_):
            unknown[0] += q_policy.state(obs) not in q_policy.q
            unknown[1] += 1
    else:
        visited = memory = DASH
        count = None

    theoretical = (AISLE_CELLS * HEADINGS * TASK_SITUATIONS) ** n
    return [str(n), f"{N_ACTIONS ** n:,}".replace(",", " "), f"{theoretical:.2e}", visited, memory,
            deliveries(q_env, q_policy, args.episodes, count),
            deliveries(greedy_env, greedy, args.episodes),
            deliveries(ppo_env, ppo_policy, args.episodes),
            f"{100 * unknown[0] / unknown[1]:.1f}" if unknown[1] else DASH]


def make_env(obs, n, max_steps):
    return Warehouse(DEFAULT_MAP, n_robots=n, max_steps=max_steps, obs=obs)


def load(folder, n, max_steps):
    """(policy, env) from a checkpoint folder, or (None, None) if it is missing or unusable."""
    try:
        meta = read_meta(folder)
        name = meta["policy"]
        if meta.get("n_robots") not in (None, n):
            raise ValueError(f"trained with {meta['n_robots']} robots, not {n}")
        make = POLICIES[name]
        env = make_env(make.OBS, n, max_steps)
        policy = make(env.spec, 0)
        policy.configure(**(meta.get("policy_opts") or {}))
        load_checkpoint(policy, folder, name)
    except (ValueError, KeyError) as err:
        print(f"{folder}: {err} -> {DASH}")
        return None, None
    return policy, env


def deliveries(env, policy, episodes, on_step=None):
    if policy is None:
        return DASH
    values = [run_episode(env, policy, SEED + k, train=False, on_step=on_step)["deliveries"]
              for k in range(episodes)]
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.2f} ± {spread:.2f}"


if __name__ == "__main__":
    main()
