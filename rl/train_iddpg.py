"""Train IDDPG and compare it against the greedy baseline.

Run from the project root:
    python -m rl.train_iddpg --episodes 2000
"""

import argparse
import os
import time

import numpy as np

from env.warehouse import Warehouse
from rl.policies.iddpg import IDDPG
from rl.policies.greedy import GreedyPolicy
from rl.run import run_episode


def train(
    env,
    policy,
    episodes,
    eps_start=1.0,
    eps_end=0.05,
    decay_frac=0.7,
    log_every=200,
):
    history = []
    start_time = time.time()

    for ep in range(episodes):
        # Reduce random exploration over the first part of training.
        progress = min(1.0, ep / max(1, int(decay_frac * episodes)))
        epsilon = eps_start + progress * (eps_end - eps_start)

        for agent in policy.agents:
            agent.epsilon = epsilon

        obs = env.reset(ep)
        done = False
        deliveries = 0

        while not done:
            actions = policy.act_all(obs)
            next_obs, rewards, done, info = env.step(actions)

            policy.update_all(
                obs, actions, rewards, next_obs, done
            )

            obs = next_obs
            deliveries += info["deliveries"]

        history.append(deliveries)

        if (ep + 1) % log_every == 0:
            recent = history[-log_every:]
            print(
                f"Episode {ep + 1} | "
                f"epsilon {epsilon:.2f} | "
                f"mean deliveries {np.mean(recent):.2f} | "
                f"time {time.time() - start_time:.0f}s"
            )

    return history


def evaluate(env, policy, n=200, seed=10_000):
    """Evaluate without exploration or learning."""
    scores = []

    # Remember the exploration settings so we can restore them.
    old_epsilons = [agent.epsilon for agent in policy.agents]

    try:
        for agent in policy.agents:
            agent.epsilon = 0.0

        for k in range(n):
            obs = env.reset(seed + k)
            done = False
            deliveries = 0

            while not done:
                actions = policy.act_all(obs)
                obs, _, done, info = env.step(actions)
                deliveries += info["deliveries"]

            scores.append(deliveries)
    finally:
        for agent, epsilon in zip(policy.agents, old_epsilons):
            agent.epsilon = epsilon

    return float(np.mean(scores)), float(np.std(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--out", default="models/iddpg.pt")
    args = parser.parse_args()

    env = Warehouse(
        n_robots=args.robots,
        max_steps=args.max_steps,
    )

    policy = IDDPG(num_agents=args.robots)

    print("Training IDDPG...")
    train(env, policy, args.episodes)

    # Create the output directory if necessary.
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    policy.save(args.out)
    print(f"Model saved to: {args.out}")

    # Evaluate the trained policy.
    iddpg_mean, iddpg_std = evaluate(
        env, policy, n=args.eval_episodes
    )

    # Evaluate the greedy baseline on the same number of episodes.
    greedy_policies = [
        GreedyPolicy(env.grid, seed=0)
        for _ in range(args.robots)
    ]

    greedy_scores = [
        run_episode(
            env,
            greedy_policies,
            10_000 + k,
            False,
            0,
        )["deliveries"]
        for k in range(args.eval_episodes)
    ]

    print(f"\nEvaluation over {args.eval_episodes} episodes:")
    print(
        f"  IDDPG  : {iddpg_mean:.2f} deliveries "
        f"(std {iddpg_std:.2f})"
    )
    print(
        f"  Greedy : {np.mean(greedy_scores):.2f} deliveries "
        f"(std {np.std(greedy_scores):.2f})"
    )


if __name__ == "__main__":
    main()