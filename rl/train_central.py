"""Entraîne CentralQPolicy puis la compare au glouton.

    python -m rl.train_central --episodes 20000
"""
import argparse
import os
import time

import numpy as np

from env.warehouse import Warehouse
from rl.policies.central_q import CentralQPolicy
from rl.policies.greedy import GreedyPolicy
from rl.run import run_episode


def train(env, policy, episodes, eps_start=1.0, eps_end=0.05, decay_frac=0.7, log_every=2000):
    history = []
    t0 = time.time()
    for ep in range(episodes):
        # epsilon décroît linéairement sur les premiers decay_frac des épisodes
        policy.epsilon = max(eps_end, eps_start - (eps_start - eps_end) * ep / (decay_frac * episodes))
        obs = env.reset(ep)
        done, deliveries = False, 0
        while not done:
            actions = policy.act_all(obs)
            next_obs, rewards, done, info = env.step(actions)
            policy.update_all(obs, actions, rewards, next_obs, done)
            obs = next_obs
            deliveries += info["deliveries"]
        history.append(deliveries)
        if (ep + 1) % log_every == 0:
            print(f"épisode {ep + 1}  epsilon {policy.epsilon:.2f}  "
                  f"livraisons moyennes {np.mean(history[-log_every:]):.2f}  "
                  f"états vus {len(policy.Q)}  {time.time() - t0:.0f}s")
    return history


def evaluate(env, policy, n=200, seed=10_000):
    """Moyenne des livraisons sur n épisodes, sans exploration ni apprentissage."""
    scores = []
    for k in range(n):
        obs = env.reset(seed + k)
        done, deliveries = False, 0
        while not done:
            obs, _, done, info = env.step(policy.act_all(obs))
            deliveries += info["deliveries"]
        scores.append(deliveries)
    return np.mean(scores), np.std(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--out", default="models/central_q.pkl")
    args = parser.parse_args()

    env = Warehouse(n_robots=args.robots, max_steps=args.max_steps)
    policy = CentralQPolicy(args.robots)
    train(env, policy, args.episodes)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    policy.save(args.out)

    policy.epsilon = 0.0
    q_mean, q_std = evaluate(env, policy)
    greedy = [GreedyPolicy(env.grid, seed=0)] * args.robots
    g = [run_episode(env, greedy, 10_000 + k, False, 0)["deliveries"] for k in range(200)]
    print(f"\nÉvaluation sur 200 épisodes :")
    print(f"  Q-learning central : {q_mean:.2f} livraisons (écart-type {q_std:.2f})")
    print(f"  Glouton            : {np.mean(g):.2f} livraisons (écart-type {np.std(g):.2f})")


if __name__ == "__main__":
    main()
