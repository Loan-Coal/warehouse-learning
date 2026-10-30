"""rl/policies/search.py: PPO plus decision-time rollouts on re-seeded env copies.

What matters here is what the search is NOT allowed to do: touch the real env, or read its true
random future. The rest (k=0 is the prior, a swap is avoided) shows it does something useful.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import rl.policies.search as mod
from env.layout import maze_layout
from env.observation import LEFT, RIGHT
from env.warehouse import FORWARD, TURN_RIGHT, Task, Warehouse
from rl.checkpoint import save_checkpoint
from rl.model import clone as real_clone
from rl.nav import legal_actions
from rl.policies.ppo import PPO
from rl.policies.search import Search

ROBOTS, STEPS = 4, 40
FACTORS = dict(mixed_crates=True, mixed_robots=True, priority=True, humans=2)
CORRIDOR = """
.......
.SSSSS.
.......
D.....D
"""


def maze_env(seed=0):
    return Warehouse(maze_layout(seed, 11, 9, n_robots=ROBOTS), n_robots=ROBOTS, max_steps=STEPS,
                     obs=Search.OBS, **FACTORS)


def ready(env, **opts):
    policy = Search(env.spec, seed=0)
    policy.configure(**opts)
    policy.reset()
    policy.see_env(env)
    return policy


class TestSearch(unittest.TestCase):
    def test_k_zero_is_the_prior(self):
        env = maze_env()
        policy, prior = ready(env, k="0"), PPO(env.spec, seed=0)
        prior.reset()
        obs = env.reset(0)
        policy.reset()
        for _ in range(20):
            torch.manual_seed(0)
            got = policy.act(obs)
            torch.manual_seed(0)
            self.assertEqual(got, prior.sample([obs])[0])
            obs = env.step(got)[0]

    def test_act_never_touches_the_real_env(self):
        env = maze_env()
        obs = env.reset(0)
        policy = ready(env, k="2", depth="3", when="always")
        before, rng = env.get_state(), env.rng.getstate()
        policy.act(obs)
        self.assertEqual(env.get_state(), before)
        self.assertEqual(env.rng.getstate(), rng)

    def test_rollouts_do_not_see_the_true_future(self):
        env = maze_env()
        obs = env.reset(0)
        policy = ready(env, k="2", depth="2", when="always")
        with patch.object(mod, "clone", wraps=real_clone) as spy:
            policy.act(obs)
        seeds = [call.kwargs.get("seed") for call in spy.call_args_list]
        self.assertTrue(seeds)
        self.assertTrue(all(isinstance(s, int) for s in seeds), seeds)

    def test_common_random_numbers(self):
        env = maze_env()
        obs = env.reset(0)
        policy = ready(env, k="3", depth="2", when="always")
        with patch.object(mod, "clone", wraps=real_clone) as spy:
            policy.act(obs)
        seeds = [call.kwargs["seed"] for call in spy.call_args_list]
        self.assertEqual(len(set(seeds)), 3)
        self.assertEqual(len({seeds.count(s) for s in set(seeds)}), 1)     # every candidate got each seed

    def test_search_avoids_a_swap_the_prior_would_walk_into(self):
        env = Warehouse(CORRIDOR, n_robots=2, max_steps=40, obs=Search.OBS)
        env.reset(0)
        for robot, pos, heading in ((env.robots[0], (2, 0), RIGHT), (env.robots[1], (3, 0), LEFT)):
            robot.pos, robot.heading, robot.crate = pos, heading, None
        env.robots[0].task = Task("retrieve", (5, 1), (6, 3))
        env.robots[1].task = Task("retrieve", (1, 1), (0, 3))
        env.crate_at.update({(5, 1): 900, (1, 1): 901})
        env.crate_info[900] = env.crate_info[901] = (1, 1)
        env.reserved |= {(5, 1), (1, 1)}
        env.queue, env.humans = [], ()
        grid = env.spec.grid
        policy = ready(env, k="4", depth="4", when="always")
        forward = lambda fleets: [[FORWARD if legal_actions(grid, o)[FORWARD] else TURN_RIGHT
                                   for o in one] for one in fleets]
        with patch.object(policy.prior, "sample", forward), \
                patch.object(policy.prior, "values", lambda fleets: torch.zeros(len(fleets), 2)):
            chosen = policy.act(env._observe_all())
        self.assertNotEqual(chosen, [FORWARD, FORWARD])
        self.assertEqual(env.step(chosen)[3]["blocked"], 0)

    def test_settings_are_validated(self):
        policy = Search(maze_env().spec)
        for bad in ({"when": "x"}, {"k": "-1"}, {"depth": "0"}, {"bogus": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                policy.configure(**bad)

    def test_prior_checkpoint_is_checked(self):
        env = maze_env()
        trained = PPO(env.spec, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            good, other = Path(tmp, "ppo"), Path(tmp, "other")
            save_checkpoint(trained, good, "ppo", ROBOTS)
            save_checkpoint(trained, other, "greedy", ROBOTS)
            policy = Search(env.spec, seed=0)
            policy.configure(prior=str(good))
            for a, b in zip(trained.actor.parameters(), policy.prior.actor.parameters()):
                self.assertTrue(torch.equal(a, b))
            with self.assertRaisesRegex(ValueError, "not 'ppo'"):
                Search(env.spec, seed=0).configure(prior=str(other))


if __name__ == "__main__":
    unittest.main()
