"""The greedy baseline must actually deliver, alone, with company and with every factor on."""
import unittest

from env.layout import DEFAULT_MAP, maze_layout, random_layout
from env.observation import obs_type
from env.warehouse import (BACKWARD, DOWN, FORWARD, LEFT, PICKUP, RIGHT, TURN_LEFT, TURN_RIGHT, UP, EnvSpec,
                           Warehouse)
from rl.loop import run_episode
from rl.policies import POLICIES
from rl.policies.greedy import GreedyRule

SPEC = EnvSpec(Warehouse().grid, n_robots=1, obs_features=GreedyRule.OBS)
Obs = obs_type(GreedyRule.OBS)


def obs(x, y, heading, carrying, has_task, tx, ty):
    """A hand-made observation with every ray empty."""
    return Obs(x, y, heading, carrying, has_task, tx, ty, *(0, 0) * 4)


def deliveries(layout, n_robots, seed, max_steps=100, **factors):
    env = Warehouse(layout, n_robots=n_robots, obs=GreedyRule.OBS, max_steps=max_steps, **factors)
    return run_episode(env, POLICIES["greedy"](env.spec, 0), seed, train=False)["deliveries"]


class TestGreedy(unittest.TestCase):
    def test_single_robot_delivers(self):
        for seed in range(5):
            self.assertGreaterEqual(deliveries(DEFAULT_MAP, 1, seed), 1, f"seed {seed}")

    def test_two_robots_deliver(self):
        for seed in range(5):
            self.assertGreaterEqual(deliveries(DEFAULT_MAP, 2, seed), 2, f"seed {seed}")

    def test_random_layouts_and_mazes_deliver(self):
        for seed in range(5):
            self.assertGreaterEqual(deliveries(random_layout(seed), 2, seed), 1, f"seed {seed}")
            # Turning costs steps and maze corridors are one wide: mazes get 200 steps, as in the scenarios.
            self.assertGreaterEqual(deliveries(maze_layout(seed), 2, seed, max_steps=200), 1, f"seed {seed}")

    def test_delivers_with_every_factor_on(self):
        total = sum(deliveries(DEFAULT_MAP, 3, seed, mixed_crates=True, mixed_robots=True, priority=True,
                               humans=2) for seed in range(5))
        self.assertGreaterEqual(total, 5)

    def test_is_reproducible_from_its_seed(self):
        a, b = GreedyRule(SPEC, seed=7), GreedyRule(SPEC, seed=7)
        o = obs(0, 0, UP, 0, 1, 3, 1)
        self.assertEqual([a.act_one(o) for _ in range(50)], [b.act_one(o) for _ in range(50)])

    def test_without_epsilon_it_turns_drives_and_picks_up(self):
        policy = GreedyRule(SPEC, epsilon=0.0)
        # The shelf (3, 1) is served from (3, 0), facing down.
        self.assertEqual(policy.act_one(obs(0, 0, RIGHT, 0, 1, 3, 1)), FORWARD)
        self.assertEqual(policy.act_one(obs(0, 0, LEFT, 0, 1, 3, 1)), BACKWARD)
        self.assertEqual(policy.act_one(obs(0, 0, UP, 0, 1, 3, 1)), TURN_RIGHT)
        self.assertEqual(policy.act_one(obs(3, 0, RIGHT, 0, 1, 3, 1)), TURN_RIGHT)
        self.assertEqual(policy.act_one(obs(3, 0, LEFT, 0, 1, 3, 1)), TURN_LEFT)
        self.assertEqual(policy.act_one(obs(3, 0, DOWN, 0, 1, 3, 1)), PICKUP)
        self.assertEqual(policy.act_one(obs(3, 0, DOWN, 0, 0, 3, 0)), TURN_RIGHT)  # idle turns in place


if __name__ == "__main__":
    unittest.main()
