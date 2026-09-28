"""The greedy baseline must actually deliver, alone and with company."""
import unittest

from env.layout import DEFAULT_MAP, maze_layout, random_layout
from env.warehouse import INTERACT, RIGHT, WAIT, EnvSpec, Obs, Warehouse
from rl.loop import run_episode
from rl.policies import POLICIES
from rl.policies.greedy import GreedyRule

SPEC = EnvSpec(Warehouse().grid, n_robots=1)


def obs(*fields):
    """A hand-made observation: the first six fields, all rays empty."""
    return Obs(*fields, *(0, 0) * 4)


def deliveries(layout, n_robots, seed):
    env = Warehouse(layout, n_robots=n_robots)
    return run_episode(env, POLICIES["greedy"](env.spec, 0), seed, train=False)["deliveries"]


class TestGreedy(unittest.TestCase):
    def test_single_robot_delivers(self):
        for seed in range(5):
            self.assertGreaterEqual(deliveries(DEFAULT_MAP, 1, seed), 1, f"seed {seed}")

    def test_two_robots_deliver(self):
        for seed in range(5):
            self.assertGreaterEqual(deliveries(DEFAULT_MAP, 2, seed), 2, f"seed {seed}")

    def test_random_layouts_deliver(self):
        for seed in range(5):
            self.assertGreaterEqual(deliveries(random_layout(seed), 2, seed), 1, f"seed {seed}")

    def test_mazes_deliver(self):
        for seed in range(5):
            self.assertGreaterEqual(deliveries(maze_layout(seed), 2, seed), 1, f"seed {seed}")

    def test_is_reproducible_from_its_seed(self):
        a, b = GreedyRule(SPEC, seed=7), GreedyRule(SPEC, seed=7)
        o = obs(0, 0, 0, 1, 3, 1)
        self.assertEqual([a.act_one(o) for _ in range(50)], [b.act_one(o) for _ in range(50)])

    def test_without_epsilon_it_follows_the_shortest_path(self):
        policy = GreedyRule(SPEC, epsilon=0.0)
        # From (0, 0) the shelf (3, 1) is served from (3, 0): walk right, then interact.
        self.assertEqual(policy.act_one(obs(0, 0, 0, 1, 3, 1)), RIGHT)
        self.assertEqual(policy.act_one(obs(3, 0, 0, 1, 3, 1)), INTERACT)
        self.assertEqual(policy.act_one(obs(3, 0, 0, 0, 3, 0)), WAIT)  # idle waits


if __name__ == "__main__":
    unittest.main()
