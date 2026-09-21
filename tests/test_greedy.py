"""The greedy baseline must actually deliver, alone and with company."""
import unittest

from env.layout import DEFAULT_MAP, random_layout
from env.warehouse import Warehouse
from rl.policies.greedy import GreedyPolicy


def deliveries(layout, n_robots, seed):
    env = Warehouse(layout, n_robots=n_robots)
    obs = env.reset(seed)
    policy = GreedyPolicy(env.grid)
    total, done = 0, False
    while not done:
        obs, _, done, info = env.step([policy.act(o) for o in obs])
        total += info["deliveries"]
    return total


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

    def test_is_reproducible_from_its_seed(self):
        grid = DEFAULT_MAP.strip().splitlines()
        obs = (0, 0, 0, 1, 3, 1) + (0, 0) * 4
        a, b = GreedyPolicy(grid, seed=7), GreedyPolicy(grid, seed=7)
        self.assertEqual([a.act(obs) for _ in range(50)], [b.act(obs) for _ in range(50)])

    def test_without_epsilon_it_follows_the_shortest_path(self):
        policy = GreedyPolicy(DEFAULT_MAP.strip().splitlines(), epsilon=0.0)
        # From (0, 0) the shelf (3, 1) is served from (3, 0): walk right, then interact.
        self.assertEqual(policy.act((0, 0, 0, 1, 3, 1) + (0, 0) * 4), 3)
        self.assertEqual(policy.act((3, 0, 0, 1, 3, 1) + (0, 0) * 4), 5)
        self.assertEqual(policy.act((3, 0, 0, 0, 3, 0) + (0, 0) * 4), 4)  # idle waits


if __name__ == "__main__":
    unittest.main()
