"""rl/features.py: the env properties that decide what is worth showing a network.

Five feature groups were dropped from the input after checking them rather than assuming. These
are the tests that stop them coming back, and what fails first if the env stops behaving this
way. They are about the env, not about any one algorithm.
"""
import random
import unittest

from env.layout import maze_layout
from env.observation import NOTHING
from env.warehouse import N_ACTIONS, Warehouse
from rl.features import Features

ROBOTS, STEPS = 4, 60
FACTORS = dict(mixed_crates=True, mixed_robots=True, priority=True, humans=2)


def maze_env(seed=0, size=(11, 9), robots=ROBOTS):
    return Warehouse(maze_layout(seed, *size, n_robots=robots), n_robots=robots, max_steps=STEPS,
                     obs=Features.OBS, **FACTORS)


class TestDroppedFeatures(unittest.TestCase):
    def test_a_ray_reports_nothing_exactly_when_its_distance_is_zero(self):
        """So a "nothing in range" flag duplicates the NOTHING slot of the ray's own one-hot."""
        env, rng = maze_env(), random.Random(0)
        checked = 0
        for seed in range(4):
            obs = env.reset(seed)
            for _ in range(STEPS):
                for o in obs:
                    for dist, kind in ((o.front_d, o.front_t), (o.right_d, o.right_t),
                                       (o.back_d, o.back_t), (o.left_d, o.left_t)):
                        self.assertEqual(dist == 0, kind == NOTHING)
                        checked += 1
                obs, _, done, _ = env.step([rng.randrange(N_ACTIONS) for _ in range(ROBOTS)])
                if done:
                    break
        self.assertGreater(checked, 3000)


class TestScaleIndependence(unittest.TestCase):
    def test_a_row_is_the_same_width_whatever_the_fleet_or_the_map(self):
        """The headline claim: a checkpoint trained small loads straight into something bigger."""
        widths = set()
        for robots, size in ((2, (11, 9)), (4, (11, 9)), (6, (21, 21)), (10, (15, 13))):
            env = maze_env(size=size, robots=robots)
            inputs = Features(env.spec.obs_features)
            inputs.use(env.spec.grid)
            obs = env.reset(0)
            widths.add(inputs.width)
            self.assertEqual({len(inputs.row(o)) for o in obs}, {inputs.width})
            self.assertEqual(tuple(inputs.batch([obs]).shape), (1, robots, inputs.width))
        self.assertEqual(len(widths), 1, f"the row changed width with the scenario: {widths}")


if __name__ == "__main__":
    unittest.main()
