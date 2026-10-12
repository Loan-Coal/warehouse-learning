"""rl/features.py: the env properties that decide what is worth showing a network.

Five feature groups were dropped from the input after checking them rather than assuming. These
are the tests that stop them coming back, and what fails first if the env stops behaving this
way. They are about the env, not about any one algorithm.
"""
import random
import unittest

from env.layout import maze_layout
from env.observation import NOTHING, RIGHT
from env.warehouse import N_ACTIONS, Task, Warehouse
from rl.features import PAIR_WIDTH, Features

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


CORRIDOR = """
.......
.SSSSS.
.......
D.....D
"""


class TestPairFeatures(unittest.TestCase):
    def fleet(self, *robots):
        """Observations for hand-placed robots: (x, y, task or None) on the corridor map."""
        env = Warehouse(CORRIDOR, n_robots=len(robots), max_steps=STEPS, obs=Features.OBS)
        env.reset(0)
        for robot, (x, y, task) in zip(env.robots, robots):
            robot.pos, robot.heading, robot.task, robot.crate = (x, y), RIGHT, task, None
        inputs = Features(Features.OBS)
        inputs.use(env.spec.grid)
        return inputs, env._observe_all()

    def test_pair_tensor_shape_and_zero_diagonal(self):
        env = maze_env(robots=6)
        inputs = Features(Features.OBS)
        inputs.use(env.spec.grid)
        pairs = inputs.pairs([env.reset(0)])
        self.assertEqual(tuple(pairs.shape), (1, 6, 6, PAIR_WIDTH))
        for i in range(6):
            self.assertEqual(pairs[0, i, i].abs().sum().item(), 0)

    def test_on_path_and_distance_on_a_corridor(self):
        job = Task("retrieve", (3, 1), (0, 3))
        inputs, obs = self.fleet((0, 0, job), (2, 0, None))
        row = inputs.pair_row(obs[0], obs[1], inputs.best_next(obs[0]), inputs.best_next(obs[1]))
        self.assertAlmostEqual(row[0], 2 / (7 + 4))
        self.assertEqual(row[1], 1.0)
        inputs, obs = self.fleet((0, 0, job), (2, 2, None))
        row = inputs.pair_row(obs[0], obs[1], inputs.best_next(obs[0]), inputs.best_next(obs[1]))
        self.assertEqual(row[1], 0.0)

    def test_same_next_and_toward(self):
        job = Task("retrieve", (3, 1), (0, 3))
        inputs, obs = self.fleet((2, 0, job), (4, 0, job))
        nxt = [inputs.best_next(o) for o in obs]
        self.assertEqual(nxt, [(3, 0), (3, 0)])
        self.assertEqual(inputs.pair_row(obs[0], obs[1], *nxt)[2], 1.0)
        inputs, obs = self.fleet((2, 0, job), (3, 0, None))
        nxt = [inputs.best_next(o) for o in obs]
        row = inputs.pair_row(obs[0], obs[1], *nxt)
        self.assertEqual((row[2], row[4]), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
