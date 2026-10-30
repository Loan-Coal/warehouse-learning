"""rl/model.py: a clone must step exactly like a deepcopy and never touch the original.

clone() names Warehouse's attributes, so EXPECTED_ATTRS pins them: a new mutable field in the env
fails here before any policy silently shares it between a copy and the real episode.
"""
import copy
import random
import unittest

from env.layout import maze_layout
from env.warehouse import Warehouse
from rl.model import clone

ROBOTS = 6
EXPECTED_ATTRS = {"obs_features", "n_robots", "max_steps", "view_range", "alpha", "task_prob",
                  "mixed_crates", "mixed_robots", "priority", "n_humans", "grid", "width",
                  "height", "shelves", "docks", "rng", "t", "totals", "robots", "humans",
                  "crate_at", "crate_info", "next_crate_id", "queue", "reserved"}


def big_env():
    env = Warehouse(maze_layout(0, 21, 21, n_robots=ROBOTS), n_robots=ROBOTS, max_steps=200,
                    mixed_crates=True, mixed_robots=True, priority=True, humans=3)
    env.reset(0)
    return env


def random_actions(rng, n):
    return [rng.randrange(6) for _ in range(n)]


def snapshot(env):
    return (env.get_state(), env.totals, env.rng.getstate(), list(env.queue), set(env.reserved),
            dict(env.crate_at), dict(env.crate_info), env.next_crate_id, env.humans)


class TestClone(unittest.TestCase):
    def test_every_env_attribute_is_known(self):
        self.assertEqual(set(vars(big_env())), EXPECTED_ATTRS)

    def test_clone_matches_deepcopy_step_for_step(self):
        env = big_env()
        a, b = clone(env), copy.deepcopy(env)
        rng = random.Random(1)
        for _ in range(100):
            actions = random_actions(rng, ROBOTS)
            a.step(actions)
            b.step(actions)
            self.assertEqual(snapshot(a), snapshot(b))

    def test_clone_never_touches_the_original(self):
        env = big_env()
        before = snapshot(env)
        model, rng = clone(env), random.Random(2)
        for _ in range(50):
            model.step(random_actions(rng, ROBOTS))
        self.assertEqual(snapshot(env), before)
        self.assertEqual(env.t, 0)

    def test_reseeded_clone_diverges_from_the_true_stream(self):
        env = big_env()
        true, reseeded = clone(env), clone(env, seed=1)
        rng, differ = random.Random(3), False
        for _ in range(100):
            actions = random_actions(rng, ROBOTS)
            true.step(actions)
            reseeded.step(actions)
            differ = differ or true.humans != reseeded.humans
        self.assertTrue(differ)


if __name__ == "__main__":
    unittest.main()
