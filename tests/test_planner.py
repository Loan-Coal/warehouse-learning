"""rl/policies/planner.py: the benchmark planner. Robots never block each other by construction."""
import unittest

from env.layout import maze_layout
from env.observation import LEFT, RIGHT
from env.warehouse import DROPOFF, PICKUP, Task, Warehouse
from rl.loop import run_episode
from rl.policies import POLICIES
from rl.policies.planner import Planner

FACTORS = dict(mixed_crates=True, mixed_robots=True, priority=True)
CORRIDOR = """
.......
.SSSSS.
.......
D.....D
"""


def maze_env(seed=0, size=(11, 9), robots=4, humans=0, steps=60):
    return Warehouse(maze_layout(seed, *size, n_robots=robots), n_robots=robots, max_steps=steps,
                     obs=Planner.OBS, humans=humans, **FACTORS)


def planner(env, **opts):
    policy = Planner(env.spec)
    policy.configure(**opts)
    return policy


class TestPlanner(unittest.TestCase):
    def test_no_robot_robot_blocks_without_humans(self):
        for info in ("obs", "state"):
            for seed in range(10):
                with self.subTest(info=info, seed=seed):
                    env = maze_env(seed, (15, 13), robots=6, steps=80)
                    stats = run_episode(env, planner(env, info=info), seed, train=False)
                    self.assertEqual((stats["blocked"], stats["invalid"]), (0, 0))

    def test_obs_level_ignores_the_env(self):
        def record(blind):
            env = maze_env(humans=2)
            policy = planner(env, info="obs")
            if blind:
                policy.see_env = lambda env: None
            seen = []
            run_episode(env, policy, 0, False, lambda obs, actions, *_: seen.append(actions))
            return seen

        self.assertEqual(record(False), record(True))
        env = maze_env(humans=2)
        policy = planner(env, info="state")
        policy.see_env = lambda env: None
        with self.assertRaises(RuntimeError):
            run_episode(env, policy, 0, False)

    def test_slow_robot_never_plans_a_move_or_a_finish_on_an_off_tick(self):
        env = maze_env(robots=4)
        policy = planner(env, info="state")
        obs = env.reset(0)
        policy.reset()
        policy.see_env(env)
        paths, _ = policy.plan_all(obs, set(), {})
        slow = [i for i, o in enumerate(obs) if o.period == 2]
        self.assertTrue(slow)
        for i in slow:
            o = obs[i]
            before = ((o.x, o.y), o.heading)
            for t, step in enumerate(paths.get(i, []), 1):
                if (t - 1 + (0 if o.ready else 1)) % 2 != 0:
                    self.assertEqual((step.cell, step.heading), before)
                    self.assertNotIn(step.action, (PICKUP, DROPOFF))
                before = (step.cell, step.heading)

    def test_head_on_pair_resolves(self):
        env = Warehouse(CORRIDOR, n_robots=2, max_steps=40, obs=Planner.OBS)
        env.reset(0)
        for robot, pos, heading in ((env.robots[0], (1, 0), RIGHT), (env.robots[1], (5, 0), LEFT)):
            robot.pos, robot.heading, robot.crate = pos, heading, None
        env.robots[0].task = Task("retrieve", (5, 1), (6, 3))
        env.robots[1].task = Task("retrieve", (1, 1), (0, 3))
        env.crate_at.update({(5, 1): 900, (1, 1): 901})
        env.crate_info[900] = env.crate_info[901] = (1, 1)
        env.reserved |= {(5, 1), (1, 1)}
        env.queue, env.humans = [], ()
        policy = planner(env, info="state")
        policy.reset()
        policy.see_env(env)
        carried, blocked = set(), 0
        for _ in range(40):
            _, _, _, info = env.step(policy.act(env._observe_all()))
            blocked += info["blocked"]
            carried |= {i for i, r in enumerate(env.robots) if r.crate is not None}
        self.assertEqual(carried, {0, 1})
        self.assertEqual(blocked, 0)

    def test_future_level_leaves_the_real_env_untouched(self):
        env = maze_env(humans=3)
        policy = planner(env, info="future")
        obs = env.reset(0)
        policy.reset()
        policy.see_env(env)
        for _ in range(5):
            before, rng = env.get_state(), env.rng.getstate()
            actions = policy.act(obs)
            self.assertEqual(env.get_state(), before)
            self.assertEqual(env.rng.getstate(), rng)
            obs = env.step(actions)[0]

    def test_beats_greedy_on_hard_seed_0(self):
        def deliveries(name, **opts):
            make = POLICIES[name]
            env = Warehouse(maze_layout(0, 11, 9, n_robots=4), n_robots=4, max_steps=200,
                            obs=make.OBS, humans=2, **FACTORS)
            policy = make(env.spec, 0)
            policy.configure(**opts)
            return sum(run_episode(env, policy, k, False)["deliveries"] for k in range(5))

        self.assertGreaterEqual(deliveries("planner", info="state"), deliveries("greedy"))

    def test_settings_are_validated(self):
        policy = Planner(maze_env().spec)
        for bad in ({"info": "x"}, {"window": "0"}, {"restarts": "-1"}, {"bogus": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                policy.configure(**bad)


if __name__ == "__main__":
    unittest.main()
