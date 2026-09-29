"""rl/policies/central_dqn.py: the action mask, the features it does not have, and learning.

The mask comes first and is the reason this file exists.

The env never exposes which actions are possible: it silently ignores an impossible one and
fines it R_INVALID, reporting only a per-step count. So the policy reconstructs legality from
the map and its observation, and this file pins that reconstruction to the env's own rules. If
env/warehouse.py ever changes when an action counts, these tests fail instead of the agent
quietly going back to wasting most of its moves.
"""
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import rl.policies.central_dqn as mod
from env.layout import is_walkable, maze_layout
from env.observation import HEADINGS
from env.warehouse import N_ACTIONS, TURN_LEFT, TURN_RIGHT, Warehouse
from rl.nav import legal_actions
from rl.policies.central_dqn import CentralDQN

ROBOTS, STEPS = 4, 60
FACTORS = dict(mixed_crates=True, mixed_robots=True, priority=True, humans=2)


def maze_env(seed=0, size=(11, 9)):
    return Warehouse(maze_layout(seed, *size, n_robots=ROBOTS), n_robots=ROBOTS, max_steps=STEPS,
                     obs=CentralDQN.OBS, **FACTORS)


class TestLegalActions(unittest.TestCase):
    def test_only_legal_actions_are_ever_invalid(self):
        """Playing only actions the mask allows must never earn a single invalid penalty."""
        for seed in range(4):
            with self.subTest(seed=seed):
                env = maze_env(seed)
                obs = env.reset(seed)
                rng = random.Random(seed)
                for _ in range(STEPS):
                    actions = [rng.choice([a for a in range(N_ACTIONS) if legal_actions(env.grid, o)[a]])
                               for o in obs]
                    obs, _, done, info = env.step(actions)
                    self.assertEqual(info["invalid"], 0, "the mask allowed an action the env rejected")
                    if done:
                        break

    def test_some_action_is_always_legal(self):
        """Turning is never invalid, so the mask can never leave a robot with nothing to do."""
        env = maze_env()
        obs = env.reset(0)
        rng = random.Random(0)
        for _ in range(STEPS):
            for o in obs:
                legal = legal_actions(env.grid, o)
                self.assertTrue(any(legal))
                self.assertTrue(legal[TURN_RIGHT] and legal[TURN_LEFT])
            obs, _, done, _ = env.step([rng.randrange(N_ACTIONS) for _ in obs])
            if done:
                break

    def test_what_the_mask_forbids_the_env_rejects(self):
        """The other direction, so a mask that just forbade everything would not pass.

        One ready robot plays a forbidden action; the rest turn in place, which is always
        accepted. The env must then report exactly that one invalid action.
        """
        env = maze_env()
        obs = env.reset(0)
        checked = 0
        for _ in range(STEPS):
            actions, victim = [TURN_RIGHT] * len(obs), None
            for i, o in enumerate(obs):
                illegal = [a for a in range(N_ACTIONS) if not legal_actions(env.grid, o)[a]]
                if victim is None and illegal and env.ready(i):
                    victim, actions[i] = i, illegal[0]
            obs, _, done, info = env.step(actions)
            if victim is not None:
                self.assertEqual(info["invalid"], 1, "the env accepted an action the mask forbade")
                checked += 1
            if done:
                break
        self.assertGreater(checked, 0, "the episode never offered an illegal action to test")


class TestMaskedPolicy(unittest.TestCase):
    def test_policy_never_plays_an_invalid_action(self):
        """End to end: the untrained net, its exploration and its teacher all stay inside the mask."""
        for training in (True, False):
            with self.subTest(training=training):
                env = maze_env(1)
                policy = CentralDQN(env.spec, seed=1)
                policy.training = training
                obs = env.reset(1)
                policy.reset()
                for _ in range(STEPS):
                    actions = policy.act(obs)
                    next_obs, rewards, done, info = env.step(actions)
                    self.assertEqual(info["invalid"], 0)
                    if training:
                        policy.update(obs, actions, rewards, next_obs, done)
                    obs = next_obs
                    if done:
                        break



class TestDroppedFeatures(unittest.TestCase):
    """The env properties that make five feature groups pointless to feed the network.

    robot_features() used to carry all of them. They are gone; these tests are what stops them
    coming back, and what fails first if the env ever stops behaving this way.
    """

    def test_a_ray_reports_nothing_exactly_when_its_distance_is_zero(self):
        """So a "nothing in range" flag duplicates the NOTHING slot of the ray's own one-hot."""
        from env.observation import NOTHING
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

    def test_a_robot_is_only_ever_given_a_task_it_can_carry(self):
        """So crate size and robot capacity cannot change the value of any action it can pick."""
        env, rng = maze_env(), random.Random(1)
        checked = 0
        for seed in range(4):
            env.reset(seed)
            for _ in range(STEPS):
                for robot in env.robots:
                    if robot.task is not None:
                        self.assertLessEqual(robot.task.weight, robot.kind.max_weight)
                        self.assertLessEqual(robot.task.volume, robot.kind.max_volume)
                        checked += 1
                _, _, done, _ = env.step([rng.randrange(N_ACTIONS) for _ in range(ROBOTS)])
                if done:
                    break
        self.assertGreater(checked, 100)

    def test_the_input_width_does_not_depend_on_the_fleet_or_the_map(self):
        """The headline claim: a checkpoint trained small loads straight into something bigger."""
        widths = set()
        for robots, size in ((2, (11, 9)), (4, (11, 9)), (6, (21, 21)), (10, (15, 13))):
            env = Warehouse(maze_layout(0, *size, n_robots=robots), n_robots=robots,
                            max_steps=STEPS, obs=CentralDQN.OBS)
            policy = CentralDQN(env.spec, seed=0)
            widths.add((policy.robot_dim, sum(t.numel() for t in policy.net.parameters())))
            self.assertEqual(len(policy.act(env.reset(0))), robots)
        self.assertEqual(len(widths), 1, f"the net changed shape with the scenario: {widths}")


class TestLearning(unittest.TestCase):
    def shaping(self, policy, before, after):
        """What the net is paid for the step `before` -> `after`, on top of the env reward."""
        return policy.learning_rewards([before], [0.0], [after])[0]

    def busy_robot(self):
        """A policy plus one observation of a robot that has a task and can reach it."""
        env = maze_env()
        policy = CentralDQN(env.spec, seed=0)
        policy.training = True
        obs = env.reset(0)
        policy.reset()
        rng = random.Random(2)
        for _ in range(STEPS):
            for o in obs:
                if o.has_task and policy.potential(o) != 0.0:
                    return policy, o
            obs, _, done, _ = env.step([rng.randrange(N_ACTIONS) for _ in range(ROBOTS)])
            if done:
                obs = env.reset(0)
        raise AssertionError("no robot ever had a reachable task")

    def test_standing_still_earns_exactly_nothing(self):
        """The measured failure of Ng, Harada and Russell's form, pinned as a test.

        Their GAMMA * Phi(s') - Phi(s) is policy-invariant, but Phi is negative here, so it pays
        a motionless robot (GAMMA - 1) * Phi per step -- positive, and at any real distance
        bigger than the step cost, which buys loitering. The plain difference pays it zero.
        """
        policy, o = self.busy_robot()
        self.assertEqual(self.shaping(policy, o, o), 0.0)
        textbook = mod.SHAPE * (mod.GAMMA * policy.potential(o) - policy.potential(o))
        self.assertGreater(textbook, 0.0, "the form we rejected pays a robot for doing nothing")

    def test_a_round_trip_earns_exactly_nothing(self):
        """Phi is a function of the state, so any closed walk sums to zero. No loop farms it."""
        policy, here = self.busy_robot()
        field = policy.spec.grid
        step = next((d for d in HEADINGS
                     if is_walkable(field, here.x + d[0], here.y + d[1])
                     and policy.potential(here._replace(x=here.x + d[0], y=here.y + d[1]))
                     != policy.potential(here)), None)
        self.assertIsNotNone(step, "nowhere to walk that changes the potential")
        there = here._replace(x=here.x + step[0], y=here.y + step[1])
        out, back = self.shaping(policy, here, there), self.shaping(policy, there, here)
        self.assertNotEqual(out, 0.0, "the two cells should differ, or this proves nothing")
        self.assertAlmostEqual(out + back, 0.0, places=12)

    def test_shaping_is_suppressed_when_a_delivery_swaps_the_target(self):
        """Phi jumps discontinuously then, and the jump is not progress the robot made."""
        policy, o = self.busy_robot()
        swapped = o._replace(carrying=1 - o.carrying, target_x=o.target_x + 2)
        self.assertTrue(policy.target_changed(o, swapped))
        self.assertEqual(self.shaping(policy, o, swapped), 0.0)

    def test_shaping_off_leaves_the_env_reward_alone(self):
        policy, o = self.busy_robot()
        policy.configure(shaping="0")
        self.assertEqual(policy.learning_rewards([o], [3.0], [o._replace(x=o.x + 1)]),
                         [mod.REWARD_SCALE * 3.0])

    def test_a_real_gradient_step_runs_and_stays_finite(self):
        """The plumbing test the mask tests skip: 60 steps never fills a 64-deep batch."""
        env = maze_env()
        policy = CentralDQN(env.spec, seed=0)
        policy.training = True
        small, mod.BATCH, mod.UPDATE_EVERY = (mod.BATCH, mod.UPDATE_EVERY), 8, 1
        try:
            obs = env.reset(0)
            policy.reset()
            for _ in range(STEPS):
                actions = policy.act(obs)
                next_obs, rewards, done, _ = env.step(actions)
                policy.update(obs, actions, rewards, next_obs, done)
                obs = next_obs
                if done:
                    break
            self.assertGreater(policy.updates, 0, "no gradient step fired")
            self.assertTrue(math.isfinite(policy.loss_ema), f"loss went to {policy.loss_ema}")
            self.assertTrue(all(t.isfinite().all() for t in policy.net.parameters()))
        finally:
            mod.BATCH, mod.UPDATE_EVERY = small

    def test_a_checkpoint_restores_the_feature_width_it_was_trained_with(self):
        """--policy-opt bfs=0 makes a narrower net; loading must not silently use today's width."""
        env = maze_env()
        narrow = CentralDQN(env.spec, seed=0)
        narrow.configure(bfs="0")
        self.assertLess(narrow.robot_dim, CentralDQN(env.spec, seed=0).robot_dim)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            narrow.save(folder)
            self.assertEqual(json.loads((folder / "state.json").read_text(encoding="utf-8"))["bfs"],
                             False)
            wide = CentralDQN(env.spec, seed=0)          # defaults to bfs on
            wide.load(folder)
            self.assertEqual((wide.robot_dim, wide.use_bfs), (narrow.robot_dim, False))
            self.assertEqual(wide.act(env.reset(0)), narrow.act(env.reset(0)))

    def test_an_unknown_policy_opt_is_refused_by_name(self):
        policy = CentralDQN(maze_env().spec, seed=0)
        with self.assertRaisesRegex(ValueError, "unknown setting nope"):
            policy.configure(nope="1")
        with self.assertRaisesRegex(ValueError, "bfs wants 0 or 1"):
            policy.configure(bfs="maybe")

if __name__ == "__main__":
    unittest.main()
