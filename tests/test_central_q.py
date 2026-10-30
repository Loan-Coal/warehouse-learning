"""rl/policies/central_q.py: joint actions, the legal mask, the TD target, epsilon, the memory cap."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from env.layout import DEFAULT_MAP
from env.warehouse import N_ACTIONS, TURN_LEFT, TURN_RIGHT, Warehouse
from rl.nav import legal_actions
from rl.policies.central_q import MB, CentralQ


def setup(n_robots=2, seed=0):
    env = Warehouse(DEFAULT_MAP, n_robots=n_robots, obs=CentralQ.OBS)
    obs = env.reset(seed)
    return env, CentralQ(env.spec, seed), obs


class TestJointActions(unittest.TestCase):
    def test_encode_decode_round_trip_robot_0_most_significant(self):
        _, policy, _ = setup(n_robots=3)
        self.assertEqual(policy.n_joint, N_ACTIONS ** 3)
        self.assertEqual(policy.encode([1, 0, 0]), N_ACTIONS ** 2)
        self.assertEqual(policy.decode(N_ACTIONS ** 2 + 5), [1, 0, 5])
        for joint in range(policy.n_joint):
            self.assertEqual(policy.encode(policy.decode(joint)), joint)

    def test_mask_is_the_product_of_each_robots_legal_actions(self):
        env, policy, obs = setup()
        for _ in range(15):
            legal = [legal_actions(env.spec.grid, o) for o in obs]
            expected = [j for j in range(policy.n_joint)
                        if all(legal[i][a] for i, a in enumerate(policy.decode(j)))]
            self.assertEqual(policy.legal_joint(obs).tolist(), expected)
            obs, *_ = env.step(policy.act(obs))


class TestLearning(unittest.TestCase):
    def test_td_target_sums_rewards_and_maxes_over_legal_next_actions(self):
        env, policy, obs = setup()
        actions = [TURN_RIGHT, TURN_LEFT]
        next_obs, *_ = env.step(actions)
        legal_next = policy.legal_joint(next_obs)
        illegal_next = sorted(set(range(policy.n_joint)) - set(legal_next.tolist()))
        self.assertTrue(illegal_next, "the test needs an illegal joint action in the next state")
        row = np.zeros(policy.n_joint, dtype=np.float32)
        row[legal_next] = np.arange(len(legal_next), dtype=np.float32)
        row[illegal_next] = 1000.0         # must be ignored by the max
        policy.q[policy.state(next_obs)] = row
        policy.update(obs, actions, [1.0, 2.0], next_obs, done=True)    # done: still bootstrapped
        expected = policy.lr * (3.0 + policy.gamma * (len(legal_next) - 1))
        got = policy.q[policy.state(obs)][policy.encode(actions)]
        self.assertAlmostEqual(float(got), expected, places=4)

    def test_epsilon_decays_linearly_then_holds(self):
        _, policy, _ = setup()
        policy.configure(eps_start="1.0", eps_end="0.05", eps_steps="100")
        for steps, eps in ((0, 1.0), (50, 0.525), (100, 0.05), (1000, 0.05)):
            policy.steps = steps
            self.assertAlmostEqual(policy.epsilon, eps)

    def test_act_never_stores_and_unknown_state_plays_first_legal(self):
        _, policy, obs = setup()
        policy.training = False
        self.assertEqual(policy.encode(policy.act(obs)), policy.legal_joint(obs)[0])
        policy.training = True
        for _ in range(20):
            policy.act(obs)
        self.assertEqual(policy.q, {})

    def test_memory_cap_rejects_new_states_and_counts_them(self):
        env, policy, obs = setup()
        policy.configure(max_mb=str(1.5 * policy.n_joint * 4 / MB))     # room for one row
        for _ in range(2):
            actions = [TURN_RIGHT, TURN_RIGHT]
            next_obs, rewards, done, _ = env.step(actions)
            policy.update(obs, actions, rewards, next_obs, done)
            obs = next_obs
        self.assertEqual(policy.stats()["states"], 1)
        self.assertEqual(policy.stats()["dropped"], 0.5)
        self.assertLessEqual(policy.stats()["table_mb"], policy.max_mb)


class TestRobotCount(unittest.TestCase):
    def test_other_robot_count_is_refused(self):
        _, policy, _ = setup(n_robots=2)
        env3, policy3, obs3 = setup(n_robots=3)
        with self.assertRaisesRegex(ValueError, "built for 2 robots"):
            policy.spec = env3.spec
        with self.assertRaisesRegex(ValueError, "built for 2 robots"):
            policy.act(obs3)
        with tempfile.TemporaryDirectory() as tmp:
            policy.save(Path(tmp))
            with self.assertRaisesRegex(ValueError, "for 2 robots"):
                policy3.load(Path(tmp))


if __name__ == "__main__":
    unittest.main()
