"""The action mask in rl/policies/central_dqn.py must agree with what the env accepts.

The env never exposes which actions are possible: it silently ignores an impossible one and
fines it R_INVALID, reporting only a per-step count. So the policy reconstructs legality from
the map and its observation, and this file pins that reconstruction to the env's own rules. If
env/warehouse.py ever changes when an action counts, these tests fail instead of the agent
quietly going back to wasting most of its moves.
"""
import random
import unittest

from env.layout import maze_layout
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


if __name__ == "__main__":
    unittest.main()
