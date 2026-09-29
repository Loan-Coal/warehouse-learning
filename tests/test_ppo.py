"""rl/policies/ppo.py: the mask, the truncation bootstrap, and one real update.

Two of these cover mistakes that produce no error and no obviously wrong number, and so would
otherwise only show up as "PPO does not work on this problem":

  * zeroing the bootstrap at an episode boundary, which teaches the policy that the world ends
    every max_steps and silently caps what any value estimate can be worth;
  * carrying the GAE recursion ACROSS a boundary, which credits one episode's actions with the
    next episode's rewards on a different map.

The action mask is checked against the env the same way tests/test_central_dqn.py checks the
DQN's, because the guarantee has to hold for the sampling path and the argmax path alike.
"""
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import torch

import rl.policies.ppo as mod
from env.layout import maze_layout
from env.warehouse import Warehouse
from rl.policies.ppo import PPO, pooled
from rl.vecenv import VecWarehouse

ROBOTS, STEPS = 4, 40
FACTORS = dict(mixed_crates=True, mixed_robots=True, priority=True, humans=2)


def maze_env(seed=0, size=(11, 9), steps=STEPS):
    return Warehouse(maze_layout(seed, *size, n_robots=ROBOTS), n_robots=ROBOTS, max_steps=steps,
                     obs=PPO.OBS, **FACTORS)


def vec_of(n_envs, steps=STEPS):
    envs = [maze_env(steps=steps) for _ in range(n_envs)]
    return VecWarehouse(envs, lambda n: (n, maze_layout(n // n_envs, 11, 9, n_robots=ROBOTS)))


def play(policy, vec, steps, train=True):
    """Drive the policy through the vec env the way rl/vecloop.py does."""
    policy.training = train
    obs = vec.reset()
    policy.spec = vec.envs[0].spec
    policy.reset()
    for _ in range(steps):
        actions = policy.act_vec(obs)
        nxt, rewards, dones, ended, _ = vec.step(actions)
        if train:
            policy.update_vec(obs, actions, rewards,
                              [e if e is not None else o for e, o in zip(ended, nxt)], dones)
        obs = nxt
        if any(dones):
            policy.spec = vec.envs[0].spec
            policy.reset()


class TestMask(unittest.TestCase):
    def test_it_never_plays_an_action_the_env_would_refuse(self):
        for train in (True, False):
            for greedy in ("0", "1"):
                with self.subTest(train=train, eval_greedy=greedy):
                    env = maze_env()
                    policy = PPO(env.spec, seed=1)
                    policy.configure(eval_greedy=greedy)
                    policy.training = train
                    obs = env.reset(1)
                    policy.reset()
                    for _ in range(STEPS):
                        actions = policy.act(obs)
                        self.assertTrue(all(type(a) is int and 0 <= a < 6 for a in actions))
                        obs, rewards, done, info = env.step(actions)
                        self.assertEqual(info["invalid"], 0)
                        if train:
                            policy.update(obs, actions, rewards, obs, done)
                        if done:
                            break

    def test_a_masked_action_gets_no_probability_and_entropy_stays_finite(self):
        """NEG is finite on purpose: -inf would make 0 * -inf a NaN inside the entropy."""
        env = maze_env()
        policy = PPO(env.spec, seed=1)
        obs = env.reset(1)
        policy.reset()
        mask = policy.masks([obs])
        with torch.no_grad():
            logits = policy.actor(pooled(policy.inputs.batch([obs]))).masked_fill(~mask, mod.NEG)
            dist = torch.distributions.Categorical(logits=logits)
        self.assertTrue(torch.all(dist.probs[~mask] == 0.0), "an illegal action kept probability")
        self.assertTrue(torch.isfinite(dist.entropy()).all())
        self.assertTrue(torch.allclose(dist.probs.sum(-1), torch.ones(1, ROBOTS)))
        self.assertFalse(mask.all(), "this maze should offer some illegal action, or it proves nothing")


class TestTruncationBootstrap(unittest.TestCase):
    """GAE over a rollout that spans an episode boundary. `done` cuts the recursion, not the bootstrap."""

    def gae(self, reward, value, after, done):
        """_gae on one env with one robot, so the arithmetic is checkable by hand."""
        def column(values):
            return torch.tensor(values, dtype=torch.float32).reshape(len(values), 1, 1)

        return PPO._gae(column(reward), column(value), column(after),
                        torch.tensor(done, dtype=torch.float32).reshape(len(done), 1))

    def test_the_value_past_a_time_limit_is_used_not_zeroed(self):
        """The env never terminates, so the return beyond the limit is real."""
        # One step, which is the last of an episode. delta = r + GAMMA * after - value.
        got = self.gae([1.0], [0.0], [5.0], [1.0])
        self.assertAlmostEqual(got.item(), 1.0 + mod.GAMMA * 5.0, places=5)
        # If `done` had zeroed the bootstrap it would be 1.0, which is the bug this pins.
        self.assertNotAlmostEqual(got.item(), 1.0, places=3)

    def test_the_advantage_does_not_carry_across_a_boundary(self):
        """The state after the boundary is an unrelated episode, usually on another map."""
        reward, value, after = [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]
        crossed = self.gae(reward, value, after, [1.0, 0.0])[0].item()
        joined = self.gae(reward, value, after, [0.0, 0.0])[0].item()
        self.assertAlmostEqual(crossed, 0.0, places=6, msg="step 1 inherited step 2's advantage")
        self.assertGreater(joined, 0.0, "without the boundary it should carry, or this proves nothing")

    def test_it_matches_a_longhand_gae_when_nothing_ends(self):
        reward, value, after = [1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [0.5, 0.5, 9.0]
        got = self.gae(reward, value, after, [0.0, 0.0, 0.0]).reshape(-1).tolist()
        want, running = [0.0] * 3, 0.0
        for t in reversed(range(3)):
            delta = reward[t] + mod.GAMMA * after[t] - value[t]
            running = delta + mod.GAMMA * mod.LAMBDA * running
            want[t] = running
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=5)

    def test_a_rollout_spanning_a_boundary_learns_without_complaint(self):
        """End to end: the rollout is longer than the episode, so a boundary lands inside it."""
        rollout, mod.ROLLOUT = mod.ROLLOUT, 24
        try:
            vec = vec_of(2, steps=10)       # boundaries at 10 and 20, inside a 24-step rollout
            policy = PPO(vec.envs[0].spec, seed=0)
            play(policy, vec, 24)
            self.assertEqual(policy.updates, 1, "the rollout should have filled exactly once")
            self.assertTrue(math.isfinite(policy.log["loss"]))
        finally:
            mod.ROLLOUT = rollout


class TestUpdate(unittest.TestCase):
    def test_a_real_update_runs_and_leaves_everything_finite(self):
        rollout, mod.ROLLOUT = mod.ROLLOUT, 8
        try:
            vec = vec_of(4)
            policy = PPO(vec.envs[0].spec, seed=0)
            before = [p.clone() for p in policy.actor.parameters()]
            play(policy, vec, 32)
            self.assertGreaterEqual(policy.updates, 4)
            self.assertEqual(policy.steps, 32 * 4)
            for name, value in policy.log.items():
                self.assertTrue(math.isfinite(value), f"{name} went to {value}")
            self.assertTrue(all(p.isfinite().all() for p in policy.actor.parameters()))
            self.assertTrue(all(p.isfinite().all() for p in policy.critic.parameters()))
            moved = any(not torch.equal(a, b) for a, b in zip(before, policy.actor.parameters()))
            self.assertTrue(moved, "the actor did not change: nothing was learned")
            self.assertEqual(policy.rollout, [], "the rollout should be cleared after an update")
        finally:
            mod.ROLLOUT = rollout

    def test_the_policy_loss_ignores_a_discarded_action_but_the_value_loss_does_not(self):
        """A slow robot acting on its off step: the env threw the action away.

        There is nothing for the POLICY to learn from an action that never happened, but time
        still passed and a reward still arrived, so the critic should still predict that return.
        """
        env = maze_env()
        policy = PPO(env.spec, seed=0)
        obs = env.reset(0)
        policy.reset()
        x = policy.inputs.batch([obs])
        mask = policy.masks([obs])
        acted = torch.zeros(1, ROBOTS, dtype=torch.long)
        for row in range(ROBOTS):
            acted[0, row] = next(a for a in range(6) if mask[0, row, a])
        args = [x, mask, acted, torch.zeros(1, ROBOTS), torch.ones(1, ROBOTS),
                torch.full((1, ROBOTS), 3.0), torch.zeros(1, ROBOTS)]     # ready all zero
        policy._step([t.clone() for t in args])
        self.assertTrue(math.isfinite(policy.log["loss"]))
        # With no live row the policy term and the entropy are both zero by masked_mean, so the
        # whole gradient came from the value loss -- which is the point.
        self.assertEqual(policy.log["entropy"], 0.0)
        self.assertEqual(policy.log["clipfrac"], 0.0)

    def test_an_unknown_policy_opt_is_refused_by_name(self):
        policy = PPO(maze_env().spec, seed=0)
        with self.assertRaisesRegex(ValueError, "unknown setting nope"):
            policy.configure(nope="1")
        with self.assertRaisesRegex(ValueError, "shaping wants 0 or 1"):
            policy.configure(shaping="perhaps")
        with self.assertRaisesRegex(ValueError, "bc wants a number"):
            policy.configure(bc="lots")

    def test_shaping_off_leaves_the_env_reward_alone(self):
        env = maze_env()
        policy = PPO(env.spec, seed=0)
        obs = env.reset(0)
        policy.reset()
        policy.configure(shaping="0")
        self.assertEqual(policy.learning_rewards(obs, [2.0] * ROBOTS, obs),
                         [mod.REWARD_SCALE * 2.0] * ROBOTS)
        policy.configure(shaping="1")
        self.assertEqual(policy.learning_rewards(obs, [2.0] * ROBOTS, obs),
                         [mod.REWARD_SCALE * 2.0] * ROBOTS, "same state, so no shaping either way")


class TestStoredAndRecomputed(unittest.TestCase):
    """Before the first gradient step the policy has not moved, so the ratio must be exactly 1.

    This is the check that catches a whole family of silent mistakes at once: a mask built from
    the wrong map at learn time, features rebuilt in a different order, the stored action not
    being the one the log-probability was taken for. Any of those leaves PPO training on a
    corrupted objective while every number still looks plausible -- the ratio simply stops being
    1 at the start, clipping fires immediately, and learning quietly degrades.
    """

    def test_the_ratio_is_one_before_the_first_update(self):
        rollout, mod.ROLLOUT = mod.ROLLOUT, 1_000_000       # never fires on its own
        try:
            vec = vec_of(3)
            policy = PPO(vec.envs[0].spec, seed=0)
            play(policy, vec, STEPS + 5)                    # spans an episode boundary and a map change
            self.assertGreater(len(policy.rollout), STEPS)

            x, mask, acted, logp = [torch.stack(p) for p in list(zip(*policy.rollout))[:4]]
            with torch.no_grad():
                logits = policy.actor(pooled(x.flatten(0, 1))).masked_fill(~mask.flatten(0, 1),
                                                                          mod.NEG)
                again = torch.distributions.Categorical(logits=logits).log_prob(acted.flatten(0, 1))
            ratio = (again - logp.flatten(0, 1)).exp()
            self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5),
                            f"ratio drifted from 1: min {ratio.min():.6f} max {ratio.max():.6f}")
        finally:
            mod.ROLLOUT = rollout

    def test_a_stored_action_is_always_one_the_mask_allowed(self):
        rollout, mod.ROLLOUT = mod.ROLLOUT, 1_000_000
        try:
            vec = vec_of(3)
            policy = PPO(vec.envs[0].spec, seed=0)
            play(policy, vec, STEPS + 5)
            for step in policy.rollout:
                mask, acted = step[1], step[2]
                chosen = torch.gather(mask, -1, acted.unsqueeze(-1)).squeeze(-1)
                self.assertTrue(chosen.all(), "an illegal action was stored to learn from")
        finally:
            mod.ROLLOUT = rollout


class TestTransfer(unittest.TestCase):
    def test_the_networks_do_not_depend_on_the_fleet_or_the_map(self):
        shapes = set()
        for robots, size in ((2, (11, 9)), (4, (11, 9)), (6, (21, 21)), (10, (15, 13))):
            env = Warehouse(maze_layout(0, *size, n_robots=robots), n_robots=robots,
                            max_steps=STEPS, obs=PPO.OBS)
            policy = PPO(env.spec, seed=0)
            shapes.add((policy.inputs.width,
                        sum(p.numel() for p in policy.actor.parameters()),
                        sum(p.numel() for p in policy.critic.parameters())))
            self.assertEqual(len(policy.act(env.reset(0))), robots)
        self.assertEqual(len(shapes), 1, f"the networks changed shape with the scenario: {shapes}")

    def test_a_checkpoint_trained_on_four_robots_plays_six(self):
        small = Warehouse(maze_layout(0, 11, 9, n_robots=4), n_robots=4, max_steps=STEPS, obs=PPO.OBS)
        big = Warehouse(maze_layout(0, 21, 21, n_robots=6), n_robots=6, max_steps=STEPS, obs=PPO.OBS)
        trained = PPO(small.spec, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            trained.save(folder)
            bigger = PPO(big.spec, seed=1)
            bigger.load(folder)
            for a, b in zip(trained.actor.parameters(), bigger.actor.parameters()):
                self.assertTrue(torch.equal(a, b))
            obs = big.reset(0)
            bigger.reset()
            self.assertEqual(len(bigger.act(obs)), 6)

    def test_a_bfs_0_checkpoint_restores_its_own_narrower_width(self):
        env = maze_env()
        narrow = PPO(env.spec, seed=0)
        narrow.configure(bfs="0")
        self.assertLess(narrow.inputs.width, PPO(env.spec, seed=0).inputs.width)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            narrow.save(folder)
            self.assertIs(json.loads((folder / "state.json").read_text(encoding="utf-8"))["bfs"],
                          False)
            wide = PPO(env.spec, seed=0)
            wide.load(folder)
            self.assertEqual((wide.inputs.width, wide.inputs.use_bfs),
                             (narrow.inputs.width, False))


class TestTeacher(unittest.TestCase):
    def test_the_imitation_term_only_costs_anything_when_it_is_switched_on(self):
        env = maze_env()
        off = PPO(env.spec, seed=0)
        self.assertEqual(off.bc, 0.0)
        obs = env.reset(0)
        off.reset()
        off.act(obs)
        self.assertIsNone(off.acting[5], "no teacher actions should be computed with bc=0")
        on = PPO(env.spec, seed=0)
        on.configure(bc="0.5")
        on.reset()
        on.act_vec([obs])
        teacher = on.acting[5]
        self.assertEqual(tuple(teacher.shape), (1, ROBOTS))
        mask = on.masks([obs])
        for row in range(ROBOTS):
            self.assertTrue(mask[0, row, teacher[0, row]], "the teacher suggested an illegal move")


if __name__ == "__main__":
    unittest.main()
