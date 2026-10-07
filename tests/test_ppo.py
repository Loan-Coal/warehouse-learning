"""rl/policies/ppo.py: the centralised encoder, the mask, the truncation bootstrap, one real update.

Two of these cover mistakes that produce no error and no obviously wrong number, and so would
otherwise only show up as "PPO does not work on this problem":

  * zeroing the bootstrap at an episode boundary, which teaches the policy that the world ends
    every max_steps and silently caps what any value estimate can be worth;
  * carrying the GAE recursion ACROSS a boundary, which credits one episode's actions with the
    next episode's rewards on a different map.

The mask itself is pinned against the env in tests/test_nav.py; what is checked here is that the
policy stays inside it on both the sampling path and the argmax path.
"""
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import rl.policies.ppo as mod
from env.layout import maze_layout
from env.observation import HEADINGS, RIGHT, ROBOT, obs_type
from env.warehouse import N_ACTIONS, Warehouse
from rl.policies.ppo import PPO, visible
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


def fleet(*robots):
    """Blind observations (no rays hit anything, no task) for robots given as (x, y, heading)."""
    fields = obs_type(PPO.OBS)._fields
    blank = obs_type(PPO.OBS)(*[0] * len(fields))
    return [blank._replace(x=x, y=y, heading=h, target_x=x, target_y=y, period=1, ready=1)
            for x, y, h in robots]


def ahead(o, k, d, kind=ROBOT):
    """The same observation with ray k (0 front, 1 right, 2 back, 3 left) hitting `kind` at d."""
    side = ("front", "right", "back", "left")[k]
    return o._replace(**{f"{side}_d": d, f"{side}_t": kind})


class TestVisible(unittest.TestCase):
    """Who each robot may know about. central=False is the ablation: only what its own rays hit."""

    def setUp(self):
        a, b, c = fleet((2, 2, RIGHT), (5, 2, RIGHT), (3, 3, RIGHT))
        self.obs = [[ahead(a, 0, 3), b, c]]          # a's front ray hits b, three cells east

    def test_a_ray_hit_is_visible_and_a_diagonal_robot_is_not(self):
        got = visible(self.obs, central=False)[0]
        self.assertEqual(got.tolist(), [[False, True, False], [False] * 3, [False] * 3])

    def test_central_sees_every_other_robot_and_never_itself(self):
        got = visible(self.obs, central=True)[0]
        self.assertEqual(got.tolist(), [[False, True, True], [True, False, True], [True, True, False]])

    def test_a_ray_uses_the_robots_heading(self):
        dx, dy = HEADINGS[(RIGHT + 1) % 4]           # right of a robot facing east is south
        a, b = fleet((2, 2, RIGHT), (2 + 2 * dx, 2 + 2 * dy, RIGHT))
        self.assertTrue(visible([[ahead(a, 1, 2), b]], central=False)[0, 0, 1])

    def test_a_human_in_a_ray_is_not_a_robot(self):
        a, b = fleet((2, 2, RIGHT), (5, 2, RIGHT))
        self.assertFalse(visible([[ahead(a, 0, 3, kind=ROBOT + 1), b]], central=False).any())


class TestCentralised(unittest.TestCase):
    def setUp(self):
        self.policy = PPO(maze_env().spec, seed=0)
        self.policy.reset()

    def run_net(self, obs, central):
        x = self.policy.inputs.batch([obs])
        with torch.no_grad():
            return self.policy.actor(x, self.policy.reach([obs], central))

    def test_a_robot_outside_every_ray_changes_the_decision_only_when_central(self):
        near = fleet((2, 2, RIGHT), (5, 2, RIGHT), (3, 3, RIGHT))
        far = [near[0], near[1], near[2]._replace(x=1, y=3, target_x=1, target_y=3)]
        for central, expect_change in ((True, True), (False, False)):
            with self.subTest(central=central):
                same = torch.equal(self.run_net(near, central)[0, 0], self.run_net(far, central)[0, 0])
                self.assertEqual(not same, expect_change)

    def test_permuting_the_robots_permutes_the_answers(self):
        obs = fleet((2, 2, 0), (5, 2, 1), (3, 3, 2), (1, 1, 3))
        order = [2, 0, 3, 1]
        with torch.no_grad():
            x = self.policy.inputs.batch([obs])
            allow = self.policy.reach([obs], True)
            base = self.policy.actor(x, allow)
            moved = self.policy.actor(x[:, order], allow[:, order][:, :, order])
            value = self.policy.critic(x, allow)
            value_moved = self.policy.critic(x[:, order], allow[:, order][:, :, order])
        torch.testing.assert_close(moved, base[:, order])
        torch.testing.assert_close(value_moved, value[:, order])

    def test_a_lone_robot_gets_finite_logits_and_value(self):
        obs = fleet((2, 2, RIGHT))
        x = self.policy.inputs.batch([obs])
        allow = self.policy.reach([obs], True)
        self.assertEqual(tuple(allow.shape), (1, 1, 1))
        with torch.no_grad():
            self.assertTrue(torch.isfinite(self.policy.actor(x, allow)).all())
            self.assertTrue(torch.isfinite(self.policy.critic(x, allow)).all())

    def test_envs_never_mix(self):
        first, second = fleet((2, 2, 0), (5, 2, 1)), fleet((1, 1, 3), (3, 3, 2))
        with torch.no_grad():
            both = self.policy.actor(self.policy.inputs.batch([first, second]),
                                     self.policy.reach([first, second], True))
            alone = self.policy.actor(self.policy.inputs.batch([first]),
                                      self.policy.reach([first], True))
        torch.testing.assert_close(both[:1], alone)


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
                        self.assertTrue(all(type(a) is int and 0 <= a < N_ACTIONS for a in actions))
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
            logits = policy.actor(policy.inputs.batch([obs]), policy.reach([obs])).masked_fill(~mask, mod.NEG)
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

    @patch.object(mod, "ROLLOUT", 24)
    def test_every_bootstrap_is_the_critic_on_the_state_that_followed(self):
        """after[t] is reused from step t+1's value rather than recomputed, except where an
        episode ended: there it must still be the state it ENDED on, not the restarted one."""
        vec = vec_of(2, steps=10)       # boundaries at 10 and 20, inside a 24-step rollout
        policy = PPO(vec.envs[0].spec, seed=0)
        expected, seen = [], {}
        update_vec, gae = policy.update_vec, policy._gae

        def recording(obs, actions, rewards, next_obs, dones):
            with torch.no_grad():
                expected.append(policy.critic(policy.inputs.batch(next_obs), policy.reach(next_obs)).squeeze(-1))
            update_vec(obs, actions, rewards, next_obs, dones)

        def spy(reward, value, after, done):
            seen["after"] = after.clone()
            return gae(reward, value, after, done)

        policy.update_vec, policy._gae = recording, spy
        play(policy, vec, 24)
        torch.testing.assert_close(seen["after"], torch.stack(expected))


class TestUpdate(unittest.TestCase):
    @patch.object(mod, "ROLLOUT", 8)
    def test_a_real_update_runs_and_leaves_everything_finite(self):
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
            acted[0, row] = next(a for a in range(N_ACTIONS) if mask[0, row, a])
        args = [x, mask, acted, torch.zeros(1, ROBOTS), torch.ones(1, ROBOTS),
                torch.full((1, ROBOTS), 3.0), torch.zeros(1, ROBOTS),     # ready all zero
                policy.reach([obs])]
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
        with self.assertRaisesRegex(ValueError, "lr wants a number"):
            policy.configure(lr="fast")

    def test_shaping_off_leaves_the_env_reward_alone(self):
        env = maze_env()
        policy = PPO(env.spec, seed=0)
        obs = env.reset(0)
        policy.reset()
        policy.configure(shaping="0")
        self.assertEqual(policy.learning_rewards(obs, [2.0] * ROBOTS, obs),
                         [mod.REWARD_SCALE * 2.0] * ROBOTS)


class TestStoredAndRecomputed(unittest.TestCase):
    """Before the first gradient step the policy has not moved, so the ratio must be exactly 1.

    This is the check that catches a whole family of silent mistakes at once: a mask built from
    the wrong map at learn time, features rebuilt in a different order, the stored action not
    being the one the log-probability was taken for. Any of those leaves PPO training on a
    corrupted objective while every number still looks plausible -- the ratio simply stops being
    1 at the start, clipping fires immediately, and learning quietly degrades.
    """

    @patch.object(mod, "ROLLOUT", 1_000_000)       # never fires on its own
    def test_the_ratio_is_one_before_the_first_update(self):
        vec = vec_of(3)
        policy = PPO(vec.envs[0].spec, seed=0)
        play(policy, vec, STEPS + 5)                    # spans an episode boundary and a map change
        self.assertGreater(len(policy.rollout), STEPS)

        x, mask, acted, logp, allow = [torch.stack([getattr(s, f) for s in policy.rollout])
                                       for f in ("x", "mask", "acted", "logp", "allow")]
        with torch.no_grad():
            logits = policy.actor(x.flatten(0, 1), allow.flatten(0, 1)).masked_fill(
                ~mask.flatten(0, 1), mod.NEG)
            again = torch.distributions.Categorical(logits=logits).log_prob(acted.flatten(0, 1))
        ratio = (again - logp.flatten(0, 1)).exp()
        self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5),
                        f"ratio drifted from 1: min {ratio.min():.6f} max {ratio.max():.6f}")


class TestShaping(unittest.TestCase):
    """Why the plain difference and not Ng, Harada and Russell's discounted form."""

    def shaping(self, policy, before, after):
        """What the net is paid for the step `before` -> `after`, on top of the env reward."""
        return policy.learning_rewards([before], [0.0], [after])[0]

    def busy_robot(self):
        """A policy plus one observation of a robot that has a task and can reach it."""
        env = maze_env()
        policy = PPO(env.spec, seed=0)
        policy.training = True
        obs = env.reset(0)
        policy.reset()
        rng = random.Random(2)
        for _ in range(STEPS):
            for o in obs:
                if o.has_task and policy.inputs.potential(o) != 0.0:
                    return policy, o
            obs, _, done, _ = env.step([rng.randrange(N_ACTIONS) for _ in range(ROBOTS)])
            if done:
                obs = env.reset(0)
        raise AssertionError("no robot ever had a reachable task")

    def test_standing_still_earns_exactly_nothing(self):
        """Their form is policy-invariant but pays a motionless robot, because Phi is negative.

        (GAMMA - 1) * Phi is positive here, and at any real distance it beats the step cost,
        which buys loitering. That was measured, not guessed. The plain difference pays zero.
        """
        policy, o = self.busy_robot()
        self.assertEqual(self.shaping(policy, o, o), 0.0)
        textbook = mod.SHAPE * (mod.GAMMA * policy.inputs.potential(o) - policy.inputs.potential(o))
        self.assertGreater(textbook, 0.0, "the form we rejected pays a robot for doing nothing")

    def test_it_is_suppressed_when_a_delivery_swaps_the_target(self):
        """Phi jumps discontinuously then, and the jump is not progress the robot made."""
        policy, o = self.busy_robot()
        swapped = o._replace(carrying=1 - o.carrying, target_x=o.target_x + 2)
        self.assertEqual(self.shaping(policy, o, swapped), 0.0)

class TestTransfer(unittest.TestCase):
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


class TestEncoderCheckpoint(unittest.TestCase):
    def test_a_checkpoint_from_the_mean_pool_ppo_is_refused_by_name(self):
        env = maze_env()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            PPO(env.spec, seed=0).save(folder)
            state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
            state.pop("encoder")
            (folder / "state.json").write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ppo-mean"):
                PPO(env.spec, seed=0).load(folder)

    def test_central_is_saved_and_a_typed_value_beats_it(self):
        env = maze_env()
        trained = PPO(env.spec, seed=0)
        trained.configure(central="0")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            trained.save(folder)
            loaded = PPO(env.spec, seed=0)
            loaded.load(folder)
            self.assertFalse(loaded.central)
            typed = PPO(env.spec, seed=0)
            typed.configure(central="1")
            typed.load(folder)
            self.assertTrue(typed.central)
        with self.assertRaisesRegex(ValueError, "central wants 0 or 1"):
            trained.configure(central="maybe")


class TestSettingsAcrossALoad(unittest.TestCase):
    """rl/cli.py configures BEFORE it loads, so load() must not undo what was typed.

    It used to: --checkpoint X --policy-opt eval_greedy=1 quietly evaluated with sampling.
    """

    def test_a_typed_setting_beats_the_checkpoint(self):
        env = maze_env()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            PPO(env.spec, seed=0).save(folder)
            loaded = PPO(env.spec, seed=0)
            loaded.configure(eval_greedy="1", ent="0.05", lr="0.0001")
            loaded.load(folder)
            self.assertEqual((loaded.eval_greedy, loaded.ent, loaded.lr), (True, 0.05, 0.0001))
            self.assertEqual({g["lr"] for g in loaded.opt.param_groups}, {0.0001},
                             "the saved optimiser state carries its own lr")

    def test_an_untyped_setting_comes_from_the_checkpoint(self):
        env = maze_env()
        trained = PPO(env.spec, seed=0)
        trained.configure(shaping="0", ent="0.02")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            trained.save(folder)
            loaded = PPO(env.spec, seed=0)
            loaded.load(folder)
            self.assertEqual((loaded.use_shaping, loaded.ent, loaded.eval_greedy), (False, 0.02, False))


if __name__ == "__main__":
    unittest.main()
