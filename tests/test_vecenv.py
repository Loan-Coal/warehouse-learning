"""Several envs stepped together must be the same experience as one env stepped repeatedly.

The equivalence test is the point of this file: --envs 1 has to reproduce rl/loop.py episode for
episode, because that is what says the vectorised path is a speed-up rather than a different
experiment. The other tests cover the thing that would be invisible if it were wrong -- which
observation a finished episode hands back for a value bootstrap.
"""
import unittest
from argparse import Namespace

from env.layout import is_walkable, maze_layout
from env.warehouse import Warehouse
from rl.cli import build, layout_for, make_env, make_parser, parse
from rl.loop import run_episode
from rl.policies import POLICIES
from rl.policy import Policy
from rl.run import build_parser
from rl.vecenv import VecWarehouse
from rl.vecloop import _crossed, train_vec

ROBOTS, STEPS = 3, 30


def settings(*argv):
    return parse(build_parser(), ["--maze", "--new-map", "--robots", str(ROBOTS),
                                  "--max-steps", str(STEPS), "--no-render", *argv])


def vec_of(args, n_envs, obs_features):
    envs = [make_env(args, obs_features, layout_for(args, 0)) for _ in range(n_envs)]

    def episode(n):
        return args.seed + n, layout_for(args, n // n_envs)

    return VecWarehouse(envs, episode)


class TestEquivalence(unittest.TestCase):
    def test_one_env_reproduces_the_plain_loop_episode_for_episode(self):
        args = settings("--train", "--envs", "1", "--episodes", "6")
        make = POLICIES["greedy"]

        plain_env, plain = build(args)
        plain_rows = []
        for k in range(args.episodes):
            plain_rows.append(run_episode(plain_env, plain, args.seed + k, train=True,
                                          layout=layout_for(args, k)))

        vec_env, policy = build(args)
        rows = []
        vec = train_vec(args, vec_env, policy, _Collector(rows), lambda episode: None)

        self.assertEqual(vec.finished, args.episodes)
        self.assertEqual(rows, plain_rows, "--envs 1 diverged from the one-episode loop")
        self.assertEqual(make.OBS, policy.spec.obs_features)

    def test_more_envs_play_the_same_maps_in_the_same_rounds(self):
        args = settings("--train", "--envs", "4", "--episodes", "8")
        vec = vec_of(args, 4, POLICIES["greedy"].OBS)
        vec.reset()
        first = [env.grid for env in vec.envs]
        self.assertEqual(len(set(first)), 1, "a round shares one map, by design")
        self.assertEqual(first[0], tuple(layout_for(args, 0).splitlines()))
        for _ in range(STEPS):
            vec.step([[2] * ROBOTS] * 4)
        second = [env.grid for env in vec.envs]
        self.assertEqual(len(set(second)), 1)
        self.assertNotEqual(second[0], first[0], "the next round should draw a new map")
        self.assertEqual(second[0], tuple(layout_for(args, 1).splitlines()))


class TestTruncationBootstrap(unittest.TestCase):
    """The env has no terminal state, so the return past the step limit is real, not zero.

    A learner therefore needs the observation the episode ENDED on. If step() handed back the
    restarted one instead, every episode boundary would bootstrap from an unrelated state on a
    different map -- and nothing would crash or even look wrong.
    """

    def test_a_finished_episode_reports_the_state_it_ended_on(self):
        args = settings("--envs", "1")
        vec = vec_of(args, 1, POLICIES["greedy"].OBS)
        obs = vec.reset()
        for step in range(STEPS):
            was = vec.envs[0].grid
            obs, _, dones, ended, stats = vec.step([[2] * ROBOTS])
            if step < STEPS - 1:
                self.assertEqual((dones, ended, stats), ([False], [None], []))
                continue
            self.assertEqual(dones, [True])
            self.assertEqual(len(stats), 1)
            self.assertIsNotNone(ended[0])
            self.assertNotEqual(vec.envs[0].grid, was, "a new map means the two are distinguishable")
            for o in ended[0]:
                self.assertTrue(is_walkable(was, o.x, o.y), "the ended state is on the old map")
            for o in obs[0]:
                self.assertTrue(is_walkable(vec.envs[0].grid, o.x, o.y),
                                "and what to act on next is on the new one")

    def test_every_slot_ends_on_the_same_step(self):
        """Lockstep is what lets one spec describe every slot; it is not an accident."""
        args = settings("--envs", "5")
        vec = vec_of(args, 5, POLICIES["greedy"].OBS)
        vec.reset()
        for step in range(STEPS * 2):
            _, _, dones, _, stats = vec.step([[2] * ROBOTS] * 5)
            self.assertIn(len(set(dones)), (1,), f"slots disagreed at step {step}: {dones}")
            self.assertEqual(len(stats), 5 if dones[0] else 0)
        self.assertEqual(vec.finished, 10)

    def test_returns_are_accumulated_per_slot_and_reset_with_the_episode(self):
        args = settings("--envs", "2")
        vec = vec_of(args, 2, POLICIES["greedy"].OBS)
        vec.reset()
        for _ in range(STEPS):
            _, _, _, _, stats = vec.step([[2] * ROBOTS] * 2)
        self.assertEqual(len(stats), 2)
        for row in stats:
            # Turning in place for the whole episode: no deliveries, and the step cost only.
            self.assertEqual(row["deliveries"], 0)
            self.assertAlmostEqual(row["mean_return"], STEPS * -0.05, places=6)
        self.assertEqual(vec.returns[0], [0.0] * ROBOTS, "the next episode starts from zero")


class TestBatchedPolicyDefault(unittest.TestCase):
    def test_the_default_act_vec_is_just_act_per_env(self):
        """A colleague's policy works under --envs without being touched."""
        args = settings("--envs", "3")
        vec = vec_of(args, 3, POLICIES["greedy"].OBS)
        obs = vec.reset()
        policy = POLICIES["greedy"](vec.envs[0].spec, 0)
        policy.spec = vec.envs[0].spec
        policy.reset()
        batched = policy.act_vec(obs)
        self.assertEqual(len(batched), 3)
        self.assertTrue(all(len(one) == ROBOTS for one in batched))
        again = POLICIES["greedy"](vec.envs[0].spec, 0)
        again.spec = vec.envs[0].spec
        again.reset()
        self.assertEqual(batched, [again.act(one) for one in obs])

    def test_the_default_update_vec_is_just_update_per_env(self):
        seen = []

        class Counter(Policy):
            OBS = POLICIES["greedy"].OBS

            def act(self, obs):
                return [2] * len(obs)

            def update(self, obs, actions, rewards, next_obs, done):
                seen.append((len(obs), done))

        env = Warehouse(maze_layout(0, n_robots=ROBOTS), n_robots=ROBOTS, obs=Counter.OBS)
        policy = Counter(env.spec)
        obs = [env.reset(0), env.reset(1)]
        policy.update_vec(obs, [[2] * ROBOTS] * 2, [[0.0] * ROBOTS] * 2, obs, [False, True])
        self.assertEqual(seen, [(ROBOTS, False), (ROBOTS, True)])


class TestEpisodeMilestones(unittest.TestCase):
    """With N envs the episode count jumps by N, so "every N-th" cannot be an equality test."""

    def test_a_milestone_inside_a_jump_still_fires_once(self):
        self.assertTrue(_crossed(0, 4, 2), "episodes 2 and 4 both landed in this jump")
        self.assertTrue(_crossed(8, 12, 10))
        self.assertFalse(_crossed(12, 16, 10))
        self.assertTrue(_crossed(16, 20, 10))
        self.assertFalse(_crossed(0, 4, 0), "0 means never")
        self.assertFalse(_crossed(4, 4, 2), "nothing finished")

    def test_evaluation_and_snapshots_fire_once_per_milestone(self):
        args = settings("--train", "--envs", "4", "--episodes", "12", "--eval-every", "4",
                        "--eval-episodes", "1")
        env, policy = build(args)
        calls = []
        train_vec(args, env, policy, None, calls.append)
        self.assertEqual(calls, [4, 8, 12])


class _Collector:
    """Stands in for rl.run.TrainLog, keeping the rows instead of writing them."""

    def __init__(self, rows):
        self.rows = rows

    def write(self, episode, stats, mode="train"):
        self.rows.append({k: v for k, v in stats.items() if k != "mode"})


if __name__ == "__main__":
    unittest.main()
