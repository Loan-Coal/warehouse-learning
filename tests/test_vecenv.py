"""Several envs stepped together must be the same experience as one env stepped repeatedly.

The equivalence test is the point of this file: --envs 1 has to reproduce rl/loop.py episode for
episode, because that is what says the vectorised path is a speed-up rather than a different
experiment. The other tests cover the thing that would be invisible if it were wrong -- which
observation a finished episode hands back for a value bootstrap.
"""
import unittest

from env.layout import is_walkable, maze_layout
from rl.cli import build, layout_for, make_env, parse
from rl.loop import run_episode
from rl.policies import POLICIES
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


class TestEvaluationDoesNotLeak(unittest.TestCase):
    """rl.run.evaluate plays on its own env and its own held-out map, via run_episode -- which
    sets policy.spec and policy.training and does not put them back.

    The vectorised loop sets them once per round rather than once per episode, so anything left
    behind survives until the next round. The policy then builds its action masks from the
    EVALUATION maze while acting in the training ones, and spends a whole round playing illegal
    moves. Nothing raises. It was visible only as a spike in the invalid column of the chart.
    """

    def test_training_resumes_on_the_training_map_after_an_evaluation(self):
        args = settings("--train", "--envs", "4", "--episodes", "8", "--eval-every", "4",
                        "--eval-episodes", "2")
        env, policy = build(args)
        seen = []

        def evaluate(episode):
            # What rl.run.evaluate does to the policy, without needing the whole of rl.run.
            other = make_env(args, env.spec.obs_features, maze_layout(777, 11, 9, n_robots=ROBOTS))
            run_episode(other, policy, 90_000, train=False)
            seen.append(policy.spec.grid)

        train_vec(args, env, policy, None, evaluate)
        self.assertTrue(seen, "the evaluation hook never ran")
        self.assertNotEqual(seen[0], env.grid, "the evaluation should be on another map")
        self.assertEqual(policy.spec.grid, env.grid, "the policy is still on the evaluation map")
        self.assertIs(policy.training, True, "training was left switched off after an evaluation")

    def test_no_invalid_actions_survive_an_evaluation(self):
        """The symptom the bug actually produced, end to end, with a policy that masks."""
        args = settings("--policy", "ppo", "--train", "--envs", "2", "--episodes", "6",
                        "--eval-every", "2", "--eval-episodes", "1")
        env, policy = build(args)
        rows = []

        def evaluate(episode):
            other = make_env(args, env.spec.obs_features, maze_layout(555, 11, 9, n_robots=ROBOTS))
            run_episode(other, policy, 90_000, train=False)

        train_vec(args, env, policy, _Collector(rows), evaluate)
        worst = max(row["invalid"] for row in rows)
        self.assertEqual(worst, 0, f"a training episode played {worst} illegal moves after an eval")


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
