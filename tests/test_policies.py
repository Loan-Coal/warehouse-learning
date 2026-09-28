"""Every registered policy must plug into the loop, both runners and checkpoints.

Registering your policy in rl/policies/__init__.py is enough: these tests pick it up.
The template is tested too, so a freshly copied template always starts from working code.
"""
import json
import tempfile
import unittest
from pathlib import Path

from env.layout import maze_layout
from env.warehouse import N_ACTIONS, TURN_RIGHT, Warehouse
from rl.checkpoint import load_checkpoint, save_checkpoint
from rl.loop import run_episode
from rl.policies import POLICIES
from rl.policies._template import MyAlgo
from rl.policy import Policy, RobotPolicy, independent, shared

CANDIDATES = dict(POLICIES, template_shared=shared(MyAlgo), template_independent=independent(MyAlgo))
STEPS, ROBOTS = 20, 3


def small_env(obs=MyAlgo.OBS, **factors):
    return Warehouse(maze_layout(0, n_robots=ROBOTS), n_robots=ROBOTS, max_steps=STEPS, obs=obs, **factors)


def files(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


class TestEveryPolicy(unittest.TestCase):
    def test_plays_in_train_and_eval_mode(self):
        for name, make in CANDIDATES.items():
            with self.subTest(policy=name):
                # Every factor on: a policy must cope with any scenario, even if it plays badly.
                env = small_env(make.OBS, mixed_crates=True, mixed_robots=True, priority=True, humans=2)
                policy = make(env.spec, 0)
                self.assertIsInstance(policy, Policy)
                for train in (True, False):
                    seen = []
                    run_episode(env, policy, 0, train, lambda obs, actions, *_: seen.append(actions))
                    self.assertEqual(len(seen), STEPS)
                    for actions in seen:
                        self.assertEqual(len(actions), ROBOTS)
                        self.assertTrue(all(type(a) is int and 0 <= a < N_ACTIONS for a in actions))

    def test_checkpoint_round_trip(self):
        # save -> load -> save must write the same files: load restores everything save wrote.
        for name, make in CANDIDATES.items():
            with self.subTest(policy=name), tempfile.TemporaryDirectory() as tmp:
                env = small_env(make.OBS)
                trained = make(env.spec, 0)
                run_episode(env, trained, 0, train=True)
                first, second = Path(tmp, "first"), Path(tmp, "second")
                save_checkpoint(trained, first, name, env.n_robots)
                restored = make(env.spec, 0)
                load_checkpoint(restored, first, name)
                save_checkpoint(restored, second, name, env.n_robots)
                self.assertEqual(files(first), files(second))


class Recorder(RobotPolicy):
    """Turns in place, and records what the loop did to it."""

    def __init__(self, spec, seed=0):
        super().__init__(spec, seed)
        self.updates, self.flags = 0, set()

    def act_one(self, obs):
        self.flags.add(self.training)
        return TURN_RIGHT

    def update_one(self, *step):
        self.updates += 1


class NumpyLikeInt:
    """Behaves like np.int64: usable as an index, but not an int subclass."""

    def __index__(self):
        return TURN_RIGHT


class TestPlumbing(unittest.TestCase):
    def test_update_and_training_flag_follow_the_mode(self):
        env = small_env()
        policy = shared(Recorder)(env.spec)
        run_episode(env, policy, 0, train=False)
        robot = policy.distinct[0]
        self.assertEqual((robot.updates, robot.flags), (0, {False}))
        run_episode(env, policy, 0, train=True)
        self.assertEqual((robot.updates, robot.flags), (STEPS * ROBOTS, {False, True}))

    def test_shared_is_one_instance_independent_is_one_per_robot(self):
        env = small_env()
        self.assertEqual(len({id(r) for r in shared(Recorder)(env.spec).robots}), 1)
        self.assertEqual(len({id(r) for r in independent(Recorder)(env.spec).robots}), ROBOTS)

    def test_numpy_style_integer_actions_are_accepted(self):
        class Central(Policy):
            def act(self, obs):
                return [NumpyLikeInt() for _ in obs]

        env = small_env()
        self.assertEqual(run_episode(env, Central(env.spec), 0, train=False)["invalid"], 0)

    def test_on_step_can_stop_the_episode(self):
        env = small_env()
        calls = []
        run_episode(env, shared(Recorder)(env.spec), 0, False, lambda *_: calls.append(1) or True)
        self.assertEqual(len(calls), 1)

    def test_checkpoint_for_another_contract_or_policy_is_refused(self):
        env = small_env()
        policy = shared(Recorder)(env.spec)
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(policy, tmp, "recorder", ROBOTS)
            with self.assertRaisesRegex(ValueError, "not 'other'"):
                load_checkpoint(policy, tmp, "other")
            meta = Path(tmp, "meta.json")
            meta.write_text(json.dumps(dict(json.loads(meta.read_text()), contract_version=0)))
            with self.assertRaisesRegex(ValueError, "contract v0"):
                load_checkpoint(policy, tmp, "recorder")
            with self.assertRaisesRegex(ValueError, "not a checkpoint"):
                load_checkpoint(policy, Path(tmp, "missing"), "recorder")

    def test_checkpoint_with_other_obs_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(shared(Recorder)(small_env().spec), tmp, "recorder", ROBOTS)
            other = small_env(obs=("position", "rays"))
            with self.assertRaisesRegex(ValueError, "trained with OBS"):
                load_checkpoint(shared(Recorder)(other.spec), tmp, "recorder")

    def test_spec_follows_a_new_map(self):
        env = small_env()
        policy = independent(Recorder)(env.spec)
        run_episode(env, policy, 0, train=False, layout=maze_layout(5, 13, 11, n_robots=ROBOTS))
        self.assertEqual(len(policy.distinct[1].spec.grid), 11)

    def test_independent_checkpoint_needs_the_same_robot_count(self):
        env, bigger = small_env(), Warehouse(maze_layout(0, n_robots=4), n_robots=4)
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(independent(MyAlgo)(env.spec), tmp, "t", ROBOTS)
            with self.assertRaisesRegex(ValueError, "same --robots"):
                load_checkpoint(independent(MyAlgo)(bigger.spec), tmp, "t")
            # Loading them as shared weights looks for q.json at the top: a clear error, no traceback.
            with self.assertRaisesRegex(ValueError, "has no q.json"):
                load_checkpoint(shared(MyAlgo)(env.spec), tmp, "t")
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(shared(MyAlgo)(env.spec), tmp, "t", ROBOTS)
            load_checkpoint(shared(MyAlgo)(bigger.spec), tmp, "t")   # shared weights fit any count


if __name__ == "__main__":
    unittest.main()
