"""The command-line tools: flag precedence, training logs and snapshots, benchmark, charts."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rl.bench import evaluate
from rl.checkpoint import read_meta
from rl.cli import build, env_args, layout_for, make_parser, parse, train_seeds
from rl.loop import run_episode
from rl.policies import POLICIES
from rl.policy import Policy
from rl.run import build_parser, check_args, eval_settings
from rl.run import evaluate as run_evaluate     # rl.bench.evaluate is imported above under its own name


class MapSpy(Policy):
    """Plays greedy, but records the map it was actually given at the start of each episode."""

    OBS = POLICIES["greedy"].OBS

    def __init__(self, spec, seed=0):
        super().__init__(spec, seed)
        self.inner = POLICIES["greedy"](spec, seed)
        self.seen = []

    def reset(self):
        self.inner.spec = self.spec
        self.inner.reset()
        self.seen.append(self.spec.grid)

    def act(self, obs):
        return self.inner.act(obs)

ROOT = Path(__file__).resolve().parent.parent


def run(*argv):
    """python -m <argv> from the repo root; returns stdout, fails the test on a non-zero exit."""
    done = subprocess.run([sys.executable, "-m", *argv], cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        raise AssertionError(f"{' '.join(argv)} failed:\n{done.stderr}")
    return done.stdout


class TestFlags(unittest.TestCase):
    def test_everything_is_off_by_default(self):
        args = parse(make_parser(), [])
        self.assertEqual(args.policy, "greedy")
        self.assertEqual(env_args(args), {"seed": 0, "robots": 2, "map": "default", "size": (11, 9),
                                          "max_steps": 100, "new_map": False, "mixed_crates": False,
                                          "mixed_robots": False, "priority": False, "humans": 0})

    def test_typed_flag_beats_scenario(self):
        args = parse(make_parser(), ["--scenario", "hard", "--humans", "5"])
        self.assertEqual((args.humans, args.robots, args.map, args.priority), (5, 4, "maze", True))
        self.assertEqual(parse(make_parser(), ["--scenario", "hard", "--random-map"]).map, "random")

    def test_build_uses_the_policy_obs_and_the_factors(self):
        env, policy = build(parse(make_parser(), ["--scenario", "hard", "--policy", "template"]))
        self.assertEqual((env.n_robots, env.n_humans, env.priority), (4, 2, True))
        self.assertEqual(policy.spec.obs_features, ("position", "heading", "task", "rays"))

    def test_new_map_changes_the_map_per_episode(self):
        args = parse(make_parser(), ["--maze", "--new-map"])
        self.assertNotEqual(layout_for(args, 0), layout_for(args, 1))
        fixed = parse(make_parser(), ["--maze"])
        self.assertEqual(layout_for(fixed, 0), layout_for(fixed, 1))
        with self.assertRaises(ValueError):
            layout_for(parse(make_parser(), ["--new-map"]), 0)

    def test_unknown_names_are_refused(self):
        with self.assertRaises(ValueError):
            build(parse(make_parser(), ["--policy", "nope"]))
        with self.assertRaises(SystemExit):
            parse(make_parser(), ["--scenario", "nope"])


class TestHeldOutEvaluation(unittest.TestCase):
    """Evaluation must never replay a training map.

    rl.run used to hand evaluate() the training env with layout=None, which means "keep whatever
    map is installed", so under --new-map it measured the policy on the maze it had just trained
    on. Every number from such a run is meaningless, and nothing failed when it happened.
    """

    def args(self, *argv):
        return parse(build_parser(), [*argv, "--no-render"])

    def test_eval_is_held_out_even_when_training_used_one_fixed_maze(self):
        # Then it answers the more interesting question: does the policy generalise off that maze?
        args = self.args("--maze", "--train", "--eval-every", "1")
        held_out = {layout_for(eval_settings(args), j) for j in range(args.eval_episodes)}
        self.assertEqual(len(held_out), args.eval_episodes)
        self.assertNotIn(layout_for(args, 0), held_out)

    def test_the_default_map_has_nothing_to_hold_out(self):
        args = self.args("--train", "--eval-every", "1")
        self.assertEqual(layout_for(eval_settings(args), 0), layout_for(args, 0))

    def test_eval_can_use_a_different_fleet_and_maze_size(self):
        args = self.args("--scenario", "hard", "--train", "--eval-every", "1",
                         "--eval-robots", "6", "--eval-size", "21", "21")
        settings = eval_settings(args)
        self.assertEqual((settings.robots, tuple(settings.size)), (6, (21, 21)))
        self.assertEqual(len(layout_for(settings, 0).splitlines()), 21)
        self.assertEqual((args.robots, tuple(args.size)), (4, (11, 9)), "training is untouched")

    def test_eval_scenario_overrides_the_training_preset(self):
        args = self.args("--scenario", "hard", "--train", "--eval-every", "1",
                         "--eval-scenario", "big")
        settings = eval_settings(args)
        self.assertEqual((settings.robots, tuple(settings.size), settings.humans), (6, (21, 21), 3))

    def test_maze_pool_bounds_the_training_seeds(self):
        args = self.args("--maze", "--new-map", "--maze-pool", "4", "--train", "--episodes", "10")
        maps = [layout_for(args, k) for k in range(10)]
        self.assertEqual(len(set(maps)), 4, "the pool should cycle")
        self.assertEqual(maps[0], maps[4])
        self.assertEqual(train_seeds(args, 10), (0, 3))
        self.assertEqual(train_seeds(self.args("--maze", "--new-map", "--episodes", "10"), 10), (0, 9))

    def test_training_maps_running_into_the_eval_maps_is_refused(self):
        parser = build_parser()
        args = parse(parser, ["--scenario", "hard", "--new-map", "--train", "--episodes", "200",
                              "--eval-every", "50", "--eval-seed", "100", "--no-render"])
        with self.assertRaises(SystemExit):
            check_args(parser, args)
        bounded = parse(parser, ["--scenario", "hard", "--new-map", "--maze-pool", "50", "--train",
                                 "--episodes", "200", "--eval-every", "50", "--eval-seed", "100",
                                 "--no-render"])
        check_args(parser, bounded)      # a pool keeps training below --eval-seed

    def test_evaluate_plays_the_held_out_maps_and_not_the_training_one(self):
        """The regression test for the actual bug: evaluate() inheriting the training map.

        eval_settings() producing disjoint maps is not enough -- evaluate() has to use them. The
        old code called run_episode(..., layout=None), which keeps whatever map is installed.
        """
        parser = build_parser()
        args = parse(parser, ["--scenario", "hard", "--new-map", "--train", "--episodes", "4",
                              "--max-steps", "20", "--eval-every", "4", "--eval-episodes", "3",
                              "--no-render"])
        env, _ = build(args)
        spy = MapSpy(env.spec)
        trained_on = [layout_for(args, k) for k in range(args.episodes)]
        for k, layout in enumerate(trained_on):
            run_episode(env, spy, args.seed + k, train=True, layout=layout)
        installed = len(spy.seen)

        run_evaluate(env, spy, args, args.episodes, log=None)
        played = spy.seen[installed:]
        self.assertEqual(len(played), 3)
        expected = [layout_for(eval_settings(args), j) for j in range(3)]
        self.assertEqual(played, [tuple(m.splitlines()) for m in expected])
        self.assertNotIn(tuple(trained_on[-1].splitlines()), played,
                         "evaluation inherited the last training map")

    def test_eval_rows_are_logged_against_the_episode_that_was_printed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp, "ev")
            out = run("rl.run", "--policy", "greedy", "--maze", "--new-map", "--max-steps", "20",
                      "--train", "--episodes", "4", "--eval-every", "2", "--eval-episodes", "2",
                      "--no-render", "--save", str(folder))
            with open(folder / "train_log.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            train = [r for r in rows if r["mode"] == "train"]
            held_out = [r for r in rows if r["mode"] == "eval"]
            self.assertEqual(len(train), 4)
            self.assertEqual(len(held_out), 4, "two eval points, two episodes each")
            self.assertEqual(sorted({r["episode"] for r in held_out}), ["2", "4"])
            self.assertIn("eval @ 2:", out)
            self.assertIn("eval @ 4:", out)


class TestPipeline(unittest.TestCase):
    """Train with snapshots -> replay a snapshot with its settings -> benchmark -> chart."""

    def test_train_replay_bench_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp, "tmpl")
            run("rl.run", "--policy", "template", "--scenario", "mixed", "--max-steps", "20", "--train",
                "--episodes", "4", "--no-render", "--save", str(folder), "--snapshot-every", "2")
            with open(folder / "train_log.csv", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 4)
            meta = read_meta(folder / "ep_000002")
            self.assertEqual((meta["policy"], meta["episode"], meta["env_args"]["mixed_robots"]),
                             ("template", 2, True))

            # --checkpoint alone brings back the policy and the training settings.
            args = parse(make_parser(), ["--checkpoint", str(folder / "ep_000004")])
            self.assertEqual((args.policy, args.mixed_crates, args.max_steps), ("template", True, 20))
            out = run("rl.run", "--checkpoint", str(folder / "ep_000004"), "--no-render")
            self.assertIn("episode 0:", out)

            # The benchmark ignores training settings: every entry plays the scenario's env.
            fair = parse(make_parser(), ["--checkpoint", str(folder), "--scenario", "basic"], env_from_checkpoint=False)
            self.assertEqual((fair.policy, fair.mixed_crates, fair.max_steps), ("template", False, 100))
            rows = evaluate(f"template={folder}", "basic", 2, 1000)
            self.assertEqual([r["episode"] for r in rows], [0, 1])
            self.assertEqual(rows, evaluate(f"template={folder}", "basic", 2, 1000), "same seeds, same result")
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                self.skipTest("matplotlib not installed")
            run("rl.bench", "greedy", f"template={folder}", "--scenarios", "basic", "humans",
                "--episodes", "2", "--out", str(Path(tmp, "bench")))
            self.assertTrue(Path(tmp, "bench.csv").is_file() and Path(tmp, "bench.png").is_file())
            run("rl.plot", str(folder))
            self.assertTrue((folder / "learning_curve.png").is_file())


if __name__ == "__main__":
    unittest.main()
