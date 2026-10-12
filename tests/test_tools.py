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
from rl.run import BASE_FIELDS, build_parser, check_args, eval_settings
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
    """Evaluation must never replay a training map (nothing fails if it does: the numbers are just wrong)."""

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

    def test_training_maps_running_into_the_eval_maps_is_refused(self):
        parser = build_parser()
        args = parse(parser, ["--scenario", "hard", "--new-map", "--train", "--episodes", "200",
                              "--eval-every", "50", "--eval-seed", "100", "--no-render"])
        self.assertEqual(train_seeds(args, 200), (0, 199))
        with self.assertRaises(SystemExit):
            check_args(parser, args)
        short = parse(parser, ["--scenario", "hard", "--new-map", "--train", "--episodes", "50",
                               "--eval-every", "50", "--eval-seed", "100", "--no-render"])
        check_args(parser, short)      # seeds 0-49 stay below --eval-seed

    def test_evaluate_plays_the_held_out_maps_and_not_the_training_one(self):
        """eval_settings() producing disjoint maps is not enough: evaluate() has to play them."""
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
            run("rl.bench", "greedy", f"template={folder}", "--scenarios", "basic", "humans",
                "--episodes", "2", "--out", str(Path(tmp, "bench")))
            self.assertTrue(Path(tmp, "bench.csv").is_file())
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                self.skipTest("matplotlib not installed")
            run("rl.plot", str(folder))
            self.assertTrue((folder / "learning_curve.png").is_file())


class TestBenchPolicyOpts(unittest.TestCase):
    """--policy-opt in rl.bench: how evaluation with argmax actions (eval_greedy=1) is benchmarked."""

    def test_an_opt_reaches_the_policy(self):
        rows = evaluate("ppo", "basic", 1, 1000, policy_opts=["eval_greedy=1"])
        self.assertEqual(len(rows), 1)

    def test_an_opt_beats_the_checkpoint_it_is_typed_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            run("rl.run", "--policy", "ppo", "--train", "--max-steps", "10", "--episodes", "1",
                "--no-render", "--save", tmp)
            args = parse(make_parser(), ["--checkpoint", tmp, "--policy-opt", "eval_greedy=1"])
            _, policy = build(args)
            self.assertTrue(policy.eval_greedy, "the checkpoint's eval_greedy=False won")

    def test_an_opt_the_policy_does_not_take_is_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, "eval_greedy"):
            evaluate("greedy", "basic", 1, 1000, policy_opts=["eval_greedy=1"])


class TestTraining(unittest.TestCase):
    def test_a_log_has_the_policys_own_columns_after_the_base_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            run("rl.run", "--policy", "ppo", "--train", "--max-steps", "10", "--episodes", "2",
                "--no-render", "--save", tmp)
            with open(Path(tmp, "train_log.csv"), encoding="utf-8") as f:
                header = next(csv.reader(f))
            base = len(BASE_FIELDS)
            self.assertEqual(header[:base], list(BASE_FIELDS))
            self.assertEqual(header[base:], ["loss", "entropy", "clipfrac", "explained_var"])

    def test_training_can_be_watched_in_the_terminal(self):
        out = run("rl.run", "--policy", "greedy", "--train", "--max-steps", "3", "--episodes", "1",
                  "--delay", "0")
        self.assertEqual(out.count("legend:"), 3, "one rendered frame per step")


class TestMaxMinutes(unittest.TestCase):
    def test_a_negative_budget_is_refused(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            check_args(parser, parse(parser, ["--train", "--max-minutes", "-1", "--no-render"]))

    def test_the_checkpoint_records_the_episodes_actually_played(self):
        """A run stopped by the clock played fewer episodes than --episodes allowed, in whole rounds."""
        with tempfile.TemporaryDirectory() as tmp:
            run("rl.run", "--policy", "ppo", "--train", "--envs", "2", "--max-steps", "10",
                "--episodes", "100000", "--max-minutes", "0.03", "--no-render", "--save", tmp)
            played = read_meta(tmp)["episode"]
            self.assertTrue(0 < played < 100000 and played % 2 == 0, played)


if __name__ == "__main__":
    unittest.main()
