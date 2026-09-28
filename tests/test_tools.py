"""The command-line tools: flag precedence, training logs and snapshots, benchmark, charts."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rl.bench import evaluate
from rl.checkpoint import read_meta
from rl.cli import build, env_args, layout_for, make_parser, parse

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
