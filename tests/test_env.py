"""Contract tests for env/: dynamics, observation, rewards and the get_state schema.

Both teams depend on these; a change that breaks one is a cross-team change.
Run from the repo root: python -m unittest discover -s tests
"""
import random
import unittest

from env.layout import DEFAULT_MAP, cells_of, maze_layout, parse, problems, random_layout
from env.warehouse import Warehouse

UP, DOWN, LEFT, RIGHT, WAIT, INTERACT = range(6)

# Open rows on top so robots can be lined up freely; the bottom row makes the map valid.
OPEN_MAP = """
.....
.....
S...D
"""


def env_with_robots(*positions, layout=OPEN_MAP, alpha=0.0):
    """An env with no task generation and robots placed exactly."""
    env = Warehouse(layout, n_robots=len(positions), alpha=alpha, task_prob=0.0)
    env.reset(0)
    for robot, pos in zip(env.robots, positions):
        robot.pos = pos
    return env


def give_task(env, kind, robot=0):
    env.queue.append(env._new_task(kind))
    env._assign_idle()
    return env.robots[robot].task


def beside(cell):
    """An aisle cell adjacent to a DEFAULT_MAP shelf or dock: row-1 shelves are served from
    row 0, row-2 shelves from row 3, docks (row 5) from row 4."""
    x, y = cell
    return (x, {1: 0, 2: 3, 5: 4}[y])


class TestMovementAndBlocking(unittest.TestCase):
    def test_free_move(self):
        env = env_with_robots((0, 0))
        _, rewards, _, info = env.step([RIGHT])
        self.assertEqual(env.robots[0].pos, (1, 0))
        self.assertAlmostEqual(rewards[0], -0.05)
        self.assertEqual(info["blocked"], 0)

    def test_swap_is_blocked(self):
        env = env_with_robots((1, 0), (2, 0))
        _, rewards, _, info = env.step([RIGHT, LEFT])
        self.assertEqual([r.pos for r in env.robots], [(1, 0), (2, 0)])
        self.assertEqual(info["blocked"], 2)
        self.assertAlmostEqual(rewards[0], -1.05)

    def test_same_cell_contest_has_one_winner(self):
        winners = set()
        for seed in range(20):
            env = env_with_robots((1, 0), (3, 0))
            env.reset(seed)
            env.robots[0].pos, env.robots[1].pos = (1, 0), (3, 0)
            _, rewards, _, info = env.step([RIGHT, LEFT])
            positions = [r.pos for r in env.robots]
            self.assertEqual(info["blocked"], 1)
            self.assertIn(positions, [[(2, 0), (3, 0)], [(1, 0), (2, 0)]])
            winner = positions.index((2, 0))
            self.assertAlmostEqual(rewards[winner], -0.05)
            self.assertAlmostEqual(rewards[1 - winner], -1.05)
            winners.add(winner)
        self.assertEqual(winners, {0, 1}, "the winner is random, not always the same robot")

    def test_three_way_contest_has_one_winner(self):
        env = env_with_robots((1, 1), (3, 1), (2, 0))
        _, _, _, info = env.step([RIGHT, LEFT, DOWN])
        self.assertEqual(sum(r.pos == (2, 1) for r in env.robots), 1)
        self.assertEqual(info["blocked"], 2)

    def test_rotation_without_pairwise_swap_is_allowed(self):
        # Four robots turning around a 2x2 square: nobody swaps or enters a stayer's cell.
        env = env_with_robots((0, 0), (1, 0), (1, 1), (0, 1))
        _, _, _, info = env.step([RIGHT, DOWN, LEFT, UP])
        self.assertEqual([r.pos for r in env.robots], [(1, 0), (1, 1), (0, 1), (0, 0)])
        self.assertEqual(info["blocked"], 0)

    def test_move_into_a_staying_robot_is_blocked(self):
        env = env_with_robots((1, 0), (2, 0), (3, 0))
        _, _, _, info = env.step([RIGHT, WAIT, LEFT])
        self.assertEqual([r.pos for r in env.robots], [(1, 0), (2, 0), (3, 0)])
        self.assertEqual(info["blocked"], 2)

    def test_only_the_blocked_robot_is_penalised(self):
        env = env_with_robots((1, 0), (2, 0))
        _, rewards, _, _ = env.step([RIGHT, WAIT])
        self.assertAlmostEqual(rewards[0], -1.05)
        self.assertAlmostEqual(rewards[1], -0.05)

    def test_follow_the_leader_is_allowed(self):
        env = env_with_robots((1, 0), (2, 0))
        _, _, _, info = env.step([RIGHT, RIGHT])
        self.assertEqual([r.pos for r in env.robots], [(2, 0), (3, 0)])
        self.assertEqual(info["blocked"], 0)

    def test_chain_block_resolves(self):
        env = env_with_robots((1, 0), (2, 0), (3, 0))
        _, _, _, info = env.step([RIGHT, RIGHT, WAIT])
        self.assertEqual([r.pos for r in env.robots], [(1, 0), (2, 0), (3, 0)])
        self.assertEqual(info["blocked"], 2)

    def test_moves_into_obstacles_are_invalid(self):
        for pos, action in [((0, 0), UP), ((0, 0), LEFT), ((0, 1), DOWN), ((4, 1), DOWN)]:
            env = env_with_robots(pos)
            _, rewards, _, info = env.step([action])
            self.assertEqual(env.robots[0].pos, pos)
            self.assertEqual(info["invalid"], 1)
            self.assertAlmostEqual(rewards[0], -0.15)


class TestTasks(unittest.TestCase):
    def test_interact_while_idle_is_invalid(self):
        env = env_with_robots((0, 0))
        _, rewards, _, info = env.step([INTERACT])
        self.assertEqual(info["invalid"], 1)
        self.assertAlmostEqual(rewards[0], -0.15)

    def test_retrieve_cycle(self):
        env = env_with_robots((0, 0), layout=DEFAULT_MAP)
        crates_before = len(env.get_state()["crates"])
        task = give_task(env, "retrieve")
        env.robots[0].pos = beside(task.pickup)
        env.step([INTERACT])
        self.assertIsNotNone(env.robots[0].crate)
        env.robots[0].pos = beside(task.dropoff)
        _, rewards, _, info = env.step([INTERACT])
        self.assertEqual(info["deliveries"], 1)
        self.assertAlmostEqual(rewards[0], 9.95)
        self.assertIsNone(env.robots[0].task)
        self.assertEqual(len(env.get_state()["crates"]), crates_before - 1)

    def test_store_cycle(self):
        env = env_with_robots((0, 0), layout=DEFAULT_MAP)
        task = give_task(env, "store")
        self.assertIn(task.pickup, env.crate_at, "store crate appears on the dock at assignment")
        env.robots[0].pos = beside(task.pickup)
        env.step([INTERACT])
        env.robots[0].pos = beside(task.dropoff)
        _, rewards, _, info = env.step([INTERACT])
        self.assertEqual(info["deliveries"], 1)
        self.assertAlmostEqual(rewards[0], 9.95)
        self.assertIn(task.dropoff, env.crate_at)

    def test_interact_away_from_target_is_invalid(self):
        env = env_with_robots((3, 4), layout=DEFAULT_MAP)
        task = give_task(env, "retrieve")
        self.assertNotIn(task.pickup, [(3, 3), (3, 5), (2, 4), (4, 4)])
        _, _, _, info = env.step([INTERACT])
        self.assertEqual(info["invalid"], 1)

    def test_nearest_dock_ties_go_to_lowest_index(self):
        env = env_with_robots((0, 0), layout=DEFAULT_MAP)
        env.crate_at = {(3, 1): 99}
        task = give_task(env, "retrieve")
        self.assertEqual(task.dropoff, (0, 5))

    def test_retrieve_delivery_keeps_a_store_dock_reserved(self):
        env = env_with_robots((0, 0), (6, 0), layout=DEFAULT_MAP)
        env.crate_at = {(1, 1): 50, (6, 5): 51}  # a crate on (6, 5) leaves only (0, 5) for a store
        retrieve = give_task(env, "retrieve", robot=0)
        store = give_task(env, "store", robot=1)
        self.assertEqual((retrieve.dropoff, store.pickup), ((0, 5), (0, 5)))
        env.robots[0].pos = beside((1, 1))
        env.step([INTERACT, WAIT])
        env.robots[0].pos = beside((0, 5))
        env.step([INTERACT, WAIT])
        self.assertIn((0, 5), env.reserved, "store task still holds its dock")
        self.assertIn((0, 5), env.crate_at)

    def test_reservations_and_invariants_hold(self):
        for seed in range(5):
            env = Warehouse(DEFAULT_MAP, n_robots=3, task_prob=1.0)
            env.reset(seed)
            rng = random.Random(seed)
            for _ in range(100):
                env.step([rng.randrange(6) for _ in range(3)])
                tasks = [r.task for r in env.robots if r.task] + list(env.queue)
                shelves = [t.pickup if t.kind == "retrieve" else t.dropoff for t in tasks]
                store_docks = [t.pickup for t in tasks if t.kind == "store"]
                self.assertEqual(len(shelves), len(set(shelves)))
                self.assertEqual(len(store_docks), len(set(store_docks)))
                positions = [r.pos for r in env.robots]
                self.assertEqual(len(positions), len(set(positions)))
                self.assertTrue(all(parse(DEFAULT_MAP)[y][x] == "." for x, y in positions))
                self.assertLessEqual(len(env.get_state()["crates"]), 6 + 2)  # shelves + docks


class TestObservation(unittest.TestCase):
    def test_shape_and_idle_target(self):
        env = env_with_robots((2, 0))
        obs = env.step([WAIT])[0][0]
        self.assertEqual(len(obs), 14)
        self.assertEqual(obs[:6], (2, 0, 0, 0, 2, 0))

    def test_rays(self):
        env = env_with_robots((0, 0), (3, 0), layout=DEFAULT_MAP)
        env.crate_at = {}
        up, down, left, right = env._rays(env.robots[0])
        self.assertEqual(up, (1, 1))       # map edge
        self.assertEqual(left, (1, 1))
        self.assertEqual(right, (3, 5))    # robot 1 at distance 3
        self.assertEqual(down, (5, 4))     # dock at (0, 5)

    def test_rays_occlusion_range_and_crates(self):
        env = env_with_robots((0, 1), (6, 3), layout=DEFAULT_MAP)
        env.crate_at = {(1, 1): 7}
        self.assertEqual(env._rays(env.robots[0])[3], (1, 3))  # shelf with crate hides the rest
        env.crate_at = {}
        self.assertEqual(env._rays(env.robots[0])[3], (1, 2))  # empty shelf
        # From (0, 3) looking right, the robot at (6, 3) is 6 away: out of range 5.
        env.robots[0].pos = (0, 3)
        self.assertEqual(env._rays(env.robots[0])[3], (0, 0))

    def test_target_follows_task_stage(self):
        env = env_with_robots((0, 0), layout=DEFAULT_MAP)
        task = give_task(env, "retrieve")
        obs = env._observe(0)
        self.assertEqual(obs[2:6], (0, 1, *task.pickup))
        env.robots[0].crate = env.crate_at.pop(task.pickup)
        self.assertEqual(env._observe(0)[2:6], (1, 1, *task.dropoff))


class TestRewardMixing(unittest.TestCase):
    def test_alpha_zero_is_individual(self):
        env = env_with_robots((1, 0), (4, 0), alpha=0.0)
        _, rewards, _, _ = env.step([UP, WAIT])  # robot 0 bumps the edge
        self.assertAlmostEqual(rewards[0], -0.15)
        self.assertAlmostEqual(rewards[1], -0.05)

    def test_alpha_one_is_shared(self):
        env = env_with_robots((1, 0), (4, 0), alpha=1.0)
        _, rewards, _, _ = env.step([UP, WAIT])
        self.assertAlmostEqual(rewards[0], rewards[1])
        self.assertAlmostEqual(rewards[0], -0.10)


class TestEpisodeAndApi(unittest.TestCase):
    def test_seed_is_deterministic(self):
        def rollout():
            env = Warehouse(DEFAULT_MAP, n_robots=2)
            obs = [env.reset(42)]
            rng = random.Random(1)
            for _ in range(100):
                obs.append(env.step([rng.randrange(6), rng.randrange(6)]))
            return obs
        self.assertEqual(rollout(), rollout())

    def test_truncation_at_max_steps(self):
        env = Warehouse(DEFAULT_MAP, max_steps=3)
        env.reset(0)
        for _ in range(2):
            self.assertFalse(env.step([WAIT, WAIT])[2])
        _, _, done, info = env.step([WAIT, WAIT])
        self.assertTrue(done)
        self.assertTrue(info["truncated"])
        with self.assertRaises(RuntimeError):
            env.step([WAIT, WAIT])

    def test_bad_actions_are_rejected(self):
        env = Warehouse(DEFAULT_MAP)
        env.reset(0)
        with self.assertRaises(ValueError):
            env.step([WAIT])
        with self.assertRaises(ValueError):
            env.step([WAIT, 6])
        with self.assertRaises(ValueError):
            env.step([2.0, WAIT])

    def test_get_state_schema_and_isolation(self):
        env = Warehouse(DEFAULT_MAP)
        env.reset(0)
        state = env.get_state()
        self.assertEqual(
            set(state), {"width", "height", "grid", "robots", "crates", "queue_len", "step"})
        self.assertEqual((state["width"], state["height"]), (7, 6))
        self.assertEqual(len(state["crates"]), 4)
        self.assertEqual(set(state["robots"][0]), {"x", "y", "carrying"})
        self.assertEqual(set(state["crates"][0]), {"id", "x", "y", "carried_by"})
        state["robots"][0]["x"] = 99
        state["grid"][0] = "XXXXXXX"
        self.assertNotEqual(env.get_state()["robots"][0]["x"], 99)
        self.assertEqual(env.get_state()["grid"][0], ".......")

    def test_render_mentions_every_robot(self):
        env = Warehouse(DEFAULT_MAP)
        env.reset(0)
        text = env.render()
        self.assertIn("0", text)
        self.assertIn("1", text)


class TestLayouts(unittest.TestCase):
    def test_default_map_is_valid(self):
        self.assertEqual(problems(parse(DEFAULT_MAP), 2), [])

    def test_random_layouts_are_valid_and_deterministic(self):
        for seed in range(200):
            text = random_layout(seed)
            self.assertEqual(problems(parse(text), 2), [], text)
            self.assertEqual(text, random_layout(seed))

    def test_mazes_are_valid_and_deterministic(self):
        for seed in range(200):
            text = maze_layout(seed)
            grid = parse(text)
            self.assertEqual((len(grid[0]), len(grid)), (11, 9))
            self.assertEqual(problems(grid, 2), [], text)
            self.assertEqual(text, maze_layout(seed))

    def test_maze_docks_are_counted_and_spread_out(self):
        for seed in range(50):
            grid = parse(maze_layout(seed))
            xs = [x for x, _ in cells_of(grid, "D")]
            self.assertEqual(len(xs), 2, f"seed {seed}")
            self.assertLessEqual(min(xs), 11 // 2, f"seed {seed}")    # one in each half of the bottom row
            self.assertGreater(max(xs), 11 // 2, f"seed {seed}")
        self.assertEqual(len(cells_of(parse(maze_layout(0, width=15, height=13, docks=3)), "D")), 3)
        with self.assertRaises(ValueError):
            maze_layout(0, docks=0)

    def test_maze_size_is_a_parameter(self):
        grid = parse(maze_layout(0, width=15, height=13, n_robots=4))
        self.assertEqual((len(grid[0]), len(grid)), (15, 13))
        self.assertEqual(problems(grid, 4), [])
        for width, height in ((10, 9), (11, 8), (3, 9)):
            with self.assertRaises(ValueError):
                maze_layout(0, width=width, height=height)

    def test_maze_corridors_are_one_cell_wide_with_loops(self):
        for seed in range(50):
            grid = parse(maze_layout(seed))
            aisles = set(cells_of(grid, "."))
            # No open 2x2 square anywhere: every corridor is one cell wide.
            for x, y in aisles:
                self.assertFalse({(x + 1, y), (x, y + 1), (x + 1, y + 1)} <= aisles, f"seed {seed}")
            # A tree of n cells has n - 1 links; more links means at least one loop.
            links = sum((x + 1, y) in aisles for x, y in aisles) + sum((x, y + 1) in aisles for x, y in aisles)
            self.assertGreater(links, len(aisles) - 1, f"seed {seed}")

    def test_invalid_maps_are_rejected(self):
        with self.assertRaises(ValueError):
            Warehouse("...\n.X.\nSD.")
        with self.assertRaises(ValueError):
            Warehouse("SSS\nS.S\nSDS")  # one aisle cell, and the shelf corners are unreachable


if __name__ == "__main__":
    unittest.main()
