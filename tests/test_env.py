"""Contract tests for env/: dynamics, observation, rewards, complexity factors, get_state schema.

Both teams depend on these; a change that breaks one is a cross-team change.
Run from the repo root: python -m unittest discover -s tests
"""
import random
import unittest

from env.layout import DEFAULT_MAP, cells_of, maze_layout, parse, problems, random_layout
from env.observation import FEATURES, obs_type
from env.warehouse import (BACKWARD, CONTRACT_VERSION, DOCK_HIT, DOWN, DROPOFF, FORWARD, HUMAN, LEFT,
                           PICKUP, R_DELIVERY, RIGHT, ROBOT, ROBOT_TYPES, TURN_LEFT, TURN_RIGHT, UP,
                           VALUE_FLOOR, WALL, Task, Warehouse)

# Open rows on top so robots can be lined up freely; the bottom row makes the map valid.
OPEN_MAP = """
.....
.....
S...D
"""


def env_with_robots(*placements, layout=OPEN_MAP, alpha=0.0, **kwargs):
    """An env with no task generation and robots placed exactly: placements are (pos, heading)."""
    env = Warehouse(layout, n_robots=len(placements), alpha=alpha, task_prob=0.0, **kwargs)
    env.reset(0)
    for robot, (pos, heading) in zip(env.robots, placements):
        robot.pos, robot.heading = pos, heading
    return env


def give_task(env, kind, robot=0):
    env.queue.append(env._new_task(kind))
    env._assign_idle()
    return env.robots[robot].task


def facing(cell):
    """(aisle cell, heading) that faces a DEFAULT_MAP shelf or dock: row-1 shelves are served from
    row 0 looking down, row-2 shelves from row 3 looking up, docks (row 5) from row 4 looking down."""
    x, y = cell
    return {1: ((x, 0), DOWN), 2: ((x, 3), UP), 5: ((x, 4), DOWN)}[y]


def place(env, i, cell):
    env.robots[i].pos, env.robots[i].heading = facing(cell)


class TestMovement(unittest.TestCase):
    def test_forward_and_backward_follow_the_heading(self):
        env = env_with_robots(((1, 0), RIGHT))
        _, rewards, _, info = env.step([FORWARD])
        self.assertEqual(env.robots[0].pos, (2, 0))
        self.assertAlmostEqual(rewards[0], -0.05)
        env.step([BACKWARD])
        self.assertEqual((env.robots[0].pos, env.robots[0].heading), ((1, 0), RIGHT))

    def test_turns_rotate_in_place_and_are_always_valid(self):
        env = env_with_robots(((0, 0), UP))
        headings = []
        for action in (TURN_RIGHT, TURN_RIGHT, TURN_LEFT, TURN_LEFT, TURN_LEFT):
            _, _, _, info = env.step([action])
            headings.append(env.robots[0].heading)
            self.assertEqual((env.robots[0].pos, info["invalid"]), ((0, 0), 0))
        self.assertEqual(headings, [RIGHT, DOWN, RIGHT, UP, LEFT])

    def test_moves_into_obstacles_are_invalid(self):
        for pos, heading, action in [((0, 0), UP, FORWARD), ((0, 0), RIGHT, BACKWARD),
                                     ((0, 1), DOWN, FORWARD), ((4, 1), UP, BACKWARD)]:
            env = env_with_robots((pos, heading))
            _, rewards, _, info = env.step([action])
            self.assertEqual(env.robots[0].pos, pos)
            self.assertEqual(info["invalid"], 1)
            self.assertAlmostEqual(rewards[0], -0.15)

    def test_swap_is_blocked(self):
        env = env_with_robots(((1, 0), RIGHT), ((2, 0), LEFT))
        _, rewards, _, info = env.step([FORWARD, FORWARD])
        self.assertEqual([r.pos for r in env.robots], [(1, 0), (2, 0)])
        self.assertEqual(info["blocked"], 2)
        self.assertAlmostEqual(rewards[0], -1.05)

    def test_same_cell_contest_has_one_winner(self):
        winners = set()
        for seed in range(20):
            env = env_with_robots(((1, 0), RIGHT), ((3, 0), LEFT))
            env.reset(seed)
            for robot, (pos, heading) in zip(env.robots, [((1, 0), RIGHT), ((3, 0), LEFT)]):
                robot.pos, robot.heading = pos, heading
            _, rewards, _, info = env.step([FORWARD, FORWARD])
            positions = [r.pos for r in env.robots]
            self.assertEqual(info["blocked"], 1)
            winner = positions.index((2, 0))
            self.assertAlmostEqual(rewards[1 - winner], -1.05)
            winners.add(winner)
        self.assertEqual(winners, {0, 1}, "the winner is random, not always the same robot")

    def test_move_into_a_staying_robot_is_blocked_and_only_it_is_penalised(self):
        env = env_with_robots(((1, 0), RIGHT), ((2, 0), UP))
        _, rewards, _, info = env.step([FORWARD, TURN_RIGHT])
        self.assertEqual([r.pos for r in env.robots], [(1, 0), (2, 0)])
        self.assertAlmostEqual(rewards[0], -1.05)
        self.assertAlmostEqual(rewards[1], -0.05)

    def test_follow_the_leader_is_allowed_and_chains_block(self):
        env = env_with_robots(((1, 0), RIGHT), ((2, 0), RIGHT))
        self.assertEqual(env.step([FORWARD, FORWARD])[3]["blocked"], 0)
        self.assertEqual([r.pos for r in env.robots], [(2, 0), (3, 0)])
        env = env_with_robots(((1, 0), RIGHT), ((2, 0), RIGHT), ((3, 0), UP))
        self.assertEqual(env.step([FORWARD, FORWARD, TURN_LEFT])[3]["blocked"], 2)


class TestTasks(unittest.TestCase):
    def test_pickup_or_dropoff_while_idle_is_invalid(self):
        for action in (PICKUP, DROPOFF):
            env = env_with_robots(((0, 0), UP))
            _, rewards, _, info = env.step([action])
            self.assertEqual(info["invalid"], 1)
            self.assertAlmostEqual(rewards[0], -0.15)

    def test_pickup_needs_to_face_the_shelf(self):
        env = env_with_robots(((0, 0), UP), layout=DEFAULT_MAP)
        task = give_task(env, "retrieve")
        place(env, 0, task.pickup)
        env.robots[0].heading = (env.robots[0].heading + 1) % 4     # beside it, but looking away
        self.assertEqual(env.step([PICKUP])[3]["invalid"], 1)
        env.step([TURN_LEFT])
        self.assertEqual(env.step([PICKUP])[3]["invalid"], 0)
        self.assertIsNotNone(env.robots[0].crate)

    def test_retrieve_cycle(self):
        env = env_with_robots(((0, 0), UP), layout=DEFAULT_MAP)
        crates_before = len(env.get_state()["crates"])
        task = give_task(env, "retrieve")
        place(env, 0, task.pickup)
        env.step([PICKUP])
        self.assertEqual(env.step([PICKUP])[3]["invalid"], 1, "already carrying")
        place(env, 0, task.dropoff)
        _, rewards, _, info = env.step([DROPOFF])
        self.assertEqual((info["deliveries"], info["value"]), (1, 1.0))
        self.assertAlmostEqual(rewards[0], 9.95)
        self.assertIsNone(env.robots[0].task)
        self.assertEqual(len(env.get_state()["crates"]), crates_before - 1)

    def test_store_cycle(self):
        env = env_with_robots(((0, 0), UP), layout=DEFAULT_MAP)
        task = give_task(env, "store")
        self.assertIn(task.pickup, env.crate_at, "store crate appears on the dock at assignment")
        place(env, 0, task.pickup)
        env.step([PICKUP])
        place(env, 0, task.dropoff)
        _, rewards, _, info = env.step([DROPOFF])
        self.assertEqual(info["deliveries"], 1)
        self.assertAlmostEqual(rewards[0], 9.95)
        self.assertIn(task.dropoff, env.crate_at)

    def test_nearest_dock_ties_go_to_lowest_index(self):
        env = env_with_robots(((0, 0), UP), layout=DEFAULT_MAP)
        env.crate_at = {(3, 1): 99}
        env.crate_info[99] = (1, 1)
        self.assertEqual(give_task(env, "retrieve").dropoff, (0, 5))

    def test_retrieve_delivery_keeps_a_store_dock_reserved(self):
        env = env_with_robots(((0, 0), UP), ((6, 0), UP), layout=DEFAULT_MAP)
        env.crate_at = {(1, 1): 50, (6, 5): 51}  # a crate on (6, 5) leaves only (0, 5) for a store
        env.crate_info.update({50: (1, 1), 51: (1, 1)})
        retrieve = give_task(env, "retrieve", robot=0)
        store = give_task(env, "store", robot=1)
        self.assertEqual((retrieve.dropoff, store.pickup), ((0, 5), (0, 5)))
        place(env, 0, (1, 1))
        env.step([PICKUP, TURN_RIGHT])
        place(env, 0, (0, 5))
        env.step([DROPOFF, TURN_RIGHT])
        self.assertIn((0, 5), env.reserved, "store task still holds its dock")
        self.assertIn((0, 5), env.crate_at)

    def test_reservations_and_invariants_hold_with_every_factor_on(self):
        for seed in range(5):
            env = Warehouse(DEFAULT_MAP, n_robots=3, task_prob=1.0, mixed_crates=True, mixed_robots=True,
                            priority=True, humans=2)
            env.reset(seed)
            rng = random.Random(seed)
            for _ in range(100):
                env.step([rng.randrange(6) for _ in range(3)])
                tasks = [r.task for r in env.robots if r.task] + list(env.queue)
                shelves = [t.pickup if t.kind == "retrieve" else t.dropoff for t in tasks]
                self.assertEqual(len(shelves), len(set(shelves)))
                cells = [r.pos for r in env.robots] + list(env.humans)
                self.assertEqual(len(cells), len(set(cells)), "robots and humans never share a cell")
                self.assertTrue(all(parse(DEFAULT_MAP)[y][x] == "." for x, y in cells))
                for r in env.robots:
                    if r.task:
                        self.assertLessEqual(r.task.weight, r.kind.max_weight)
                        self.assertLessEqual(r.task.volume, r.kind.max_volume)
                self.assertEqual(set(env.crate_info), {c["id"] for c in env.get_state()["crates"]})


class TestObservation(unittest.TestCase):
    def test_default_fields_are_pinned_to_the_contract_version(self):
        # If this fails you changed the observation: update the README, bump CONTRACT_VERSION
        # in env/warehouse.py, then update this expectation.
        fields = ("x", "y", "heading", "carrying", "has_task", "target_x", "target_y", "front_d", "front_t",
                  "right_d", "right_t", "back_d", "back_t", "left_d", "left_t")
        self.assertEqual((CONTRACT_VERSION, Warehouse().spec.obs_fields), (2, fields))

    def test_obs_is_a_plain_named_tuple_and_idle_target_is_own_cell(self):
        env = env_with_robots(((2, 0), RIGHT))
        obs = env.step([TURN_LEFT])[0][0]
        self.assertIsInstance(obs, tuple)
        self.assertEqual(obs[:7], (2, 0, UP, 0, 0, 2, 0))
        self.assertEqual((obs.target_x, obs.heading), (2, UP))

    def test_rays_are_relative_to_the_heading(self):
        env = env_with_robots(((0, 0), RIGHT), ((3, 0), UP), layout=DEFAULT_MAP)
        obs = env._observe_all()[0]
        self.assertEqual((obs.front_d, obs.front_t), (3, ROBOT))
        self.assertEqual((obs.left_d, obs.left_t), (1, WALL))       # looking right, left is up: map edge
        self.assertEqual((obs.right_d, obs.right_t), (5, DOCK_HIT))  # and right is down: dock at (0, 5)
        self.assertEqual((obs.back_d, obs.back_t), (1, WALL))

    def test_features_are_chosen_by_the_algorithm(self):
        env = Warehouse(obs=("position", "crate", "robot", "local_map"))
        obs = env.reset(0)[0]
        self.assertEqual(len(obs), 2 + 4 + 4 + 25)
        self.assertEqual(env.spec.obs_size, len(obs))
        self.assertEqual(obs.m12, 0, "centre of the local map is the robot's own cell")
        self.assertEqual((obs.ready, obs.period, obs.max_weight), (1, 1, 3))
        for bad in (("nope",), (), ("position", "position")):
            with self.assertRaises(ValueError):
                Warehouse(obs=bad)
        self.assertEqual(set(FEATURES), {"position", "heading", "task", "rays", "crate", "robot", "local_map"})
        self.assertIs(obs_type(("position",)), obs_type(("position",)))

    def test_target_follows_task_stage(self):
        env = env_with_robots(((0, 0), UP), layout=DEFAULT_MAP)
        task = give_task(env, "retrieve")
        obs = env._observe_all()[0]
        self.assertEqual((obs.carrying, obs.has_task, obs.target_x, obs.target_y), (0, 1, *task.pickup))
        env.robots[0].crate = env.crate_at.pop(task.pickup)
        obs = env._observe_all()[0]
        self.assertEqual((obs.carrying, obs.target_x, obs.target_y), (1, *task.dropoff))


class TestFactors(unittest.TestCase):
    def test_everything_is_off_by_default(self):
        env = Warehouse()
        self.assertEqual({r.kind.name for r in env.robots}, {"standard"})
        self.assertEqual(env.humans, ())
        self.assertEqual(set(env.crate_info.values()), {(1, 1)})

    def test_mixed_robots_alternate_and_slow_ones_skip_steps(self):
        env = env_with_robots(((0, 0), RIGHT), ((0, 1), RIGHT), mixed_robots=True)
        self.assertEqual([r.kind for r in env.robots], list(ROBOT_TYPES))
        moved = []
        for _ in range(4):
            env.step([FORWARD, FORWARD])
            moved.append(env.robots[1].pos[0])
        self.assertEqual(env.robots[0].pos, (4, 0))
        self.assertEqual(moved, [0, 1, 1, 2], "the large robot acts every second step")

    def test_dispatcher_only_gives_crates_that_fit(self):
        env = env_with_robots(((0, 0), UP), layout=DEFAULT_MAP, mixed_robots=True)  # one small robot
        heavy = Task("store", (0, 5), (1, 1), weight=3, volume=1)
        env.queue.append(heavy)
        env._assign_idle()
        self.assertIsNone(env.robots[0].task)
        env.queue.append(Task("store", (6, 5), (3, 1), weight=1, volume=2))
        env._assign_idle()
        self.assertEqual(env.robots[0].task.dropoff, (3, 1))

    def test_priority_multiplies_and_decays(self):
        env = env_with_robots(((0, 0), UP), layout=DEFAULT_MAP, priority=True)
        env.t = 50
        self.assertAlmostEqual(env._value(Task("store", (0, 5), (1, 1), priority=3, created=0)), 1.5)
        self.assertAlmostEqual(env._value(Task("store", (0, 5), (1, 1), priority=2, created=-500)),
                               2 * VALUE_FLOOR)
        env.t = 0
        env.queue += [Task("store", (0, 5), (1, 1), priority=1), Task("store", (6, 5), (3, 1), priority=3)]
        env._assign_idle()
        self.assertEqual(env.robots[0].task.priority, 3, "urgent jobs are handed out first")
        place(env, 0, (6, 5))
        env.step([PICKUP])
        place(env, 0, (3, 1))
        _, rewards, _, info = env.step([DROPOFF])
        self.assertAlmostEqual(rewards[0], -0.05 + R_DELIVERY * info["value"])
        self.assertGreater(info["value"], 2.9)

    def test_humans_block_and_are_seen(self):
        env = env_with_robots(((0, 0), RIGHT), humans=1)
        env.humans = ((1, 0),)
        obs = env._observe_all()[0]
        self.assertEqual((obs.front_d, obs.front_t), (1, HUMAN))
        _, rewards, _, info = env.step([FORWARD])
        self.assertEqual((env.robots[0].pos, info["blocked"]), ((0, 0), 1))
        self.assertAlmostEqual(rewards[0], -1.05)

    def test_humans_walk(self):
        env = Warehouse(humans=3)
        env.reset(0)
        start = env.humans
        for _ in range(10):
            env.step([TURN_RIGHT, TURN_RIGHT])
        self.assertEqual(len(env.humans), 3)
        self.assertNotEqual(env.humans, start)

    def test_too_many_humans_is_rejected(self):
        with self.assertRaises(ValueError):
            Warehouse(OPEN_MAP, n_robots=2, humans=10)

    def test_reset_can_switch_the_map(self):
        env = Warehouse(DEFAULT_MAP)
        env.reset(0, maze_layout(3))
        self.assertEqual((env.width, env.height), (11, 9))
        self.assertEqual(env.spec.grid, parse(maze_layout(3)))


class TestRewardMixing(unittest.TestCase):
    def test_alpha_zero_is_individual(self):
        env = env_with_robots(((1, 0), UP), ((4, 0), UP), alpha=0.0)
        _, rewards, _, _ = env.step([FORWARD, TURN_RIGHT])  # robot 0 bumps the edge
        self.assertAlmostEqual(rewards[0], -0.15)
        self.assertAlmostEqual(rewards[1], -0.05)

    def test_alpha_one_is_shared(self):
        env = env_with_robots(((1, 0), UP), ((4, 0), UP), alpha=1.0)
        _, rewards, _, _ = env.step([FORWARD, TURN_RIGHT])
        self.assertAlmostEqual(rewards[0], rewards[1])
        self.assertAlmostEqual(rewards[0], -0.10)


class TestEpisodeAndApi(unittest.TestCase):
    def test_seed_is_deterministic_with_every_factor_on(self):
        def rollout():
            env = Warehouse(DEFAULT_MAP, n_robots=2, mixed_crates=True, mixed_robots=True, priority=True, humans=2)
            obs = [env.reset(42)]
            rng = random.Random(1)
            for _ in range(100):
                obs.append(env.step([rng.randrange(6), rng.randrange(6)]))
            return obs, env.get_state()
        self.assertEqual(rollout(), rollout())

    def test_truncation_at_max_steps(self):
        env = Warehouse(DEFAULT_MAP, max_steps=3)
        env.reset(0)
        for _ in range(2):
            self.assertFalse(env.step([TURN_RIGHT, TURN_RIGHT])[2])
        _, _, done, info = env.step([TURN_RIGHT, TURN_RIGHT])
        self.assertTrue(done and info["truncated"])
        with self.assertRaises(RuntimeError):
            env.step([TURN_RIGHT, TURN_RIGHT])

    def test_bad_actions_are_rejected(self):
        env = Warehouse(DEFAULT_MAP)
        env.reset(0)
        for bad in ([TURN_RIGHT], [TURN_RIGHT, 6], [2.0, TURN_RIGHT]):
            with self.assertRaises(ValueError):
                env.step(bad)

    def test_get_state_schema_and_isolation(self):
        env = Warehouse(DEFAULT_MAP, humans=1)
        env.reset(0)
        state = env.get_state()
        self.assertEqual(set(state), {"width", "height", "grid", "robots", "crates", "humans", "queue_len", "step"})
        self.assertEqual((state["width"], state["height"]), (7, 6))
        self.assertEqual(len(state["crates"]), 4)
        self.assertEqual(set(state["robots"][0]),
                         {"x", "y", "heading", "carrying", "kind", "target", "priority"})
        self.assertEqual(set(state["crates"][0]), {"id", "x", "y", "carried_by", "weight", "volume"})
        self.assertEqual(set(state["humans"][0]), {"x", "y"})
        state["robots"][0]["x"] = 99
        state["grid"][0] = "XXXXXXX"
        self.assertNotEqual(env.get_state()["robots"][0]["x"], 99)
        self.assertEqual(env.get_state()["grid"][0], ".......")

    def test_render_mentions_every_robot_and_human(self):
        env = Warehouse(DEFAULT_MAP, humans=1)
        env.reset(0)
        text = env.render()
        self.assertIn("robot 0", text)
        self.assertIn("robot 1", text)
        self.assertIn("H", "".join(text.splitlines()[:6]))     # the map rows


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
