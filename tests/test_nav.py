"""rl/nav.py: what the env knows and never tells a policy.

Two things, and both are pinned against the env itself:

- which actions it would actually carry out, since an impossible one is silently ignored and
  fined rather than refused. Checked in both directions, so neither a mask that permits too much
  nor one that forbids everything can pass.
- how far a target is, checked cell by cell against a plain BFS written out longhand here.
"""
import random
import unittest
from collections import deque

from env.layout import DEFAULT_MAP, access_cells, cells_of, is_walkable, maze_layout, parse
from env.observation import HEADINGS
from env.warehouse import N_ACTIONS, TURN_LEFT, TURN_RIGHT, Warehouse
from rl import nav
from rl.features import Features
from rl.nav import UNREACHABLE, distance_field, legal_actions

ROBOTS, STEPS = 4, 60
FACTORS = dict(mixed_crates=True, mixed_robots=True, priority=True, humans=2)


def maze_env(seed=0, size=(11, 9)):
    return Warehouse(maze_layout(seed, *size, n_robots=ROBOTS), n_robots=ROBOTS, max_steps=STEPS,
                     obs=Features.OBS, **FACTORS)


class TestLegalActions(unittest.TestCase):
    def test_only_legal_actions_are_ever_invalid(self):
        """Playing only actions the mask allows must never earn a single invalid penalty."""
        for seed in range(4):
            with self.subTest(seed=seed):
                env = maze_env(seed)
                obs = env.reset(seed)
                rng = random.Random(seed)
                for _ in range(STEPS):
                    actions = [rng.choice([a for a in range(N_ACTIONS) if legal_actions(env.grid, o)[a]])
                               for o in obs]
                    obs, _, done, info = env.step(actions)
                    self.assertEqual(info["invalid"], 0, "the mask allowed an action the env rejected")
                    if done:
                        break

    def test_some_action_is_always_legal(self):
        """Turning is never invalid, so the mask can never leave a robot with nothing to do."""
        env = maze_env()
        obs = env.reset(0)
        rng = random.Random(0)
        for _ in range(STEPS):
            for o in obs:
                legal = legal_actions(env.grid, o)
                self.assertTrue(any(legal))
                self.assertTrue(legal[TURN_RIGHT] and legal[TURN_LEFT])
            obs, _, done, _ = env.step([rng.randrange(N_ACTIONS) for _ in obs])
            if done:
                break

    def test_what_the_mask_forbids_the_env_rejects(self):
        """The other direction, so a mask that just forbade everything would not pass.

        One ready robot plays a forbidden action; the rest turn in place, which is always
        accepted. The env must then report exactly that one invalid action.
        """
        env = maze_env()
        obs = env.reset(0)
        checked = 0
        for _ in range(STEPS):
            actions, victim = [TURN_RIGHT] * len(obs), None
            for i, o in enumerate(obs):
                illegal = [a for a in range(N_ACTIONS) if not legal_actions(env.grid, o)[a]]
                if victim is None and illegal and env.ready(i):
                    victim, actions[i] = i, illegal[0]
            obs, _, done, info = env.step(actions)
            if victim is not None:
                self.assertEqual(info["invalid"], 1, "the env accepted an action the mask forbade")
                checked += 1
            if done:
                break
        self.assertGreater(checked, 0, "the episode never offered an illegal action to test")


def reference_field(grid, target):
    """A dict BFS from the target's access cells, written out longhand as the thing to match."""
    dist = {cell: 0 for cell in access_cells(grid, target)}
    frontier = deque(dist)
    while frontier:
        x, y = frontier.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxt = (x + dx, y + dy)
            if nxt not in dist and is_walkable(grid, *nxt):
                dist[nxt] = dist[(x, y)] + 1
                frontier.append(nxt)
    return dist


class TestField(unittest.TestCase):
    def test_it_agrees_with_a_plain_bfs_on_every_cell(self):
        grid = parse(maze_layout(3, 21, 21, n_robots=4))
        width, height = len(grid[0]), len(grid)
        for target in (cells_of(grid, "S")[::37] + cells_of(grid, "D")):
            with self.subTest(target=target):
                field, want = distance_field(grid, target), reference_field(grid, target)
                for y in range(height):
                    for x in range(width):
                        self.assertEqual(field.at(x, y), want.get((x, y), UNREACHABLE))

    def test_zero_at_the_access_cells_not_at_the_target(self):
        # Robots act on a cell they stand NEXT to and face, so distance 0 means "in position".
        grid = parse(DEFAULT_MAP)
        shelf = cells_of(grid, "S")[0]
        field = distance_field(grid, shelf)
        self.assertEqual(field.at(*shelf), UNREACHABLE, "the rack itself is not walkable")
        for cell in access_cells(grid, shelf):
            self.assertEqual(field.at(*cell), 0)

    def test_unwalkable_and_out_of_bounds_are_unreachable(self):
        grid = parse(DEFAULT_MAP)
        field = distance_field(grid, cells_of(grid, "D")[0])
        width, height = len(grid[0]), len(grid)
        for x, y in ((-1, 0), (0, -1), (width, 0), (0, height), (-1, -1), (width, height)):
            self.assertEqual(field.at(x, y), UNREACHABLE, f"({x}, {y}) is off the map")
        for shelf in cells_of(grid, "S"):
            self.assertEqual(field.at(*shelf), UNREACHABLE)

    def test_a_negative_index_cannot_wrap_into_a_wrong_answer(self):
        """Why at() bounds-checks instead of letting callers index .steps directly.

        Row -1 wraps to the last row of the array and would report a plausible distance for a
        cell that is off the map entirely. DEFAULT_MAP has walkable cells in its bottom row, so
        the wrapped read returns a real number rather than UNREACHABLE.
        """
        grid = parse(DEFAULT_MAP)
        field = distance_field(grid, cells_of(grid, "S")[0])
        bottom = [x for x in range(field.width)
                  if field.steps[(field.height - 1) * field.width + x] != UNREACHABLE]
        self.assertTrue(bottom, "this map needs a reachable bottom row for the test to mean anything")
        x = bottom[0]
        self.assertEqual(field.at(x, -1), UNREACHABLE, "off the map, and at() says so")
        self.assertNotEqual(field.steps[-1 * field.width + x], UNREACHABLE,
                            "the raw index wrapped to the last row: the answer at() protects against")
        self.assertEqual(field.at(-1, 2), UNREACHABLE, "and the same for a negative column")

    def test_one_step_in_a_decreasing_direction_costs_exactly_one(self):
        """The property the network's delta features rely on: +1 really means a step closer."""
        grid = parse(maze_layout(2, 15, 13, n_robots=3))
        field = distance_field(grid, cells_of(grid, "S")[10])
        checked = 0
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                here = field.at(x, y)
                if here in (UNREACHABLE, 0):
                    continue
                closer = [field.at(x + dx, y + dy) for dx, dy in HEADINGS]
                self.assertIn(here - 1, closer, f"no neighbour of ({x}, {y}) is one step nearer")
                for there in closer:
                    if there != UNREACHABLE:
                        self.assertLessEqual(abs(here - there), 1, "distance jumped by more than 1")
                checked += 1
        self.assertGreater(checked, 50)


class TestCache(unittest.TestCase):
    def test_the_same_map_and_target_give_back_the_same_field(self):
        grid = parse(maze_layout(4, 11, 9, n_robots=2))
        dock = cells_of(grid, "D")[0]
        self.assertIs(distance_field(grid, dock), distance_field(grid, dock))

    def test_a_different_map_gets_its_own_field(self):
        one = parse(maze_layout(5, 11, 9, n_robots=2))
        two = parse(maze_layout(6, 11, 9, n_robots=2))
        a, b = distance_field(one, cells_of(one, "D")[0]), distance_field(two, cells_of(two, "D")[0])
        self.assertIsNot(a, b)
        # An equal grid rebuilt from the same seed is the same key, so the field is reused.
        self.assertIs(a, distance_field(parse(maze_layout(5, 11, 9, n_robots=2)),
                                        cells_of(one, "D")[0]))

    def test_the_cache_is_bounded(self):
        grid = parse(maze_layout(8, 11, 9, n_robots=2))
        kept, nav.FIELD_CACHE = nav.FIELD_CACHE, 4
        try:
            nav._FIELDS.clear()
            for target in (cells_of(grid, "S") + cells_of(grid, "D"))[:12]:
                distance_field(grid, target)
                self.assertLessEqual(len(nav._FIELDS), 4)
        finally:
            nav.FIELD_CACHE = kept
            nav._FIELDS.clear()

if __name__ == "__main__":
    unittest.main()
