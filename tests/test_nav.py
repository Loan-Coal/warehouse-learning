"""rl/nav.py: the shortest-path fields every policy here navigates by.

Action legality is pinned against the env's own rules in tests/test_central_dqn.py, which drives
nav.legal_actions both ways. This file is about the distance fields: that they agree with a
plain BFS written independently here, that they say UNREACHABLE rather than lying, and that the
cache hands back the same field for the same map.
"""
import unittest
from collections import deque

from env.layout import DEFAULT_MAP, access_cells, cells_of, is_walkable, maze_layout, parse
from env.observation import HEADINGS
from rl import nav
from rl.nav import UNREACHABLE, Field, distance_field


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
        for seed, size in ((0, (11, 9)), (3, (21, 21)), (7, (15, 13))):
            grid = parse(maze_layout(seed, *size, n_robots=4))
            width, height = len(grid[0]), len(grid)
            for target in (cells_of(grid, "S")[::37] + cells_of(grid, "D")):
                with self.subTest(seed=seed, size=size, target=target):
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

    def test_every_walkable_cell_reaches_a_dock(self):
        # maze_layout validates that the aisles are connected, so a dock is reachable from all.
        for seed in range(5):
            grid = parse(maze_layout(seed, 11, 9, n_robots=2))
            field = distance_field(grid, cells_of(grid, "D")[0])
            for y in range(len(grid)):
                for x in range(len(grid[0])):
                    if is_walkable(grid, x, y):
                        self.assertNotEqual(field.at(x, y), UNREACHABLE, f"({x}, {y}) is walled in")

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

    def test_a_field_is_a_field(self):
        grid = parse(DEFAULT_MAP)
        field = distance_field(grid, cells_of(grid, "D")[0])
        self.assertIsInstance(field, Field)
        self.assertEqual((field.width, field.height), (len(grid[0]), len(grid)))
        self.assertEqual(len(field.steps), field.width * field.height)


if __name__ == "__main__":
    unittest.main()
