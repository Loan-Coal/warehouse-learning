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
    def test_turns_are_always_legal_and_there_is_no_wait(self):
        env = maze_env(0)
        for o in env.reset(0):
            legal = legal_actions(env.grid, o)
            self.assertEqual((len(legal), legal[TURN_RIGHT], legal[TURN_LEFT]), (6, True, True))

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


class TestCache(unittest.TestCase):
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
