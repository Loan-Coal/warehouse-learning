"""The two things the env knows but never tells a policy: what is legal, and how far away.

Neither is in the observation and both are needed by any algorithm that plays this env well, so
they live here once. tests/test_nav.py pins them against the env's own rules, both directions.
"""
from array import array
from collections import deque

from env.layout import DIRS, access_cells, is_walkable
from env.warehouse import (BACKWARD, DROPOFF, FORWARD, HEADINGS, N_ACTIONS, PICKUP, TURN_LEFT,
                           TURN_RIGHT)

UNREACHABLE = -1

FIELD_CACHE = 20_000        # ~1.8 KB a field at 21x21, so ~36 MB: a rolling window over maps
_FIELDS = {}


def legal_actions(grid, o):
    """Which of the 6 actions the env would actually carry out, for one robot.

    PICKUP/DROPOFF need the robot to FACE its target, and obs.target is already the pickup cell
    while empty and the dropoff while carrying, so one comparison covers both. A turn is never
    invalid, so some action is always legal. Stepping into a human is legal too: R_BLOCKED, not
    R_INVALID.
    """
    dx, dy = HEADINGS[o.heading]
    front, back, target = (o.x + dx, o.y + dy), (o.x - dx, o.y - dy), (o.target_x, o.target_y)
    legal = [False] * N_ACTIONS
    legal[PICKUP] = bool(o.has_task) and not o.carrying and front == target
    legal[DROPOFF] = bool(o.carrying) and front == target
    legal[TURN_RIGHT] = legal[TURN_LEFT] = True
    legal[FORWARD] = is_walkable(grid, *front)
    legal[BACKWARD] = is_walkable(grid, *back)
    return legal


class Field:
    """Steps from every cell of one map to one target, indexed by y * width + x.

    17x less memory than the dict it replaces and 1.6x faster to build, though at() is ~3.8x
    slower than dict.get. Building dominates. Never index .steps directly: a neighbour can be off
    the map, and the negative index would wrap into the opposite row (tests/test_nav.py).
    """
    __slots__ = ("width", "height", "steps")

    def __init__(self, width, height, steps):
        self.width, self.height, self.steps = width, height, steps

    def at(self, x, y):
        """Steps from (x, y), or UNREACHABLE off the map, into a rack, or from a sealed pocket."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.steps[y * self.width + x]
        return UNREACHABLE


def distance_field(grid, cell):
    """Shortest walkable distance to `cell` from anywhere, built once per (map, target).

    Navigation is solved here, not learned: path length is an iterative graph computation an MLP
    cannot represent. The network is left with the coordination, which is map-agnostic and so can
    generalise to mazes it never saw (LEARNINGS.md).
    """
    key = (grid, cell)
    if key not in _FIELDS:
        if len(_FIELDS) >= FIELD_CACHE:
            _FIELDS.clear()     # cheaper than per-hit LRU bookkeeping
        _FIELDS[key] = _build(grid, access_cells(grid, cell))
    return _FIELDS[key]


def _build(grid, goals):
    width, height = len(grid[0]), len(grid)
    steps = array("i", [UNREACHABLE]) * (width * height)
    frontier = deque()
    for x, y in goals:
        steps[y * width + x] = 0
        frontier.append((x, y))
    while frontier:
        x, y = frontier.popleft()
        here = steps[y * width + x]
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if (0 <= nx < width and 0 <= ny < height
                    and steps[ny * width + nx] == UNREACHABLE and is_walkable(grid, nx, ny)):
                steps[ny * width + nx] = here + 1
                frontier.append((nx, ny))
    return Field(width, height, steps)
