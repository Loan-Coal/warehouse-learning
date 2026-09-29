"""The two things the env knows but never tells a policy: what is legal, and how far away.

Both are needed by any algorithm that plays this env well, and neither is in the observation, so
they live here once instead of being rediscovered inside each one. tests/test_nav.py pins them
against the env's own rules, in both directions.
"""
from array import array
from collections import deque

from env.layout import DIRS, access_cells, is_walkable
from env.warehouse import (BACKWARD, DROPOFF, FORWARD, HEADINGS, N_ACTIONS, PICKUP, TURN_LEFT,
                           TURN_RIGHT)

UNREACHABLE = -1

# Distance fields kept across all maps and targets. One is ~1.8 KB at 21x21, so 20k is ~36 MB and
# covers every target of ~115 mazes. With --maze-pool bounding the maps in play the working set
# fits and nothing is ever evicted; without it the cache is a rolling window over recent maps.
FIELD_CACHE = 20_000
_FIELDS = {}


def legal_actions(grid, o):
    """Which of the 6 actions the env would actually carry out, for one robot.

    The env never tells a policy this: it silently ignores an impossible action and fines it
    R_INVALID, reporting only a count. PICKUP/DROPOFF need the robot to FACE its target cell, not
    merely stand beside it (env/warehouse.py _pickup/_dropoff); obs.target_x/target_y is already
    the pickup cell while empty and the dropoff cell while carrying, so one comparison covers
    both. Turning is never invalid, so at least one action is always legal.

    Stepping into a human IS legal here: the env charges R_BLOCKED for that, not R_INVALID.
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
    """Steps from every cell of one map to one target, as a flat array indexed by y * width + x.

    Measured against the dict of (x, y) -> steps it replaces, on a 21x21 maze: 1.8 KB instead of
    30 KB per field (17x), and 1.6x faster to build, but individual lookups are ~3.8x SLOWER,
    because at() is a Python call where dict.get is C. Building dominates here -- a new map every
    episode rebuilds ~60 fields, which costs far more than the ~50 reads per env step -- and the
    memory is what lets one cache span many maps at all. Do not index .steps directly: a
    neighbour of a walkable cell can be out of bounds and a negative index would wrap silently.
    """
    __slots__ = ("width", "height", "steps")

    def __init__(self, width, height, steps):
        self.width, self.height, self.steps = width, height, steps

    def at(self, x, y):
        """Steps from (x, y) to the target, or UNREACHABLE for a wall, a rack or a sealed pocket."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.steps[y * self.width + x]
        return UNREACHABLE


def distance_field(grid, cell):
    """Shortest walkable distance to `cell` from anywhere, built once per (map, target).

    An MLP cannot work out shortest paths through a maze from coordinates and rays -- path length
    is an iterative graph computation, and with a new maze every episode there is nothing to
    memorise either. So navigation is solved analytically here and handed to the network as a few
    numbers; what the network has to learn is the coordination, which is map-agnostic.
    """
    key = (grid, cell)
    field = _FIELDS.get(key)
    if field is None:
        if len(_FIELDS) >= FIELD_CACHE:
            # Cheaper than per-hit LRU bookkeeping, and a bounded --maze-pool never gets here.
            _FIELDS.clear()
        field = _FIELDS[key] = _build(grid, access_cells(grid, cell))
        return field
    return field


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
            if 0 <= nx < width and 0 <= ny < height and steps[ny * width + nx] == UNREACHABLE                     and is_walkable(grid, nx, ny):
                steps[ny * width + nx] = here + 1
                frontier.append((nx, ny))
    return Field(width, height, steps)
