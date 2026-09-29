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

# Distance fields kept across all maps and targets. One is ~1.8 KB at 21x21, so 20k is ~36 MB and
# covers every target of ~115 mazes. With --maze-pool bounding the maps in play the working set
# fits and nothing is ever evicted; without it the cache is a rolling window over recent maps.
FIELD_CACHE = 20_000
_FIELDS = {}


def legal_actions(grid, o):
    """Which of the 6 actions the env would actually carry out, for one robot.

    PICKUP/DROPOFF need the robot to FACE its target, not merely stand beside it; obs.target is
    already the pickup cell while empty and the dropoff while carrying, so one comparison covers
    both. Turning is never invalid, so at least one action is always legal. Stepping into a human
    IS legal -- the env charges R_BLOCKED for that, not R_INVALID.
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

    Against the dict it replaces, on a 21x21 maze: 17x less memory and 1.6x faster to build, but
    lookups are ~3.8x slower since at() is a Python call where dict.get is C. Building dominates.
    Do not index .steps directly -- a neighbour of a walkable cell can be off the map, and a
    negative index would wrap into the opposite row and return a plausible wrong distance.
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

    Navigation is solved analytically because an MLP cannot do it: path length is an iterative
    graph computation. What the network then has to learn is the coordination, which is
    map-agnostic -- and that is what makes generalising to unseen mazes possible (LEARNINGS.md).
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
