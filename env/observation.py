"""What a robot can observe, as named feature groups an algorithm picks from.

An algorithm lists the groups it wants, e.g. OBS = ("position", "heading", "task", "rays").
The env then returns, per robot, a flat tuple of ints with named fields in that order.
Add a group here (fields + function) and every algorithm can use it.
"""
from collections import namedtuple
from functools import lru_cache

from env.layout import DOCK, SHELF, in_bounds

# Headings, clockwise: index = heading, value = (dx, dy) of "forward" (y grows downward).
HEADINGS = ((0, -1), (1, 0), (0, 1), (-1, 0))
UP, RIGHT, DOWN, LEFT = range(4)

# What a ray or a local-map cell sees.
NOTHING, WALL, EMPTY_SHELF, FULL_SHELF, DOCK_HIT, ROBOT, HUMAN = range(7)

LOCAL_MAP_RADIUS = 2    # local_map covers a (2r+1) x (2r+1) square around the robot

FEATURES = {
    # own cell (x right, y down)
    "position": ("x", "y"),
    # 0 up, 1 right, 2 down, 3 left
    "heading": ("heading",),
    # target = pickup cell if not carrying, else dropoff cell; own cell if idle
    "task": ("carrying", "has_task", "target_x", "target_y"),
    # per direction RELATIVE to the heading: distance to the first thing seen (0 if nothing) and its type
    "rays": ("front_d", "front_t", "right_d", "right_t", "back_d", "back_t", "left_d", "left_t"),
    # the current job's crate and priority; task_age = steps since the job was created; all 0 if idle
    "crate": ("crate_weight", "crate_volume", "priority", "task_age"),
    # ready = 1 if this robot's action counts next step (slow robots act every `period` steps)
    "robot": ("ready", "period", "max_weight", "max_volume"),
    # cell types around the robot, row by row, map-aligned (not rotated); m12 is the robot itself
    "local_map": tuple(f"m{k}" for k in range((2 * LOCAL_MAP_RADIUS + 1) ** 2)),
}

DEFAULT_OBS = ("position", "heading", "task", "rays")


@lru_cache(maxsize=None)
def obs_type(features):
    """The namedtuple class for a feature list. Raises ValueError on an unknown or repeated name."""
    features = tuple(features)
    unknown = [f for f in features if f not in FEATURES]
    if unknown or not features or len(set(features)) != len(features):
        raise ValueError(f"OBS must list distinct names from {', '.join(FEATURES)}; got {features}")
    return namedtuple("Obs", [field for f in features for field in FEATURES[f]])


def observe(env, i, features):
    """Robot i's observation for this feature list."""
    values = [v for f in features for v in _COMPUTE[f](env, i)]
    return obs_type(tuple(features))(*values)


def cell_type(env, x, y, robot_cells):
    if not in_bounds(env.grid, x, y):
        return WALL
    if (x, y) in robot_cells:
        return ROBOT
    if (x, y) in env.humans:
        return HUMAN
    c = env.grid[y][x]
    if c == SHELF:
        return FULL_SHELF if (x, y) in env.crate_at else EMPTY_SHELF
    if c == DOCK:
        return DOCK_HIT
    return NOTHING


def rays(env, i):
    """(distance, type) of the first thing seen front, right, back, left; (0, 0) if nothing in range."""
    robot = env.robots[i]
    others = {r.pos for r in env.robots if r is not robot}
    hits = []
    for turn in range(4):
        dx, dy = HEADINGS[(robot.heading + turn) % 4]
        hit = (0, NOTHING)
        for d in range(1, env.view_range + 1):
            kind = cell_type(env, robot.pos[0] + dx * d, robot.pos[1] + dy * d, others)
            if kind != NOTHING:
                hit = (d, kind)
                break
        hits.append(hit)
    return hits


def _task(env, i):
    r = env.robots[i]
    target = r.pos if r.task is None else (r.task.dropoff if r.crate is not None else r.task.pickup)
    return int(r.crate is not None), int(r.task is not None), target[0], target[1]


def _crate(env, i):
    t = env.robots[i].task
    return (0, 0, 0, 0) if t is None else (t.weight, t.volume, t.priority, env.t - t.created)


def _robot(env, i):
    kind = env.robots[i].kind
    return int(env.ready(i)), kind.period, kind.max_weight, kind.max_volume


def _local_map(env, i):
    x0, y0 = env.robots[i].pos
    others = {r.pos for k, r in enumerate(env.robots) if k != i}
    span = range(-LOCAL_MAP_RADIUS, LOCAL_MAP_RADIUS + 1)
    return [cell_type(env, x0 + dx, y0 + dy, others) for dy in span for dx in span]


_COMPUTE = {
    "position": lambda env, i: env.robots[i].pos,
    "heading": lambda env, i: (env.robots[i].heading,),
    "task": _task,
    "rays": lambda env, i: [v for hit in rays(env, i) for v in hit],
    "crate": _crate,
    "robot": _robot,
    "local_map": _local_map,
}
