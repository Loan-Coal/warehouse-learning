"""What an algorithm shows its network. LEARNINGS.md says why each number earns its place.

Two properties worth not breaking:
  * nothing is normalised by a running statistic, so there is no state to checkpoint;
  * the row is the same width for 4 robots on 11x9 as for 6 on 21x21, which is what lets one
    checkpoint play a scenario it never trained on.
"""
import os

import torch

from env.observation import HEADINGS, obs_type
from env.warehouse import HUMAN, ROBOT, VALUE_DECAY_STEPS, VALUE_FLOOR
from rl.nav import UNREACHABLE, distance_field, distance_from

# VIEW mirrors Warehouse(view_range=5), which EnvSpec does not carry: another value would push
# dist / VIEW quietly past 1.0.
VIEW, RAY_TYPES = 5, HUMAN + 1          # ray range, and the ray types (0 = nothing .. 6 = human)
PRIORITY_MAX = 3                        # env jobs draw a priority in 1..3 (env/warehouse.py)

PAIR_WIDTH = 9      # pair_row() below; the pairwise encoder's extra input per (i, j)

torch.set_num_threads(min(8, os.cpu_count() or 1))      # measured best; see LEARNINGS.md


def one_hot(k, n):
    return [float(i == k) for i in range(n)]


class Features:
    """Observations to network rows, for whichever map is currently installed.

    OBS is the observation groups the rows need; a policy re-exports it so rl/cli.py builds a
    matching env. Call use(grid) for a new map: distances and the potential are relative to it.
    """

    OBS = ("position", "heading", "task", "rays", "crate", "robot")

    def __init__(self, obs_features=OBS, use_bfs=True):
        self.obs_features = tuple(obs_features)
        self.use_bfs = use_bfs
        self.grid, self.w, self.h = (), 1, 1
        # A blank observation has no task, so the shortest-path branch returns its constant and
        # the still-unset grid is never touched. The width is measured, not asserted.
        blank = obs_type(self.obs_features)
        self.width = len(self.row(blank(*([0] * len(blank._fields)))))

    def use(self, grid):
        self.grid, self.w, self.h = grid, len(grid[0]), len(grid)

    def batch(self, obs_per_env):
        """[envs, robots, width], the shape the networks want. Dominated by the Python loop in row()."""
        return torch.tensor([[self.row(o) for o in one] for one in obs_per_env],
                            dtype=torch.float32)

    def pairs(self, obs_per_env):
        """[envs, robots, robots, PAIR_WIDTH]: what robot j means to robot i, per pair. The
        diagonal is zeros. The pairwise encoder (rl/policies/ppo.py) reads [row_i, row_j,
        pair_ij]; these are the relations a function of two separate rows cannot form in one
        layer: graph distance, "j is on my path", "we want the same cell next"."""
        out = []
        for one in obs_per_env:
            n = len(one)
            rows = [[[0.0] * PAIR_WIDTH for _ in range(n)] for _ in range(n)]
            nxt = [self.best_next(o) for o in one]
            for i, oi in enumerate(one):
                for j, oj in enumerate(one):
                    if i != j:
                        rows[i][j] = self.pair_row(oi, oj, nxt[i], nxt[j])
            out.append(rows)
        return torch.tensor(out, dtype=torch.float32)

    def best_next(self, o):
        """The neighbour cell that steps closer to the target (first in HEADINGS order), or None."""
        if not o.has_task:
            return None
        field = distance_field(self.grid, (o.target_x, o.target_y))
        here = field.at(o.x, o.y)
        if here == UNREACHABLE or here == 0:
            return None
        for dx, dy in HEADINGS:
            if field.at(o.x + dx, o.y + dy) == here - 1:
                return (o.x + dx, o.y + dy)
        return None

    def pair_row(self, oi, oj, next_i, next_j):
        d = distance_from(self.grid, (oj.x, oj.y)).at(oi.x, oi.y)
        dist = 1.0 if d == UNREACHABLE else min(d / (self.w + self.h), 1.0)
        on_path = 0.0
        if oi.has_task and d != UNREACHABLE:
            field = distance_field(self.grid, (oi.target_x, oi.target_y))
            here, there = field.at(oi.x, oi.y), field.at(oj.x, oj.y)
            on_path = float(here != UNREACHABLE and there != UNREACHABLE and there + d == here)
        same_next = float(next_i is not None and next_i == next_j)
        toward_me = float(next_j == (oi.x, oi.y))
        toward_them = float(next_i == (oj.x, oj.y))
        return [dist, on_path, same_next, toward_me, toward_them] + one_hot((oj.heading - oi.heading) % 4, 4)

    def row(self, o):
        """One robot's input. Every feature here is one that some action can change.

        Crate size, capacity, raw task age, Manhattan distance and a per-ray "nothing in range"
        flag were dropped as unactionable or redundant; tests/test_features.py says why not.
        """
        dx, dy = o.target_x - o.x, o.target_y - o.y
        v = [o.x / self.w, o.y / self.h, dx / self.w, dy / self.h,
             float(o.carrying), float(o.has_task), float(faces_target(o)),
             o.priority / PRIORITY_MAX, task_value(o) / PRIORITY_MAX,
             float(o.ready), float(max(o.period - 1, 0))]
        v += one_hot(o.heading, 4)
        for dist, kind in ((o.front_d, o.front_t), (o.right_d, o.right_t),
                           (o.back_d, o.back_t), (o.left_d, o.left_t)):
            # R_BLOCKED is the biggest drain on the score and the one thing the distance field
            # cannot warn about, so "this type AND adjacent" gets its own feature (LEARNINGS.md).
            occupied = float(dist == 1 and kind in (ROBOT, HUMAN))
            v += [dist / VIEW, occupied] + one_hot(kind, RAY_TYPES)
        if self.use_bfs:
            near, deltas = self.distances(o)
            v += [near] + deltas
        return v

    def distances(self, o):
        """How far the target is, and which of front/right/back/left steps closer to it.

        Always the same number of values, so the row keeps its width whatever the state.
        """
        if not o.has_task:
            return 1.0, [0.0] * 4
        field = distance_field(self.grid, (o.target_x, o.target_y))
        here = field.at(o.x, o.y)
        if here == UNREACHABLE:
            return 1.0, [0.0] * 4
        deltas = []
        for turn in range(4):
            dx, dy = HEADINGS[(o.heading + turn) % 4]
            there = field.at(o.x + dx, o.y + dy)
            deltas.append(0.0 if there == UNREACHABLE else float(here - there))  # +1 = a step closer
        return min(here / (self.w + self.h), 1.0), deltas

    def potential(self, o):
        """Phi(s): nearer the target is better, facing it better still. A function of state
        alone, which makes shaping on it potential-based (though not policy-invariant here).
        """
        if not o.has_task:
            return 0.0
        here = distance_field(self.grid, (o.target_x, o.target_y)).at(o.x, o.y)
        return 0.0 if here == UNREACHABLE else -float(here) + 0.5 * float(faces_target(o))


def task_value(o):
    """What a delivery would pay right now (env/warehouse.py _value): stationary, unlike age."""
    if not o.has_task:
        return 0.0
    return o.priority * max(VALUE_FLOOR, 1 - o.task_age / VALUE_DECAY_STEPS)


def faces_target(o):
    dx, dy = HEADINGS[o.heading]
    return bool(o.has_task) and (o.x + dx, o.y + dy) == (o.target_x, o.target_y)


def target_changed(o, o2):
    """True when a pickup or a delivery swapped the target, so the potential jumps."""
    return ((o.has_task, o.carrying, o.target_x, o.target_y)
            != (o2.has_task, o2.carrying, o2.target_x, o2.target_y))
