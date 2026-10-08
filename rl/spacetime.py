"""Space-time A* for ONE robot against a table of what the others have booked (WHCA*-style).

The planner (rl/policies/planner.py) plans robots one after another, each treating the paths of
those before it as moving obstacles; the table is what carries them. Time is part of the state,
so waiting, following and passing in a corridor all fall out of one search.
"""
import heapq
from itertools import count
from typing import NamedTuple

from env.layout import is_walkable
from env.warehouse import BACKWARD, FORWARD, HEADINGS, TURN_LEFT, TURN_RIGHT
from rl.nav import UNREACHABLE

WINDOW = 24             # ticks planned ahead; a job is ~50 actions, so plans are windowed (WHCA*)
HUMAN_HOLD = 2          # ticks a seen human's cell stays forbidden (they move with p=0.5)
HUMAN_NEAR_COST = 2.0   # extra cost of a cell next to a seen human during HUMAN_HOLD


class Step(NamedTuple):
    """The action taken at one tick, and the state AFTER it."""
    action: int
    cell: tuple
    heading: int


class Reservations:
    """Who is where when. cells: (cell, t) -> robot. edges: (from, to, t) = a move that
    ARRIVES at `to` at tick t; booking it forbids the swap to->from at the same tick."""

    def __init__(self):
        self.cells, self.edges = {}, {}
        self.owned = {}         # robot -> [("c" | "e", key)], so release() does not scan the table

    def free(self, cell, t, robot):
        return self.cells.get((cell, t), robot) == robot

    def swapped(self, a, b, t, robot):
        return self.edges.get((b, a, t), robot) != robot

    def book(self, robot, cell, t0, t1):
        """Hold `cell` from tick t0 to t1 inclusive."""
        for t in range(t0, t1 + 1):
            self.cells[(cell, t)] = robot
            self.owned.setdefault(robot, []).append(("c", (cell, t)))

    def book_move(self, robot, a, b, t):
        self.edges[(a, b, t)] = robot
        self.owned.setdefault(robot, []).append(("e", (a, b, t)))

    def book_path(self, robot, start, path, window):
        """path: list of Step (tick 1..). Books every cell per tick, every move (book_move
        when the cell changed), and the final cell from the last tick to `window`. An empty
        path books `start` from tick 0 to `window` (the robot holds)."""
        here = start
        for t, step in enumerate(path, 1):
            self.book(robot, step.cell, t, t)
            if step.cell != here:
                self.book_move(robot, here, step.cell, t)
            here = step.cell
        if not path:
            self.book(robot, start, 0, window)
        elif len(path) < window:
            self.book(robot, here, len(path) + 1, window)

    def release(self, robot):
        for kind, key in self.owned.pop(robot, []):
            table = self.cells if kind == "c" else self.edges
            if table.get(key) == robot:
                del table[key]

    def crossed(self):
        """Every cell somebody is booked on at tick 1 or later."""
        return {cell for (cell, t) in self.cells if t >= 1}


def plan(grid, robot, start, heading, ready_at, goal, heuristic, table, hard, soft,
         window=WINDOW, finish=None):
    """A* over (cell, heading, t) for one robot against the table.

    ready_at(t): False on a tick the env will ignore this robot's action (slow robot); its
      only edge then is "stay, same heading" with action TURN_RIGHT (ignored by the env).
    goal(cell, heading): True at a goal state.  heuristic(cell, heading): admissible ticks
      to a goal, or UNREACHABLE to prune.  hard: set of (cell, t) forbidden.  soft: dict
      (cell, t) -> extra cost.  finish: action to append once a goal is reached (PICKUP or
      DROPOFF; the robot then stays one more tick, and the action must land on a tick the env
      carries out), or None.
    Returns a list of Step (tick 1..), or None when nothing at all could be expanded.
    Windowed: if no goal is reached by `window`, the popped state with the lowest heuristic
    (ties: the later tick) is returned as a partial plan.
    """
    first = heuristic(start, heading)
    if first == UNREACHABLE:
        return None
    root = (start, heading, 0)
    tie = count()
    heap, parent, cost, closed = [(first, 0.0, next(tie), root)], {root: None}, {root: 0.0}, set()
    best = None                                         # (h, -t, node) of the closest popped state
    while heap:
        _, g, _, node = heapq.heappop(heap)
        if node in closed:
            continue
        closed.add(node)
        cell, facing, t = node
        if t < window and goal(cell, facing):
            if finish is None:
                return _path(parent, node)
            if (ready_at(t) and table.free(cell, t + 1, robot)
                    and (cell, t + 1) not in hard):
                return _path(parent, node) + [Step(finish, cell, facing)]
        if node != root:
            key = (heuristic(cell, facing), -t)
            if best is None or key < best[0]:
                best = (key, node)
        if t == window:
            continue
        for action, cell2, facing2 in _moves(grid, robot, node, ready_at(t), table, hard):
            nxt = (cell2, facing2, t + 1)
            if nxt in closed:
                continue
            h = heuristic(cell2, facing2)
            if h == UNREACHABLE:
                continue
            g2 = g + 1 + soft.get((cell2, t + 1), 0.0)
            if g2 < cost.get(nxt, float("inf")):
                cost[nxt], parent[nxt] = g2, (node, action)
                heapq.heappush(heap, (g2 + h, g2, next(tie), nxt))
    return None if best is None else _path(parent, best[1])


def _moves(grid, robot, node, ready, table, hard):
    """Successors (action, cell, heading) of a state one tick later that the table allows."""
    cell, heading, t = node
    t1 = t + 1
    stay = table.free(cell, t1, robot) and (cell, t1) not in hard
    if not ready:
        return [(TURN_RIGHT, cell, heading)] if stay else []
    out = []
    if stay:
        out += [(TURN_RIGHT, cell, (heading + 1) % 4), (TURN_LEFT, cell, (heading - 1) % 4)]
    dx, dy = HEADINGS[heading]
    for action, sign in ((FORWARD, 1), (BACKWARD, -1)):
        cell2 = (cell[0] + sign * dx, cell[1] + sign * dy)
        if (is_walkable(grid, *cell2) and table.free(cell2, t1, robot)
                and not table.swapped(cell, cell2, t1, robot) and (cell2, t1) not in hard):
            out.append((action, cell2, heading))
    return out


def _path(parent, node):
    """The Steps from tick 1 to `node`, following parent links back to the root."""
    steps = []
    while parent[node] is not None:
        prev, action = parent[node]
        steps.append(Step(action, node[0], node[1]))
        node = prev
    return steps[::-1]
