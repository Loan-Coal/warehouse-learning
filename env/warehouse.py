"""Multi-robot grid warehouse. Pure Python, standard library only; never imports Isaac.

Decentralized navigation, centralized dispatch: each robot acts on its own local
observation, while the env's task queue plays the warehouse management system.
The obs / action / reward / get_state() formats are the contract between the RL and
Isaac sides; tests/test_env.py guards them.
"""
import random
from dataclasses import dataclass

from env.layout import AISLE, DEFAULT_MAP, DIRS, DOCK, SHELF, cells_of, in_bounds, parse, problems

N_ACTIONS = 6
WAIT, INTERACT = 4, 5

# Ray hit types in the observation.
NOTHING, WALL, EMPTY_SHELF, FULL_SHELF, DOCK_HIT, ROBOT = range(6)

R_DELIVERY, R_STEP, R_BLOCKED, R_INVALID = 10.0, -0.05, -1.0, -0.1


@dataclass(frozen=True)
class Task:
    kind: str       # "retrieve" (shelf -> dock) or "store" (dock -> shelf)
    pickup: tuple
    dropoff: tuple


@dataclass
class Robot:
    pos: tuple
    task: Task | None = None
    crate: int | None = None    # id of the carried crate


class Warehouse:
    def __init__(self, layout=DEFAULT_MAP, n_robots=2, max_steps=100, view_range=5,
                 alpha=0.5, task_prob=0.3):
        self.grid = parse(layout)
        # Labels 0-9 / A-J in render() cap the robot count.
        if not 1 <= n_robots <= 10:
            raise ValueError("n_robots must be between 1 and 10")
        found = problems(self.grid, n_robots)
        if found:
            raise ValueError("invalid layout: " + "; ".join(found))
        if not 0.0 <= alpha <= 1.0 or not 0.0 <= task_prob <= 1.0:
            raise ValueError("alpha and task_prob must be in [0, 1]")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.width, self.height = len(self.grid[0]), len(self.grid)
        self.n_robots, self.max_steps, self.view_range = n_robots, max_steps, view_range
        self.alpha, self.task_prob = alpha, task_prob
        self.shelves, self.docks = cells_of(self.grid, SHELF), cells_of(self.grid, DOCK)
        self.reset()

    # ------------------------------------------------------------------ API

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.t = 0
        self.totals = {"deliveries": 0, "blocked": 0, "invalid": 0}
        full = self.rng.sample(self.shelves, round(2 / 3 * len(self.shelves)))
        self.crate_at = {cell: i for i, cell in enumerate(full)}   # crates resting on S/D cells
        self.next_crate_id = len(full)
        starts = self.rng.sample(cells_of(self.grid, AISLE), self.n_robots)
        self.robots = [Robot(pos) for pos in starts]
        self.queue, self.reserved = [], set()
        self._dispatch()
        return [self._observe(i) for i in range(self.n_robots)]

    def step(self, actions):
        if self.t >= self.max_steps:
            raise RuntimeError("episode is over; call reset()")
        # isinstance matters: `2.0 in range(6)` is True and would crash later as a tuple index.
        if len(actions) != self.n_robots or any(
                not isinstance(a, int) or a not in range(N_ACTIONS) for a in actions):
            raise ValueError(f"need {self.n_robots} actions, each an int in 0..{N_ACTIONS - 1}")
        n = self.n_robots
        delivered, invalid = [False] * n, [False] * n
        start = [r.pos for r in self.robots]
        intended = list(start)
        for i, a in enumerate(actions):
            if a == INTERACT:
                ok, delivered[i] = self._interact(self.robots[i])
                invalid[i] = not ok
            elif a != WAIT:
                x, y = start[i][0] + DIRS[a][0], start[i][1] + DIRS[a][1]
                if in_bounds(self.grid, x, y) and self.grid[y][x] == AISLE:
                    intended[i] = (x, y)
                else:
                    invalid[i] = True
        final = self._resolve_moves(start, intended)
        blocked = [intended[i] != start[i] and final[i] == start[i] for i in range(n)]
        for robot, pos in zip(self.robots, final):
            robot.pos = pos

        own = [R_STEP + R_DELIVERY * delivered[i] + R_BLOCKED * blocked[i] + R_INVALID * invalid[i]
               for i in range(n)]
        # alpha trickles everyone's outcome down to every robot: 0 = selfish, 1 = fully shared.
        team = sum(own) / n
        rewards = [(1 - self.alpha) * r + self.alpha * team for r in own]

        self.t += 1
        self._dispatch()
        info = {"deliveries": sum(delivered), "blocked": sum(blocked),
                "invalid": sum(invalid), "truncated": self.t >= self.max_steps}
        for key in self.totals:
            self.totals[key] += info[key]
        # There is no terminal state: done is always a time-limit truncation, so a learner
        # should keep bootstrapping from next_obs rather than treat it as an absorbing state.
        return [self._observe(i) for i in range(n)], rewards, info["truncated"], info

    def render(self):
        rows = [list(row) for row in self.grid]
        for x, y in self.shelves:
            rows[y][x] = "S" if (x, y) in self.crate_at else "s"
        for x, y in self.docks:
            rows[y][x] = "D" if (x, y) in self.crate_at else "d"
        for i, r in enumerate(self.robots):
            rows[r.pos[1]][r.pos[0]] = chr(ord("A") + i) if r.crate is not None else str(i)
        lines = ["".join(row) for row in rows]
        lines.append(f"step {self.t}/{self.max_steps}  queue {len(self.queue)}  "
                     + "  ".join(f"{k} {v}" for k, v in self.totals.items()))
        for i, r in enumerate(self.robots):
            if r.task is None:
                lines.append(f"  robot {i}: idle")
            else:
                stage = "carrying" if r.crate is not None else "going to pickup"
                lines.append(f"  robot {i}: {r.task.kind} {r.task.pickup}->{r.task.dropoff}, {stage}")
        lines.append("legend: 0-9 robot, A-J robot carrying, S/s shelf with/without crate, "
                     "D/d dock with/without crate, . aisle")
        return "\n".join(lines)

    def get_state(self):
        """Plain copied data describing everything a viewer needs to draw. Isaac reads only this."""
        crates = [{"id": cid, "x": x, "y": y, "carried_by": None}
                  for (x, y), cid in self.crate_at.items()]
        crates += [{"id": r.crate, "x": r.pos[0], "y": r.pos[1], "carried_by": i}
                   for i, r in enumerate(self.robots) if r.crate is not None]
        return {
            "width": self.width,
            "height": self.height,
            "grid": list(self.grid),
            "robots": [{"x": r.pos[0], "y": r.pos[1], "carrying": r.crate is not None}
                       for r in self.robots],
            "crates": sorted(crates, key=lambda c: c["id"]),
            "queue_len": len(self.queue),
            "step": self.t,
        }

    # ------------------------------------------------------------ dynamics

    def _resolve_moves(self, start, intended):
        """Cancel conflicting moves until none remain; correct for any number of robots.

        - Moving into a robot that stays, or swapping with another robot: cancelled.
        - Several robots moving into the same free cell: one seeded-random winner moves.
          Robots that cannot see each other (e.g. arriving diagonally) would otherwise
          collide identically forever under any deterministic policy.
        Each pass finds every conflict before cancelling any, so robot order never matters;
        a cancel can create a new conflict behind it (a chain), hence the repeated passes.
        """
        n = len(start)
        # Drawn every step, conflict or not, so the rng stream depends only on the seed.
        priority = [self.rng.random() for _ in range(n)]
        final = list(intended)
        while True:
            cancel = []
            for i in range(n):
                if final[i] == start[i]:
                    continue
                rivals = [j for j in range(n) if final[j] == final[i]]
                into_stayer = any(final[j] == start[j] for j in rivals)
                swapped = any(final[j] == start[i] and final[i] == start[j] for j in range(n) if j != i)
                lost = max(rivals, key=priority.__getitem__) != i
                if into_stayer or swapped or lost:
                    cancel.append(i)
            if not cancel:
                return final
            for i in cancel:
                final[i] = start[i]

    def _interact(self, robot):
        """Returns (valid, delivered)."""
        task = robot.task
        if task is None:
            return False, False
        if robot.crate is None and _adjacent(robot.pos, task.pickup):
            robot.crate = self.crate_at.pop(task.pickup)
            return True, False
        if robot.crate is not None and _adjacent(robot.pos, task.dropoff):
            if task.kind == "store":
                self.crate_at[task.dropoff] = robot.crate
            # A retrieved crate dropped at a dock is shipped: it simply leaves the system.
            robot.crate, robot.task = None, None
            self.reserved -= _reserved_by(task)
            return True, True
        return False, False

    # ------------------------------------------------------------ dispatch

    def _dispatch(self):
        # Always draw from the rng so trajectories stay reproducible for a given seed.
        if self.rng.random() < self.task_prob and len(self.queue) < self.n_robots:
            kinds = [k for k in ("retrieve", "store") if self._candidates(k)]
            if kinds:
                self.queue.append(self._new_task(self.rng.choice(kinds)))
        self._assign_idle()

    def _candidates(self, kind):
        """Free cells a new task of this kind could use: (pickups, dropoffs), or None."""
        if kind == "retrieve":
            pickups, dropoffs = self._free(self.shelves, holding=True), self.docks
        else:
            pickups, dropoffs = self._free(self.docks, holding=False), self._free(self.shelves, holding=False)
        return (pickups, dropoffs) if pickups and dropoffs else None

    def _free(self, cells, holding):
        return [c for c in cells if c not in self.reserved and (c in self.crate_at) == holding]

    def _new_task(self, kind):
        pickups, dropoffs = self._candidates(kind)
        pickup = self.rng.choice(pickups)
        if kind == "retrieve":
            # Nearest dock; ties go to the lowest dock index because min() is stable.
            dropoff = min(dropoffs, key=lambda d: _manhattan(pickup, d))
        else:
            dropoff = self.rng.choice(dropoffs)
        task = Task(kind, pickup, dropoff)
        self.reserved |= _reserved_by(task)
        return task

    def _assign_idle(self):
        for robot in self.robots:
            if robot.task is None and self.queue:
                robot.task = self.queue.pop(0)
                if robot.task.kind == "store":
                    # The incoming crate arrives when a robot is dispatched, so docks never pile up.
                    self.crate_at[robot.task.pickup] = self.next_crate_id
                    self.next_crate_id += 1

    # --------------------------------------------------------- observation

    def _observe(self, i):
        r = self.robots[i]
        if r.task is None:
            target = r.pos
        else:
            target = r.task.dropoff if r.crate is not None else r.task.pickup
        rays = [v for ray in self._rays(r) for v in ray]
        return (r.pos[0], r.pos[1], int(r.crate is not None), int(r.task is not None),
                target[0], target[1], *rays)

    def _rays(self, robot):
        """(distance, type) of the first thing seen up, down, left, right; (0, 0) if nothing."""
        others = {r.pos for r in self.robots if r is not robot}
        rays = []
        for dx, dy in DIRS:
            hit = (0, NOTHING)
            for d in range(1, self.view_range + 1):
                x, y = robot.pos[0] + dx * d, robot.pos[1] + dy * d
                kind = self._cell_type(x, y, others)
                if kind != NOTHING:
                    hit = (d, kind)
                    break
            rays.append(hit)
        return rays

    def _cell_type(self, x, y, others):
        if not in_bounds(self.grid, x, y):
            return WALL
        if (x, y) in others:
            return ROBOT
        c = self.grid[y][x]
        if c == SHELF:
            return FULL_SHELF if (x, y) in self.crate_at else EMPTY_SHELF
        if c == DOCK:
            return DOCK_HIT
        return NOTHING


def _reserved_by(task):
    """Cells a task holds until it is delivered.

    A retrieve's dropoff dock is not held: shipping is instant, so a dock can take any
    number of them, and releasing it must not free a dock a store task is holding.
    """
    return {task.pickup} if task.kind == "retrieve" else {task.pickup, task.dropoff}


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _adjacent(a, b):
    return _manhattan(a, b) == 1
