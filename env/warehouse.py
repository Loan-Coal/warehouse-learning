"""Multi-robot grid warehouse. Pure Python, standard library only; never imports Isaac.

Decentralized navigation, centralized dispatch: each robot acts on its own local
observation, while the env's task queue plays the warehouse management system.
The obs / action / reward / get_state() formats are the contract between the RL and
Isaac sides; tests/test_env.py guards them.

Every complexity factor is OFF by default (see Warehouse.__init__): switch on only what
you want to study, or pick a named scenario from env/scenarios.py.
"""
import random
from dataclasses import dataclass

from env.layout import AISLE, DEFAULT_MAP, DOCK, SHELF, cells_of, is_walkable, parse, problems
from env.observation import (DEFAULT_OBS, DOWN, EMPTY_SHELF, FULL_SHELF, HEADINGS, HUMAN, LEFT,  # noqa: F401
                             NOTHING, RIGHT, ROBOT, UP, WALL, DOCK_HIT, obs_type, observe)

# Bump on any change to the observations, the actions or the rewards: checkpoints trained
# on another version are refused, because their weights no longer mean anything.
CONTRACT_VERSION = 2

N_ACTIONS = 6
PICKUP, DROPOFF, TURN_RIGHT, TURN_LEFT, FORWARD, BACKWARD = range(N_ACTIONS)
ACTION_NAMES = ("pickup", "dropoff", "turn_right", "turn_left", "forward", "backward")

R_DELIVERY, R_STEP, R_BLOCKED, R_INVALID = 10.0, -0.05, -1.0, -0.1

# priority=True: a delivery is worth priority * max(VALUE_FLOOR, 1 - age / VALUE_DECAY_STEPS).
VALUE_DECAY_STEPS, VALUE_FLOOR = 100, 0.25
HUMAN_MOVE_PROB = 0.5


@dataclass(frozen=True)
class RobotType:
    name: str
    period: int         # acts every `period` steps: 1 = full speed, 2 = half speed
    max_weight: int
    max_volume: int


STANDARD = RobotType("standard", period=1, max_weight=3, max_volume=3)
# mixed_robots=True: robot i gets ROBOT_TYPES[i % 2].
ROBOT_TYPES = (RobotType("small", period=1, max_weight=1, max_volume=2),
               RobotType("large", period=2, max_weight=3, max_volume=3))


@dataclass(frozen=True)
class EnvSpec:
    """Everything a policy may know about the env before the first step of an episode."""
    grid: tuple           # the map rows; robots know the building, not where the others are
    n_robots: int
    obs_features: tuple = DEFAULT_OBS
    n_actions: int = N_ACTIONS
    contract_version: int = CONTRACT_VERSION

    @property
    def obs_fields(self):
        return obs_type(self.obs_features)._fields

    @property
    def obs_size(self):
        return len(self.obs_fields)


@dataclass(frozen=True)
class Task:
    kind: str       # "retrieve" (shelf -> dock) or "store" (dock -> shelf)
    pickup: tuple
    dropoff: tuple
    weight: int = 1
    volume: int = 1
    priority: int = 1
    created: int = 0    # step at which the job appeared


@dataclass
class Robot:
    pos: tuple
    heading: int
    kind: RobotType = STANDARD
    task: Task | None = None
    crate: int | None = None    # id of the carried crate


class Warehouse:
    def __init__(self, layout=DEFAULT_MAP, n_robots=2, max_steps=100, view_range=5,
                 alpha=0.5, task_prob=0.3, obs=DEFAULT_OBS,
                 mixed_crates=False, mixed_robots=False, priority=False, humans=0):
        """
        obs:          observation feature groups (env/observation.py), normally the policy's OBS
        mixed_crates: crates get a random weight and volume (1-3)
        mixed_robots: robots differ in speed, max weight and max volume (ROBOT_TYPES)
        priority:     jobs get a priority 1-3 that multiplies the delivery reward, which decays with age
        humans:       number of humans walking randomly in the aisles, blocking robots
        """
        # Labels 0-9 / A-J in render() cap the robot count.
        if not 1 <= n_robots <= 10:
            raise ValueError("n_robots must be between 1 and 10")
        if not 0.0 <= alpha <= 1.0 or not 0.0 <= task_prob <= 1.0:
            raise ValueError("alpha and task_prob must be in [0, 1]")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if humans < 0:
            raise ValueError("humans must be 0 or more")
        self.obs_features = tuple(obs)
        obs_type(self.obs_features)     # fail now on a bad feature name
        self.n_robots, self.max_steps, self.view_range = n_robots, max_steps, view_range
        self.alpha, self.task_prob = alpha, task_prob
        self.mixed_crates, self.mixed_robots = mixed_crates, mixed_robots
        self.priority, self.n_humans = priority, humans
        self._set_layout(layout)
        self.reset()

    # ------------------------------------------------------------------ API

    @property
    def spec(self):
        return EnvSpec(self.grid, self.n_robots, self.obs_features)

    def reset(self, seed=None, layout=None):
        """Start an episode. A new `layout` (map string) replaces the map from now on."""
        if layout is not None:
            self._set_layout(layout)
        self.rng = random.Random(seed)
        self.t = 0
        self.totals = {"deliveries": 0, "value": 0.0, "blocked": 0, "invalid": 0}
        starts = self.rng.sample(cells_of(self.grid, AISLE), self.n_robots + self.n_humans)
        kinds = [ROBOT_TYPES[i % len(ROBOT_TYPES)] if self.mixed_robots else STANDARD
                 for i in range(self.n_robots)]
        self.robots = [Robot(pos, self.rng.randrange(4), kind)
                       for pos, kind in zip(starts, kinds)]
        self.humans = tuple(starts[self.n_robots:])
        full = self.rng.sample(self.shelves, round(2 / 3 * len(self.shelves)))
        self.crate_at = {cell: i for i, cell in enumerate(full)}   # crates resting on S/D cells
        self.crate_info = {i: self._random_crate() for i in range(len(full))}   # id -> (weight, volume)
        self.next_crate_id = len(full)
        self.queue, self.reserved = [], set()
        self._dispatch()
        return self._observe_all()

    def ready(self, i):
        """True if robot i's action counts this step (slow robots skip steps)."""
        return (self.t + i) % self.robots[i].kind.period == 0

    def step(self, actions):
        if self.t >= self.max_steps:
            raise RuntimeError("episode is over; call reset()")
        # isinstance matters: `2.0 in range(6)` is True and would crash later as a tuple index.
        if len(actions) != self.n_robots or any(
                not isinstance(a, int) or a not in range(N_ACTIONS) for a in actions):
            raise ValueError(f"need {self.n_robots} actions, each an int in 0..{N_ACTIONS - 1}")
        n = self.n_robots
        value, invalid, hit_human = [0.0] * n, [False] * n, [False] * n
        start = [r.pos for r in self.robots]
        intended = list(start)
        for i, a in enumerate(actions):
            robot = self.robots[i]
            if not self.ready(i):
                continue    # a slow robot's action between its turns is ignored, not punished
            if a in (PICKUP, DROPOFF):
                ok, value[i] = self._pickup(robot) if a == PICKUP else self._dropoff(robot)
                invalid[i] = not ok
            elif a in (TURN_RIGHT, TURN_LEFT):
                robot.heading = (robot.heading + (1 if a == TURN_RIGHT else -1)) % 4
            else:
                dx, dy = HEADINGS[robot.heading]
                sign = 1 if a == FORWARD else -1
                cell = (start[i][0] + sign * dx, start[i][1] + sign * dy)
                if not is_walkable(self.grid, *cell):
                    invalid[i] = True
                elif cell in self.humans:
                    hit_human[i] = True
                else:
                    intended[i] = cell
        final = self._resolve_moves(start, intended)
        blocked = [hit_human[i] or (intended[i] != start[i] and final[i] == start[i]) for i in range(n)]
        for robot, pos in zip(self.robots, final):
            robot.pos = pos
        self._move_humans()

        own = [R_STEP + R_DELIVERY * value[i] + R_BLOCKED * blocked[i] + R_INVALID * invalid[i]
               for i in range(n)]
        # alpha trickles everyone's outcome down to every robot: 0 = selfish, 1 = fully shared.
        team = sum(own) / n
        rewards = [(1 - self.alpha) * r + self.alpha * team for r in own]

        self.t += 1
        self._dispatch()
        info = {"deliveries": sum(v > 0 for v in value), "value": sum(value), "blocked": sum(blocked),
                "invalid": sum(invalid), "truncated": self.t >= self.max_steps}
        for key in self.totals:
            self.totals[key] += info[key]
        # There is no terminal state: done is always a time-limit truncation, so a learner
        # should keep bootstrapping from next_obs rather than treat it as an absorbing state.
        return self._observe_all(), rewards, info["truncated"], info

    def render(self):
        rows = [list(row) for row in self.grid]
        for x, y in self.shelves:
            rows[y][x] = "S" if (x, y) in self.crate_at else "s"
        for x, y in self.docks:
            rows[y][x] = "D" if (x, y) in self.crate_at else "d"
        for x, y in self.humans:
            rows[y][x] = "H"
        for i, r in enumerate(self.robots):
            rows[r.pos[1]][r.pos[0]] = chr(ord("A") + i) if r.crate is not None else str(i)
        lines = ["".join(row) for row in rows]
        lines.append(f"step {self.t}/{self.max_steps}  queue {len(self.queue)}  "
                     + "  ".join(f"{k} {v:g}" for k, v in self.totals.items()))
        for i, r in enumerate(self.robots):
            facing = "^>v<"[r.heading]
            head = f"  robot {i} {facing} ({r.kind.name})"
            if r.task is None:
                lines.append(f"{head}: idle")
            else:
                stage = "carrying" if r.crate is not None else "going to pickup"
                lines.append(f"{head}: {r.task.kind} {r.task.pickup}->{r.task.dropoff} "
                             f"(prio {r.task.priority}), {stage}")
        lines.append("legend: 0-9 robot, A-J robot carrying, H human, S/s shelf with/without crate, "
                     "D/d dock with/without crate, . aisle")
        return "\n".join(lines)

    def get_state(self):
        """Plain copied data describing everything a viewer needs to draw. Isaac reads only this."""
        crates = [{"id": cid, "x": x, "y": y, "carried_by": None}
                  for (x, y), cid in self.crate_at.items()]
        crates += [{"id": r.crate, "x": r.pos[0], "y": r.pos[1], "carried_by": i}
                   for i, r in enumerate(self.robots) if r.crate is not None]
        for c in crates:
            c["weight"], c["volume"] = self.crate_info[c["id"]]
        robots = []
        for r in self.robots:
            target = None if r.task is None else list(r.task.dropoff if r.crate is not None else r.task.pickup)
            robots.append({"x": r.pos[0], "y": r.pos[1], "heading": r.heading, "carrying": r.crate is not None,
                           "kind": r.kind.name, "target": target,
                           "priority": 0 if r.task is None else r.task.priority})
        return {
            "width": self.width,
            "height": self.height,
            "grid": list(self.grid),
            "robots": robots,
            "crates": sorted(crates, key=lambda c: c["id"]),
            "humans": [{"x": x, "y": y} for x, y in self.humans],
            "queue_len": len(self.queue),
            "step": self.t,
        }

    # ------------------------------------------------------------ dynamics

    def _set_layout(self, layout):
        grid = parse(layout)
        found = problems(grid, self.n_robots)
        if not found and len(cells_of(grid, AISLE)) < 2 * self.n_robots + self.n_humans:
            found.append(f"not enough aisle cells for {self.n_robots} robots and {self.n_humans} humans")
        if found:
            raise ValueError("invalid layout: " + "; ".join(found))
        self.grid = grid
        self.width, self.height = len(grid[0]), len(grid)
        self.shelves, self.docks = cells_of(grid, SHELF), cells_of(grid, DOCK)

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

    def _move_humans(self):
        """Each human steps to a random free neighbouring aisle cell with HUMAN_MOVE_PROB."""
        occupied = {r.pos for r in self.robots} | set(self.humans)
        moved = []
        for pos in self.humans:
            new = pos
            if self.rng.random() < HUMAN_MOVE_PROB:
                options = [(pos[0] + dx, pos[1] + dy) for dx, dy in HEADINGS]
                options = [c for c in options if is_walkable(self.grid, *c) and c not in occupied]
                if options:
                    new = self.rng.choice(options)
                    occupied = (occupied - {pos}) | {new}
            moved.append(new)
        self.humans = tuple(moved)

    def _front(self, robot):
        dx, dy = HEADINGS[robot.heading]
        return robot.pos[0] + dx, robot.pos[1] + dy

    def _pickup(self, robot):
        """Returns (valid, delivered value). Valid when facing the pickup of the current job."""
        task = robot.task
        if task is None or robot.crate is not None or self._front(robot) != task.pickup:
            return False, 0.0
        robot.crate = self.crate_at.pop(task.pickup)
        return True, 0.0

    def _dropoff(self, robot):
        task = robot.task
        if robot.crate is None or self._front(robot) != task.dropoff:
            return False, 0.0
        if task.kind == "store":
            self.crate_at[task.dropoff] = robot.crate
        else:
            # A retrieved crate dropped at a dock is shipped: it simply leaves the system.
            del self.crate_info[robot.crate]
        robot.crate, robot.task = None, None
        self.reserved -= _reserved_by(task)
        return True, self._value(task)

    def _value(self, task):
        if not self.priority:
            return 1.0
        age = self.t - task.created
        return task.priority * max(VALUE_FLOOR, 1 - age / VALUE_DECAY_STEPS)

    # ------------------------------------------------------------ dispatch

    def _dispatch(self):
        # Always draw from the rng so trajectories stay reproducible for a given seed.
        if self.rng.random() < self.task_prob and len(self.queue) < self.n_robots:
            kinds = [k for k in ("retrieve", "store") if self._candidates(k)]
            if kinds:
                self.queue.append(self._new_task(self.rng.choice(kinds)))
        self._assign_idle()

    def _random_crate(self):
        """(weight, volume). Sized for a random robot of the fleet, so some robot can always carry it."""
        if not self.mixed_crates:
            return 1, 1
        kind = self.rng.choice(self.robots).kind
        return self.rng.randint(1, kind.max_weight), self.rng.randint(1, kind.max_volume)

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
            weight, volume = self.crate_info[self.crate_at[pickup]]
        else:
            dropoff = self.rng.choice(dropoffs)
            weight, volume = self._random_crate()
        priority = self.rng.randint(1, 3) if self.priority else 1
        task = Task(kind, pickup, dropoff, weight, volume, priority, self.t)
        self.reserved |= _reserved_by(task)
        return task

    def _assign_idle(self):
        """Each idle robot takes the highest-priority queued job it can carry (oldest first on ties)."""
        for robot in self.robots:
            if robot.task is not None:
                continue
            fits = [t for t in self.queue
                    if t.weight <= robot.kind.max_weight and t.volume <= robot.kind.max_volume]
            if not fits:
                continue
            task = max(fits, key=lambda t: t.priority)    # max keeps the first of equal priorities
            self.queue.remove(task)
            robot.task = task
            if task.kind == "store":
                # The incoming crate arrives when a robot is dispatched, so docks never pile up.
                self.crate_at[task.pickup] = self.next_crate_id
                self.crate_info[self.next_crate_id] = (task.weight, task.volume)
                self.next_crate_id += 1

    def _observe_all(self):
        return [observe(self, i, self.obs_features) for i in range(self.n_robots)]


def _reserved_by(task):
    """Cells a task holds until it is delivered.

    A retrieve's dropoff dock is not held: shipping is instant, so a dock can take any
    number of them, and releasing it must not free a dock a store task is holding.
    """
    return {task.pickup} if task.kind == "retrieve" else {task.pickup, task.dropoff}


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
