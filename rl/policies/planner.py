"""The benchmark planner: no learning, robots never block each other by construction.

Every step each robot books the cells it will occupy tick by tick (rl/spacetime.py), shortest
job first, so a robot planned later routes around the ones before it. Everyone re-plans every
step. `--policy-opt info=` sets what it may know about HUMANS (the only thing it cannot book):
  obs     what PPO sees: humans only through each robot's rays
  state   every human's position
  future  a copy of the env with the real random stream: it sees where humans WILL step
The three levels bound what information is worth, against what learning is worth (README).
"""
import json

from env.layout import DIRS, is_walkable
from env.observation import HEADINGS, HUMAN
from env.warehouse import DROPOFF, PICKUP, TURN_RIGHT
from rl.features import task_value
from rl.metrics import facing_field, facing_states, state_index
from rl.model import clone
from rl.nav import UNREACHABLE
from rl.policy import Policy
from rl.spacetime import HUMAN_HOLD, HUMAN_NEAR_COST, WINDOW, Reservations, plan

INFO_LEVELS = ("obs", "state", "future")
RESTARTS = 3                # extra attempts, each putting the first robot that failed ahead of the rest
FUTURE_ROUNDS = 2           # simulate-and-replan rounds at info=future
FAR = 10 ** 9               # sorts a robot with an unreachable target last
SETTINGS = "info, window, restarts"


class Planner(Policy):
    OBS = ("position", "heading", "task", "rays", "crate", "robot")

    def __init__(self, spec, seed=0):
        super().__init__(spec, seed)
        self.info, self.window, self.restarts = "obs", WINDOW, RESTARTS
        self.env = None

    def configure(self, info=None, window=None, restarts=None, **unknown):
        if unknown:
            raise ValueError(f"unknown setting {', '.join(sorted(unknown))}; this policy takes {SETTINGS}")
        if info is not None:
            if info not in INFO_LEVELS:
                raise ValueError(f"info wants one of {', '.join(INFO_LEVELS)}, got '{info}'")
            self.info = info
        if window is not None:
            self.window = _integer(window, "window", 1)
        if restarts is not None:
            self.restarts = _integer(restarts, "restarts", 0)

    def see_env(self, env):
        self.env = env

    # ------------------------------------------------------------------ acting

    def act(self, obs):
        hard, soft = self.human_constraints(self.known_humans(obs))
        paths, _ = self.plan_all(obs, hard, soft)
        if self.info == "future":
            for _ in range(FUTURE_ROUNDS):
                hard, soft = self.trajectory(len(obs), paths), {}
                paths, _ = self.plan_all(obs, hard, soft)
        return [paths[i][0].action if paths.get(i) else TURN_RIGHT for i in range(len(obs))]

    def known_humans(self, obs):
        """Cells with a human, as far as `info` lets this planner know."""
        if self.info != "obs":
            if self.env is None:
                raise RuntimeError(f"planner info={self.info} needs the env: run it through "
                                   "rl.run, rl.bench or isaac.demo")
            return set(self.env.humans)
        found = set()
        for o in obs:
            rays = ((o.front_d, o.front_t), (o.right_d, o.right_t),
                    (o.back_d, o.back_t), (o.left_d, o.left_t))
            for k, (dist, kind) in enumerate(rays):
                if kind == HUMAN:
                    dx, dy = HEADINGS[(o.heading + k) % 4]
                    found.add((o.x + dx * dist, o.y + dy * dist))
        return found

    def human_constraints(self, humans):
        """Humans move with p=0.5, so a seen one blocks its cell and taxes its neighbours briefly."""
        grid, hard, soft = self.spec.grid, set(), {}
        for cell in humans:
            near = [(cell[0] + dx, cell[1] + dy) for dx, dy in DIRS]
            near = [c for c in near if is_walkable(grid, *c)]
            for t in range(1, HUMAN_HOLD + 1):
                hard.add((cell, t))
                soft.update({(c, t): HUMAN_NEAR_COST for c in near})
        return hard, soft

    def trajectory(self, n, paths):
        """Where every human will be at each tick if everyone follows `paths` (true random stream)."""
        model, hard = clone(self.env), set()
        for t in range(1, self.window + 1):
            if model.t >= model.max_steps:
                break
            model.step([paths[i][t - 1].action if t - 1 < len(paths.get(i, ())) else TURN_RIGHT
                        for i in range(n)])
            hard |= {(cell, t) for cell in model.humans}
        return hard

    # ------------------------------------------------------------------ planning

    def plan_all(self, obs, hard, soft):
        """(paths, table) of the best attempt: fewest robots left without a plan."""
        order = self.priority_order(obs)
        best = None
        for _ in range(self.restarts + 1):
            attempt = self.attempt(obs, order, hard, soft)
            if best is None or len(attempt[2]) < len(best[2]):
                best = attempt
            if not attempt[2]:
                break
            order = [attempt[2][0]] + [i for i in order if i != attempt[2][0]]
        return best[0], best[1]

    def attempt(self, obs, order, hard, soft):
        table, paths, failed = Reservations(), {}, []
        cells = [(o.x, o.y) for o in obs]
        for i, cell in enumerate(cells):
            table.book(i, cell, 0, 1)       # a robot planned earlier never relies on a later one moving away
        for i in order:
            table.release(i)
            path = self.plan_one(i, obs[i], table, hard, soft)
            if path is None:
                failed.append(i)
                path = []
            else:
                paths[i] = path
            table.book_path(i, cells[i], path, self.window)
        return paths, table, failed

    def priority_order(self, obs):
        """Busy robots by fewest actions left (ties: richer job, then index), then idle ones."""
        grid = self.spec.grid

        def remaining(i):
            o = obs[i]
            field = facing_field(grid, (o.target_x, o.target_y))
            left = field[state_index(grid, ((o.x, o.y), o.heading))]
            return FAR if left == UNREACHABLE else left

        busy = sorted((i for i, o in enumerate(obs) if o.has_task),
                      key=lambda i: (remaining(i), -task_value(obs[i]), i))
        return busy + [i for i, o in enumerate(obs) if not o.has_task]

    def plan_one(self, i, o, table, hard, soft):
        """One robot's path against the table, [] to hold, None when it cannot be planned."""
        grid, here = self.spec.grid, (o.x, o.y)
        offset = 0 if o.ready else 1
        ready_at = lambda t: (t + offset) % o.period == 0      # noqa: E731
        if o.has_task:
            target = (o.target_x, o.target_y)
            field, goals = facing_field(grid, target), set(facing_states(grid, target))
            return plan(grid, i, here, o.heading, ready_at,
                        lambda c, h: state_index(grid, (c, h)) in goals,
                        lambda c, h: field[state_index(grid, (c, h))],
                        table, hard, soft, self.window, DROPOFF if o.carrying else PICKUP)
        crossed = table.crossed()
        if here not in crossed:
            return []
        return plan(grid, i, here, o.heading, ready_at,
                    lambda c, h: c not in crossed and c != here,
                    lambda c, h: 0, table, hard, soft, self.window)

    # ------------------------------------------------------------------ plumbing

    def save(self, folder):
        settings = {"info": self.info, "window": self.window, "restarts": self.restarts}
        (folder / "planner.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    def load(self, folder):
        saved = json.loads((folder / "planner.json").read_text(encoding="utf-8"))
        self.configure(**saved)


def _integer(value, name, least):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} wants an integer, got '{value}'") from None
    if number < least:
        raise ValueError(f"{name} must be at least {least}, got {number}")
    return number
