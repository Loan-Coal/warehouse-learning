"""How well a fleet moves, whatever policy drives it: the columns behind "never undo a move".

Deliveries say how much got done; these say how cleanly. All counts are per episode and are SUMS,
so many episodes pool into one rate with rates() instead of an average of noisy ratios. Only a
robot that has a job and whose action the env carried out (env.ready) is measured: an idle robot
has no right way to move, and a slow robot's off-step action never happened.

A job is measured in the robot's OWN actions against the fewest it could have used, so a
half-speed robot is judged on what it did, not on its speed. Jobs still running when the episode
ends are left out, which drops the longest ones: read job_ratio as slightly optimistic.

Rulers, not rewards. "Crowded" is a proxy for a deliberate yield, so putting any of this in a
reward would teach the fleet to stand near each other.
"""
from array import array
from collections import deque

from env.layout import is_walkable
from env.warehouse import DROPOFF, HEADINGS, PICKUP, TURN_LEFT, TURN_RIGHT
from rl.nav import distance_field

CROWD_RADIUS = 2            # a robot or human this close to the next path cell may justify a yield
FIELD_CACHE = 5_000         # (map, target) fields of ~7 KB each at 21x21: ~35 MB at most
MAP_CACHE = 64              # state graphs, one per map
_FACING, _GRAPHS = {}, {}

COUNTS = ("acting_steps", "undone_turns", "undone_turns_crowded", "reversals",
          "job_steps", "job_optimal")


class EpisodeMetrics:
    """Call reset(env) after env.reset, then observe(env, actions) / record(env) around env.step."""

    def reset(self, env):
        self.counts = dict.fromkeys(COUNTS, 0)
        self.last_turn = [None] * env.n_robots      # last acting action, if it was a turn
        self.last_move = [None] * env.n_robots      # (from, to) of the last move
        self.jobs = [None] * env.n_robots           # [task, actions taken on it, fewest possible]
        self._track_jobs(env)

    def observe(self, env, actions):
        """The state before the step: what was asked, of whom, from where."""
        self.actions = list(actions)
        self.before = [(env.ready(i), r.task is not None, r.pos, _target(r))
                       for i, r in enumerate(env.robots)]
        self.crowd = [r.pos for r in env.robots] + list(env.humans)

    def record(self, env):
        for i, (ready, busy, start, target) in enumerate(self.before):
            if not busy:
                self.last_turn[i] = self.last_move[i] = None
                continue
            if not ready:
                continue
            self.counts["acting_steps"] += 1
            if self.jobs[i] is not None:
                self.jobs[i][1] += 1
            self._turn(env, i, start, target)
            self._move(i, start, env.robots[i].pos)
        self._track_jobs(env)

    def totals(self):
        return dict(self.counts)

    # ------------------------------------------------------------------ the three measures

    def _turn(self, env, i, start, target):
        action = self.actions[i]
        if action not in (TURN_LEFT, TURN_RIGHT):
            self.last_turn[i] = None
            return
        if self.last_turn[i] is not None and self.last_turn[i] != action:
            self.counts["undone_turns"] += 1
            others = self.crowd[:i] + self.crowd[i + 1:]
            if _crowded(env.grid, start, target, others):
                self.counts["undone_turns_crowded"] += 1
            action = None       # a third turn back would undo nothing: it starts a new pair
        self.last_turn[i] = action

    def _move(self, i, start, end):
        if self.actions[i] in (PICKUP, DROPOFF):
            # Backing out of a dead-end shelf the way it came in is the only way out, not an undo.
            self.last_move[i] = None
            return
        if end == start:
            return
        if self.last_move[i] == (end, start):
            self.counts["reversals"] += 1
        self.last_move[i] = (start, end)

    def _track_jobs(self, env):
        """A job ends when the robot's task object changes: the env hands out the next job in the
        same step as the delivery, so waiting for task to become None would miss most of them."""
        for i, robot in enumerate(env.robots):
            job = self.jobs[i]
            if job is not None and robot.task is job[0]:
                continue
            if job is not None and job[2] is not None:
                self.counts["job_steps"] += job[1]
                self.counts["job_optimal"] += job[2]
            self.jobs[i] = None
            if robot.task is not None:
                best = job_optimum(env.grid, (robot.pos, robot.heading),
                                   robot.task.pickup, robot.task.dropoff)
                self.jobs[i] = [robot.task, 0, best]


def rates(rows):
    """Pooled rates over episode rows: sums first, one division last. None where nothing counted."""
    # .get: rows written before a count existed count as zero.
    total = {key: sum(row.get(key, 0) for row in rows) for key in COUNTS}

    def ratio(a, b):
        return total[a] / total[b] if total[b] else None

    return {"job_ratio": ratio("job_steps", "job_optimal"),
            "undone_rate": ratio("undone_turns", "acting_steps"),
            "crowded_share": ratio("undone_turns_crowded", "undone_turns"),
            "reversal_rate": ratio("reversals", "acting_steps")}


def format_rates(rows):
    """The goal numbers, for a printed progress line."""
    pooled = rates(rows)
    shown = {key: "-" if pooled[key] is None else f"{pooled[key]:.2f}"
             for key in ("job_ratio", "undone_rate", "reversal_rate")}
    return (f"job ratio {shown['job_ratio']}  undone {shown['undone_rate']}  "
            f"reversal {shown['reversal_rate']}")


# ---------------------------------------------------------------------- the optimum


def job_optimum(grid, state, pickup, dropoff):
    """Fewest actions for one robot alone to do a job from (pos, heading): turns included.

    Pickup and dropoff need the robot to FACE the cell, so the search runs over (x, y, heading).
    Every action is reversible (a turn by the other turn, a move by the opposite move), so the
    distance from a state to "facing the dropoff" is one cached field per target, while the leg
    to the pickup is searched from the start until every pickup-facing state is found.
    Returns None if the job cannot be done on this map.
    """
    graph = _graph(grid)
    goals = _facing_states(grid, pickup)
    to_drop = _facing_field(grid, dropoff)
    reached = _search(graph, [_index(grid, state)], stop_when=goals)
    legs = [reached[g] + to_drop[g] for g in goals if reached[g] >= 0 and to_drop[g] >= 0]
    return min(legs) + 2 if legs else None      # + the pickup and the dropoff themselves


def _facing_field(grid, cell):
    key = (grid, cell)
    if key not in _FACING:
        if len(_FACING) >= FIELD_CACHE:
            _FACING.clear()
        # Stored as a C int array: a quarter the memory of a list of Python ints.
        _FACING[key] = array("i", _search(_graph(grid), _facing_states(grid, cell)))
    return _FACING[key]


def _index(grid, state):
    """(pos, heading) -> one int, so a search runs on flat lists rather than tuples in dicts."""
    (x, y), heading = state
    return (y * len(grid[0]) + x) * 4 + heading


def _facing_states(grid, cell):
    """State indices from which a robot faces `cell`."""
    out = []
    for heading, (dx, dy) in enumerate(HEADINGS):
        pos = (cell[0] - dx, cell[1] - dy)
        if is_walkable(grid, *pos):
            out.append(_index(grid, (pos, heading)))
    return out


def _graph(grid):
    """Each state's neighbours: the two turns, and forward / backward where walkable.

    Built once per map; one map serves a whole round of --envs episodes.
    """
    if grid not in _GRAPHS:
        if len(_GRAPHS) >= MAP_CACHE:
            _GRAPHS.clear()
            _FACING.clear()     # its fields belong to maps that are now long gone
        width, height = len(grid[0]), len(grid)
        graph = [()] * (width * height * 4)
        for y in range(height):
            for x in range(width):
                if not is_walkable(grid, x, y):
                    continue
                for heading, (dx, dy) in enumerate(HEADINGS):
                    out = [_index(grid, ((x, y), (heading + 1) % 4)),
                           _index(grid, ((x, y), (heading - 1) % 4))]
                    for sign in (1, -1):
                        pos = (x + sign * dx, y + sign * dy)
                        if is_walkable(grid, *pos):
                            out.append(_index(grid, (pos, heading)))
                    graph[_index(grid, ((x, y), heading))] = tuple(out)
        _GRAPHS[grid] = graph
    return _GRAPHS[grid]


def _search(graph, sources, stop_when=None):
    """Breadth-first steps from the nearest source to every state; -1 where not reached.

    stop_when: stop as soon as all of these states have been reached.
    """
    steps = [-1] * len(graph)
    for s in sources:
        steps[s] = 0
    frontier = deque(sources)
    left = None if stop_when is None else {g for g in stop_when if steps[g] < 0}
    while frontier and (left is None or left):
        state = frontier.popleft()
        here = steps[state] + 1
        for nxt in graph[state]:
            if steps[nxt] < 0:
                steps[nxt] = here
                frontier.append(nxt)
                if left is not None:
                    left.discard(nxt)
    return steps


def _target(robot):
    if robot.task is None:
        return None
    return robot.task.dropoff if robot.crate is not None else robot.task.pickup


def _crowded(grid, pos, target, others):
    """Is another robot or human near the cell(s) this robot would step to next?"""
    field = distance_field(grid, target)
    here = field.at(*pos)
    ahead = [(pos[0] + dx, pos[1] + dy) for dx, dy in HEADINGS]
    ahead = [c for c in ahead if field.at(*c) == here - 1 and here > 0] or [pos]
    return any(abs(ox - cx) + abs(oy - cy) <= CROWD_RADIUS
               for ox, oy in others for cx, cy in ahead)
