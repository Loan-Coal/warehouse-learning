"""What every algorithm here shows its network, and why each number is in it.

Shared on purpose. If central_dqn and ppo built their own inputs, a comparison between them
would be a comparison of representations rather than of algorithms, and the more interesting
question -- which learner suits this problem -- would be unanswerable.

Nothing here is normalised by a running statistic. Every value is divided by a constant known
in advance (the map width, the ray range, the priority ceiling) or is a one-hot, which keeps the
feature builder pure: no state to keep in a checkpoint, and no chance of a policy behaving
differently on its first episode than on its thousandth.

Scale independence is a property, not an accident. Positions are divided by the map size and the
rays are local, so the row is the same width for a 4-robot 11x9 warehouse as for a 6-robot
21x21 one, and the pooled fleet summary the networks add (see the policies) is a mean rather
than a concatenation. That is what lets one checkpoint play a scenario it never trained on.
"""
import torch

from env.observation import HEADINGS, obs_type
from env.warehouse import HUMAN, ROBOT, VALUE_DECAY_STEPS, VALUE_FLOOR
from rl.nav import UNREACHABLE, distance_field

# VIEW mirrors Warehouse(view_range=5). Nothing in the repo passes another value and EnvSpec does
# not carry it, so a scenario that changed it would push dist / VIEW quietly past 1.0.
VIEW, RAY_TYPES = 5, HUMAN + 1          # ray range, and the ray types (0 = nothing .. 6 = human)


def one_hot(k, n):
    return [float(i == k) for i in range(n)]


class Features:
    """Observations to network rows, for whichever map is currently installed.

    use(grid) before building rows for a new map: the shortest-path features and the shaping
    potential are both relative to it.
    """

    def __init__(self, obs_features, use_bfs=True):
        self.obs_features = tuple(obs_features)
        self.use_bfs = use_bfs
        self.grid, self.w, self.h = (), 1, 1
        # Measured, not asserted: a blank observation has no task, so the shortest-path branch
        # returns its constant and the (still unset) grid is never touched.
        blank = obs_type(self.obs_features)
        self.width = len(self.row(blank(*([0] * len(blank._fields)))))

    def use(self, grid):
        self.grid, self.w, self.h = grid, len(grid[0]), len(grid)

    def fleet(self, obs):
        """[robots, width] for one env."""
        return torch.tensor([self.row(o) for o in obs], dtype=torch.float32)

    def batch(self, obs_per_env):
        """[envs, robots, width] for several envs, which is the shape the networks want.

        Not a speed-up: measured against stacking per-env fleet() calls it is 1.00x, because 83%
        of the time is the Python loop in row() and only the rest is building the tensor. At 16
        envs of 6 robots that loop costs ~950 us per vectorised step, which makes it the largest
        single cost in the whole loop now that the network calls are batched. Turning row() into
        array arithmetic is the next real optimisation if one is needed.
        """
        return torch.tensor([[self.row(o) for o in one] for one in obs_per_env],
                            dtype=torch.float32)

    def row(self, o):
        """One robot's input. Every feature here is one that some action can change.

        Crate size, robot capacity, raw task age, Manhattan distance and a per-ray "nothing in
        range" flag were all dropped as unactionable or redundant; TestDroppedFeatures in
        tests/test_central_dqn.py holds the env properties that make them so, so re-adding one
        means first breaking a test that says why not.
        """
        dx, dy = o.target_x - o.x, o.target_y - o.y
        v = [o.x / self.w, o.y / self.h, dx / self.w, dy / self.h,
             float(o.carrying), float(o.has_task), float(faces_target(o)),
             o.priority / 3, task_value(o) / 3,
             float(o.ready), float(max(o.period - 1, 0))]
        v += one_hot(o.heading, 4)
        for dist, kind in ((o.front_d, o.front_t), (o.right_d, o.right_t),
                           (o.back_d, o.back_t), (o.left_d, o.left_t)):
            # Moving into a robot or a human costs R_BLOCKED, by far the biggest drain on the
            # score, and it is the one thing the static distance field cannot warn about. The
            # conjunction "this type AND this range" is spread across a one-hot and a distance,
            # so the single fact that matters gets its own feature.
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
        """Phi(s) for one robot: nearer to its target is better, and facing it is better still.

        A function of the state alone. That is what makes reward shaping built on it
        potential-based; it is NOT enough to make it policy-invariant unless the discounted form
        is used, and the policies here deliberately do not use it -- see their learning_rewards.
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
