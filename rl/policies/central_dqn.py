"""Central agent: ONE network decides every robot's move, from every robot's observation.

Centralized training with parameter sharing (CTDE). Every robot goes through the same trunk,
and its input carries a pooled summary of the whole fleet, so each decision is made with the
team in view. The net scores one Q-value per action for one robot, and is trained on that
robot's OWN reward: the env already hands back a per-robot reward vector (and already shares
half of it between robots through its `alpha`), so summing it into a single team number, as a
VDN would, throws away the only signal that says WHICH robot wasted its move.

Three things make the warehouse learnable at all, and each is worth knowing about:

  * A legality mask (`legal_actions`). The env silently ignores an impossible action and fines
    it, so without a mask the net spends Q-values on moves that can never happen and the
    bootstrap keeps believing in them.
  * BFS distance features. A small MLP cannot work out shortest paths through a maze from
    coordinates and rays. GreedyRule computes the same field. --policy-opt bfs=0 removes them.
  * Distance-based reward shaping. A delivery is ~50 steps of correct play away, far too rare to
    stumble into, so the net is also paid for getting closer. The bonus telescopes over any run
    of steps (see learning_rewards), so circling cannot farm it, and it is added only to what the
    net learns from: every number the env reports is untouched. --policy-opt shaping=0.

A greedy teacher plays a fading share of the moves, never dropping below EXPERT_FLOOR, so the
memory box always holds real deliveries to learn from (DQN is off-policy).

Because the net's shape depends on neither the robot count nor the map size, a checkpoint
trained on a small scenario loads straight into a bigger one.
"""
import json
import os
import random
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn

from env.layout import DIRS, access_cells, is_walkable
from env.observation import obs_type
from env.warehouse import (BACKWARD, DROPOFF, FORWARD, HEADINGS, HUMAN, N_ACTIONS, PICKUP, ROBOT,
                           TURN_LEFT, TURN_RIGHT, VALUE_DECAY_STEPS, VALUE_FLOOR)
from rl.policies.greedy import GreedyRule
from rl.policy import Policy

# The batches here are tiny (a few hundred rows through 256-wide layers), so torch spends more
# time handing work to threads than doing it. Measured on a 16-thread machine: 0.46 s/episode at
# 8 threads against 0.54 at 16 and 0.77 at 1.
torch.set_num_threads(min(8, os.cpu_count() or 1))

GAMMA, LR, BATCH, MEMORY, HIDDEN = 0.99, 5e-4, 64, 60_000, 256
UPDATE_EVERY, TARGET_EVERY = 2, 500             # env steps between updates; updates between target-net copies
EXPERT_STEPS, EXPERT_FLOOR = 60_000, 0.05       # the teacher fades over this many steps, down to this share
EPS_START, EPS_END, EPS_STEPS = 0.3, 0.05, 150_000

# Every robot runs the SAME weights, so in a one-cell-wide aisle two robots meeting head-on both
# want the same cell, are both blocked, and mirror each other for the rest of the episode. A small
# random legal move breaks the tie. Like GreedyRule's own epsilon (rl/policies/greedy.py) this is
# part of the rule rather than exploration, so it stays on in evaluation; it is drawn from a
# separate, per-episode stream so evaluating never disturbs the training trajectory.
EVAL_EPS = 0.05
VIEW, RAY_TYPES = 5, HUMAN + 1                  # ray range, and the number of ray types (0 = nothing .. 6 = human)
NEG = -1e9                                      # Q of an impossible action: beaten by any real one, never -inf

# Rewards reach +30 for a delivery, which would sit far outside the Huber loss's quadratic part
# and give every delivery the same gradient as a mildly wrong idle step. Scaling puts the whole
# normal range of Q inside it. It applies only to what the net learns from, never to a metric.
REWARD_SCALE = 0.1
SHAPE = 0.3                                     # weight of the distance potential, in env reward units
GRAD_CLIP = 10.0


def one_hot(k, n):
    return [float(i == k) for i in range(n)]


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, n_out))


def legal_actions(grid, o):
    """Which of the 6 actions the env would actually carry out, for one robot.

    The env never tells a policy this: it silently ignores an impossible action and fines it
    R_INVALID, reporting only a count. PICKUP/DROPOFF need the robot to FACE its target cell,
    not merely stand beside it (env/warehouse.py _pickup/_dropoff); obs.target_x/target_y is
    already the pickup cell while empty and the dropoff cell while carrying, so one comparison
    covers both. Turning is never invalid, so at least one action is always legal.
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


def distances_to(grid, goals):
    """Steps from every walkable cell to the nearest goal cell, as {(x, y): steps}."""
    dist = {cell: 0 for cell in goals}
    frontier = deque(dist)
    while frontier:
        x, y = frontier.popleft()
        for dx, dy in DIRS:
            nxt = (x + dx, y + dy)
            if nxt not in dist and is_walkable(grid, *nxt):
                dist[nxt] = dist[(x, y)] + 1
                frontier.append(nxt)
    return dist


def _flag(value, name):
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} wants 0 or 1, got '{value}'")


class CentralDQN(Policy):
    OBS = ("position", "heading", "task", "rays", "crate", "robot")

    def __init__(self, spec, seed=0):
        super().__init__(spec, seed)
        self.rng = random.Random(seed)              # choosing actions while training
        self.replay_rng = random.Random(seed + 1)   # sampling the memory, so the two never interleave
        self.eval_rng = random.Random(seed + 2)     # reseeded per episode, see reset()
        self.expert = GreedyRule(spec, seed)
        self.use_bfs, self.use_shaping = True, True
        self.memory, self.steps, self.updates, self.loss_ema = [], 0, 0, 0.0
        self._fields, self._field_grid = {}, None
        self._measure_map()
        self._build()

    def configure(self, bfs=None, shaping=None, **unknown):
        if unknown:
            raise ValueError(f"unknown setting {', '.join(sorted(unknown))}; this policy takes bfs, shaping")
        if bfs is not None:
            self.use_bfs = _flag(bfs, "bfs")
        if shaping is not None:
            self.use_shaping = _flag(shaping, "shaping")
        self._build()       # bfs changes how wide a robot's feature vector is

    def _build(self):
        """(Re)create the net. The input is one robot's features plus the fleet's mean of them."""
        torch.manual_seed(self.seed)
        self.robot_dim = len(self.robot_features(self._zero_obs()))
        self.net, self.target = mlp(2 * self.robot_dim, N_ACTIONS), mlp(2 * self.robot_dim, N_ACTIONS)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)

    def _zero_obs(self):
        """A blank observation, only to measure how long a feature vector is."""
        cls = obs_type(tuple(self.spec.obs_features))
        return cls(*([0] * len(cls._fields)))

    def _measure_map(self):
        self.w, self.h = len(self.spec.grid[0]), len(self.spec.grid)

    def reset(self):
        self.expert.spec = self.spec
        self._measure_map()
        if not self.training:
            # Evaluation gets its own fresh stream, so measuring the policy leaves the training
            # trajectory bit-for-bit unchanged and every eval of a given net is reproducible.
            self.eval_rng = random.Random(self.seed)

    # ------------------------------------------------------------------ acting

    def act(self, obs):
        mask = self.masks(obs)
        with torch.no_grad():
            best = self.q(self.net, self.features(obs)).masked_fill(~mask, NEG).argmax(-1).tolist()
        rng = self.rng if self.training else self.eval_rng
        p_expert = self.p_expert() if self.training else 0.0
        eps = self.epsilon() if self.training else EVAL_EPS
        actions = []
        for o, row, b in zip(obs, mask, best):
            if p_expert and rng.random() < p_expert:
                # The teacher breaks ties with a random move of its own, which may be a wall.
                taught = self.expert.act_one(o)
                actions.append(taught if row[taught] else self.random_legal(rng, row))
            elif rng.random() < eps:
                actions.append(self.random_legal(rng, row))
            else:
                actions.append(b)
        return actions

    def p_expert(self):
        return max(EXPERT_FLOOR, 1 - self.steps / EXPERT_STEPS)

    def epsilon(self):
        return max(EPS_END, EPS_START - (EPS_START - EPS_END) * self.steps / EPS_STEPS)

    def masks(self, obs):
        """[n_robots, N_ACTIONS] of bools: which actions the env would carry out right now."""
        return torch.tensor([legal_actions(self.spec.grid, o) for o in obs], dtype=torch.bool)

    @staticmethod
    def random_legal(rng, row):
        """Pick among the actions that do something, not among all six."""
        return rng.choice([a for a in range(N_ACTIONS) if row[a]])

    def q(self, net, x):
        """Q-values [..., robot, action] for robot features x [..., robot, robot_dim].

        Each robot is scored by the same weights, from its own features and the fleet's mean of
        them: centralized (it sees the team), permutation-invariant, and independent of how many
        robots there are or how big the map is.
        """
        return net(torch.cat([x, x.mean(-2, keepdim=True).expand_as(x)], -1))

    # ------------------------------------------------------------------ learning

    def update(self, obs, actions, rewards, next_obs, done):
        # done is only a time limit, never a real end: always bootstrap from next_obs.
        # The next state's mask is stored, not recomputed at sample time: the memory keeps feature
        # tensors rather than observations, and with --new-map the map differs between transitions.
        # `live` drops the robots whose action the env ignored this step (slow robots act every
        # other step, but the loop still hands us an action for them).
        item = (self.features(obs), torch.tensor(actions),
                torch.tensor(self.learning_rewards(obs, rewards, next_obs)),
                self.features(next_obs), self.masks(next_obs),
                torch.tensor([float(o.ready) for o in obs]))
        if len(self.memory) < MEMORY:
            self.memory.append(item)
        else:
            self.memory[self.steps % MEMORY] = item
        self.steps += 1
        if self.steps % UPDATE_EVERY or len(self.memory) < BATCH:
            return
        s, a, r, s2, m2, live = (torch.stack(part)
                                 for part in zip(*self.replay_rng.sample(self.memory, BATCH)))
        q = self.q(self.net, s).gather(-1, a.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            goal = r + GAMMA * self.q(self.target, s2).masked_fill(~m2, NEG).max(-1).values
        per_robot = F.smooth_l1_loss(q, goal, reduction="none")
        loss = (per_robot * live).sum() / live.sum().clamp(min=1)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP)
        self.opt.step()
        self.updates += 1
        self.loss_ema = 0.99 * self.loss_ema + 0.01 * loss.item()
        if self.updates % TARGET_EVERY == 0:
            self.target.load_state_dict(self.net.state_dict())

    def learning_rewards(self, obs, rewards, next_obs):
        """The per-robot reward the NET learns from: scaled, and optionally shaped.

        What the env reports, and therefore every number in train_log.csv and rl.bench, is
        untouched by this.
        """
        out = []
        for o, r, o2 in zip(obs, rewards, next_obs):
            if self.use_shaping and not self.target_changed(o, o2):
                # Ng, Harada & Russell's exact form is GAMMA * Phi(s') - Phi(s), which leaves the
                # optimal policy provably unchanged. It is wrong here: Phi is negative, so that
                # form pays a standing-still robot (GAMMA - 1) * Phi = +0.01 per cell of distance
                # every step, which beats the -0.05 step cost and buys loitering far from the
                # target. The plain difference telescopes to Phi(end) - Phi(start) over any run of
                # steps, so no loop can farm it and standing still is worth exactly nothing.
                r += SHAPE * (self.potential(o2) - self.potential(o))
            out.append(REWARD_SCALE * r)
        return out

    @staticmethod
    def target_changed(o, o2):
        """True when a pickup or a delivery swapped the target, so the potential jumps."""
        return ((o.has_task, o.carrying, o.target_x, o.target_y)
                != (o2.has_task, o2.carrying, o2.target_x, o2.target_y))

    def potential(self, o):
        """Phi(s) for one robot: nearer to its target is better, and facing it is better still.

        A function of the state alone, which is what makes the shaping potential-based and so
        leaves the optimal policy unchanged.
        """
        if not o.has_task:
            return 0.0
        here = self.distance_field((o.target_x, o.target_y)).get((o.x, o.y))
        return 0.0 if here is None else -float(here) + 0.5 * float(self.faces_target(o))

    # ------------------------------------------------------------------ features

    def features(self, obs):
        return torch.tensor([self.robot_features(o) for o in obs], dtype=torch.float32)

    def robot_features(self, o):
        dx, dy = o.target_x - o.x, o.target_y - o.y
        v = [o.x / self.w, o.y / self.h,
             dx / self.w, dy / self.h, (abs(dx) + abs(dy)) / (self.w + self.h),
             float(o.carrying), float(o.has_task), float(self.faces_target(o)),
             o.crate_weight / 3, o.crate_volume / 3,
             o.priority / 3, self.task_value(o) / 3,
             # Raw task_age drifts upward exactly when the agent is failing, so the net would be
             # learning from a moving target. What the reward actually depends on is the value.
             min(o.task_age, 300) / 100,
             float(o.ready), float(max(o.period - 1, 0)), o.max_weight / 3, o.max_volume / 3]
        v += one_hot(o.heading, 4)
        for dist, kind in ((o.front_d, o.front_t), (o.right_d, o.right_t),
                           (o.back_d, o.back_t), (o.left_d, o.left_t)):
            # Moving into a robot or a human costs R_BLOCKED, by far the biggest drain on the
            # score, and it is the one thing the static distance field cannot warn about. The
            # conjunction "this type AND this range" is spread across a one-hot and a distance,
            # so the single fact that matters gets its own feature.
            occupied = float(dist == 1 and kind in (ROBOT, HUMAN))
            v += [dist / VIEW, float(dist == 0), occupied] + one_hot(kind, RAY_TYPES)
        if self.use_bfs:
            near, deltas = self.distance_features(o)
            v += [near] + deltas
        return v

    @staticmethod
    def task_value(o):
        """What a delivery would pay right now (env/warehouse.py _value): stationary, unlike age."""
        if not o.has_task:
            return 0.0
        return o.priority * max(VALUE_FLOOR, 1 - o.task_age / VALUE_DECAY_STEPS)

    @staticmethod
    def faces_target(o):
        dx, dy = HEADINGS[o.heading]
        return bool(o.has_task) and (o.x + dx, o.y + dy) == (o.target_x, o.target_y)

    def distance_features(self, o):
        """How far the target is, and which of front/right/back/left steps closer to it.

        Always returns the same number of values, so the feature vector keeps its width.
        """
        if not o.has_task:
            return 1.0, [0.0] * 4
        field = self.distance_field((o.target_x, o.target_y))
        here = field.get((o.x, o.y))
        if here is None:
            return 1.0, [0.0] * 4
        deltas = []
        for turn in range(4):
            dx, dy = HEADINGS[(o.heading + turn) % 4]
            there = field.get((o.x + dx, o.y + dy))
            deltas.append(0.0 if there is None else float(here - there))    # +1 = a step closer
        return min(here / (self.w + self.h), 1.0), deltas

    def distance_field(self, cell):
        """Steps to `cell` from anywhere, cached per target and rebuilt only when the map changes."""
        if self._field_grid is not self.spec.grid:
            self._fields, self._field_grid = {}, self.spec.grid
        if cell not in self._fields:
            self._fields[cell] = distances_to(self.spec.grid, access_cells(self.spec.grid, cell))
        return self._fields[cell]

    # ------------------------------------------------------------------ plumbing

    def stats(self):
        return {"epsilon": self.epsilon(), "p_expert": self.p_expert(), "loss": self.loss_ema}

    def save(self, folder):
        """Everything needed to carry on training, not just to replay.

        Without the target net, the optimizer and `steps`, resuming with --train would restart
        the teacher at 100% and throw away Adam's momentum. The replay memory is deliberately
        left out: it is hundreds of megabytes, and the teacher refills it within an episode.
        """
        torch.save(self.net.state_dict(), folder / "net.pt")
        torch.save(self.target.state_dict(), folder / "target.pt")
        torch.save(self.opt.state_dict(), folder / "opt.pt")
        state = {"bfs": self.use_bfs, "shaping": self.use_shaping, "steps": self.steps,
                 "updates": self.updates, "loss_ema": self.loss_ema}
        (folder / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load(self, folder):
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        if (state["bfs"], state["shaping"]) != (self.use_bfs, self.use_shaping):
            # The checkpoint's own settings win: they decide how wide the saved weights are.
            self.use_bfs, self.use_shaping = state["bfs"], state["shaping"]
            self._build()
        self.net.load_state_dict(torch.load(folder / "net.pt", weights_only=True))
        self.target.load_state_dict(torch.load(folder / "target.pt", weights_only=True))
        self.opt.load_state_dict(torch.load(folder / "opt.pt", weights_only=True))
        self.steps, self.updates, self.loss_ema = state["steps"], state["updates"], state["loss_ema"]
