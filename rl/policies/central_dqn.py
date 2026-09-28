"""Central agent: ONE network decides every robot's move, from every robot's observation plus the map.

Factored DQN (VDN): the net outputs one Q-value per (robot, action). The team's value of a joint action is the
SUM of the chosen Q-values, trained on the team reward (the sum of the robots' rewards). Each robot then just takes
its own best action. Registered as CentralDQN (a Policy, no shared()/independent() wrapper).

Early in training each robot follows GreedyRule with a fading probability, so the memory box holds real deliveries
from the start (DQN is off-policy: it can learn from a teacher's plays). The net's shape depends on the number of
robots and the map size, so a checkpoint only loads into the same --robots and --size.
"""
import random

import torch
import torch.nn.functional as F
from torch import nn

from env.layout import DOCK, SHELF
from env.warehouse import HUMAN, N_ACTIONS
from rl.policies.greedy import GreedyRule
from rl.policy import Policy

GAMMA, LR, BATCH, MEMORY, HIDDEN = 0.97, 5e-4, 128, 100_000, 256
UPDATE_EVERY, TARGET_EVERY = 4, 1000            # env steps between updates; updates between target-net copies
EXPERT_STEPS = 100_000                          # the teacher's share fades from 90% to 0 over this many steps
EPS_START, EPS_END, EPS_STEPS = 0.5, 0.05, 300_000
VIEW, RAY_TYPES = 5, HUMAN + 1                  # ray range, and the number of ray types (0 = nothing .. 6 = human)
ROBOT_DIM = 50                                  # length of robot_features()


def one_hot(k, n):
    return [float(i == k) for i in range(n)]


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, n_out))


class CentralDQN(Policy):
    OBS = ("position", "heading", "task", "rays", "crate", "robot")

    def __init__(self, spec, seed=0):
        super().__init__(spec, seed)
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.n = spec.n_robots
        self.expert = GreedyRule(spec, seed)
        self.set_grid()
        n_in = self.n * ROBOT_DIM + self.grid.numel()
        self.net, self.target = mlp(n_in, self.n * N_ACTIONS), mlp(n_in, self.n * N_ACTIONS)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.memory, self.steps, self.updates = [], 0, 0

    def reset(self):
        self.expert.spec = self.spec
        self.set_grid()

    def act(self, obs):
        with torch.no_grad():
            best = self.q(self.net, self.features(obs), self.grid).argmax(-1).tolist()
        if not self.training:
            return best
        p_expert = max(0.0, 1 - self.steps / EXPERT_STEPS)
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * self.steps / EPS_STEPS)
        actions = []
        for o, b in zip(obs, best):
            if self.rng.random() < p_expert:
                actions.append(self.expert.act_one(o))
            elif self.rng.random() < eps:
                actions.append(self.rng.randrange(N_ACTIONS))
            else:
                actions.append(b)
        return actions

    def update(self, obs, actions, rewards, next_obs, done):
        # done is only a time limit, never a real end: always bootstrap from next_obs.
        item = (self.grid, self.features(obs), torch.tensor(actions), torch.tensor(float(sum(rewards))),
                self.features(next_obs))
        if len(self.memory) < MEMORY:
            self.memory.append(item)
        else:
            self.memory[self.steps % MEMORY] = item
        self.steps += 1
        if self.steps % UPDATE_EVERY or len(self.memory) < BATCH:
            return
        grid, s, a, r, s2 = (torch.stack(part) for part in zip(*self.rng.sample(self.memory, BATCH)))
        team_q = self.q(self.net, s, grid).gather(-1, a.unsqueeze(-1)).squeeze(-1).sum(-1)
        with torch.no_grad():
            goal = r + GAMMA * self.q(self.target, s2, grid).max(-1).values.sum(-1)
        loss = F.smooth_l1_loss(team_q, goal)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.updates += 1
        if self.updates % TARGET_EVERY == 0:
            self.target.load_state_dict(self.net.state_dict())

    def save(self, folder):
        torch.save(self.net.state_dict(), folder / "net.pt")

    def load(self, folder):
        self.net.load_state_dict(torch.load(folder / "net.pt", weights_only=True))
        self.target.load_state_dict(self.net.state_dict())

    def q(self, net, x, grid):
        """Q-values shaped [..., robot, action] for robot features x and the map."""
        return net(torch.cat([x, grid], -1)).view(*x.shape[:-1], self.n, N_ACTIONS)

    def set_grid(self):
        """The map as two 0/1 channels (shelf, dock), flattened. The whole map is the same for every robot."""
        rows = self.spec.grid
        self.grid = torch.tensor([[float(c == kind) for row in rows for c in row] for kind in (SHELF, DOCK)]).flatten()

    def features(self, obs):
        return torch.tensor([v for o in obs for v in self.robot_features(o)])

    def robot_features(self, o):
        w, h = len(self.spec.grid[0]), len(self.spec.grid)
        v = [o.x / w, o.y / h, o.target_x / w, o.target_y / h, o.carrying, o.has_task,
             o.crate_weight / 3, o.crate_volume / 3, o.priority / 3, o.task_age / 100,
             o.ready, o.period / 2, o.max_weight / 3, o.max_volume / 3]
        v += one_hot(o.heading, 4)
        for dist, kind in ((o.front_d, o.front_t), (o.right_d, o.right_t), (o.back_d, o.back_t), (o.left_d, o.left_t)):
            v += [dist / VIEW] + one_hot(kind, RAY_TYPES)
        return v
