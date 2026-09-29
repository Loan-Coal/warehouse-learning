"""Central agent: ONE network decides every robot's move, from every robot's observation.

Centralized training with parameter sharing (CTDE). Every robot goes through the same weights,
and its input carries a pooled summary of the whole fleet, so each decision is made with the
team in view. The net scores one Q-value per action for one robot and is trained on that robot's
OWN reward: the env already hands back a per-robot reward vector (and already shares half of it
between robots through its `alpha`, so sum(rewards) == sum(own) exactly). Summing it again into
one team number, as a VDN would, therefore adds no information and destroys the only signal that
says WHICH robot wasted its move.

Three things make the warehouse learnable at all. Each is argued where it is implemented:
the legality mask and the shortest-path features in rl/nav.py, and the reward shaping in
learning_rewards below. Both feature groups have an ablation switch, --policy-opt bfs=0 and
--policy-opt shaping=0.

A greedy teacher plays a fading share of the moves, never dropping below EXPERT_FLOOR, so the
memory always holds real deliveries to learn from (DQN is off-policy).

Because the net's shape depends on neither the robot count nor the map size, a checkpoint
trained on a small scenario loads straight into a bigger one.
"""
import json
import random

import torch
import torch.nn.functional as F
from torch import nn

from env.warehouse import N_ACTIONS
from rl.features import Features, target_changed
from rl.nav import legal_actions
from rl.policies.greedy import GreedyRule
from rl.policy import Policy, flag

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
NEG = -1e9                                      # Q of an impossible action: beaten by any real one, never -inf

# Rewards reach +30 for a delivery, which would sit far outside the Huber loss's quadratic part
# and give every delivery the same gradient as a mildly wrong idle step. Scaling puts the whole
# normal range of Q inside it. It applies only to what the net learns from, never to a metric.
REWARD_SCALE = 0.1
SHAPE = 0.3                                     # weight of the distance potential, in env reward units
GRAD_CLIP = 10.0


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, n_out))


class CentralDQN(Policy):
    OBS = ("position", "heading", "task", "rays", "crate", "robot")

    def __init__(self, spec, seed=0):
        super().__init__(spec, seed)
        self.rng = random.Random(seed)              # choosing actions while training
        self.replay_rng = random.Random(seed + 1)   # sampling the memory, so the two never interleave
        self.eval_rng = random.Random(seed)         # replaced every eval episode, see reset()
        self.expert = GreedyRule(spec, seed)
        self.use_shaping = True
        self.memory, self.steps, self.updates, self.loss_ema = [], 0, 0, 0.0
        self.inputs = Features(spec.obs_features)
        self.inputs.use(spec.grid)
        self._build()

    def configure(self, bfs=None, shaping=None, **unknown):
        if unknown:
            raise ValueError(f"unknown setting {', '.join(sorted(unknown))}; this policy takes bfs, shaping")
        if bfs is not None and flag(bfs, "bfs") != self.use_bfs:
            self.inputs = Features(self.spec.obs_features, use_bfs=flag(bfs, "bfs"))
            self.inputs.use(self.spec.grid)
            self._build()   # bfs changes how wide a robot's feature row is
        if shaping is not None:
            self.use_shaping = flag(shaping, "shaping")

    @property
    def use_bfs(self):
        return self.inputs.use_bfs

    @property
    def robot_dim(self):
        return self.inputs.width

    def _build(self):
        """(Re)create the net. The input is one robot's features plus the fleet's mean of them."""
        torch.manual_seed(self.seed)
        self.net, self.target = mlp(2 * self.robot_dim, N_ACTIONS), mlp(2 * self.robot_dim, N_ACTIONS)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)

    def reset(self):
        self.expert.spec = self.spec
        self.inputs.use(self.spec.grid)
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

    target_changed = staticmethod(target_changed)

    def potential(self, o):
        return self.inputs.potential(o)

    def features(self, obs):
        return self.inputs.fleet(obs)

    # ------------------------------------------------------------------ plumbing

    def stats(self):
        return {"epsilon": self.epsilon(), "p_expert": self.p_expert(), "loss": self.loss_ema}

    def save(self, folder):
        """Everything needed to carry on training, not just to replay.

        Without the optimizer and `steps`, resuming with --train would restart the teacher at
        full strength and throw away Adam's momentum. Two things are left out on purpose: the
        replay memory, which is hundreds of megabytes and which the teacher refills within an
        episode, and the target net, which load() copies from net.pt. A resumed run therefore
        starts from a target that is fresh rather than up to TARGET_EVERY updates stale, which
        washes out within the first 500 updates and is not worth doubling the checkpoint for.
        """
        torch.save(self.net.state_dict(), folder / "net.pt")
        torch.save(self.opt.state_dict(), folder / "opt.pt")
        state = {"bfs": self.use_bfs, "shaping": self.use_shaping, "steps": self.steps,
                 "updates": self.updates, "loss_ema": self.loss_ema}
        (folder / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load(self, folder):
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        self.use_shaping = state["shaping"]
        if state["bfs"] != self.use_bfs:
            # The checkpoint's own setting wins: it decides how wide the saved weights are.
            self.inputs = Features(self.spec.obs_features, use_bfs=state["bfs"])
            self.inputs.use(self.spec.grid)
            self._build()
        self.net.load_state_dict(torch.load(folder / "net.pt", weights_only=True))
        self.target.load_state_dict(self.net.state_dict())
        self.opt.load_state_dict(torch.load(folder / "opt.pt", weights_only=True))
        self.steps, self.updates, self.loss_ema = state["steps"], state["updates"], state["loss_ema"]
