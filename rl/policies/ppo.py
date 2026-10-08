"""Centralised masked PPO: ONE agent that sees the whole fleet and picks every robot's action.

One network reads every robot's feature row at once (the list of all robots' observations, never
env.get_state()). For robot i a pairwise encoder asks, of each other robot j, a small learned
question about the pair (row_i, row_j) and keeps the strongest answer (max) and the typical one
(mean); a head turns [row_i, those answers] into 6 logits. The six actions are then sampled
independently, in parallel, and each robot is trained on its OWN advantage. Why on-policy, why the
shaping is not the textbook form, and why there are no schedules: LEARNINGS.md.

Ablations: --policy-opt central=0 (the pool holds only robots i's own rays hit, so a robot
outside every ray is invisible, as in a decentralised policy), bfs=0, shaping=0, eval_greedy=1.
The earlier mean-pool PPO (own row + fleet mean) is git tag ppo-mean.
"""
import json
import random
from typing import NamedTuple

import torch
from torch import nn

from env.observation import HEADINGS, ROBOT
from env.warehouse import N_ACTIONS
from rl.features import Features, target_changed
from rl.nav import legal_actions
from rl.policy import Policy, flag

GAMMA, LAMBDA = 0.99, 0.95
ROLLOUT = 128                   # env steps per env before an update
EPOCHS, MINIBATCHES = 4, 4
CLIP, VF_COEF, GRAD_CLIP = 0.2, 0.25, 0.5       # VF_COEF is the whole weight: 0.5 * 0.5 of the usual form
LR, ENT_COEF, HIDDEN, PAIR = 3e-4, 0.01, 256, 128     # PAIR: width of the pairwise encoder

# A constant, not a running normaliser: nothing to keep in a checkpoint (LEARNINGS.md).
REWARD_SCALE, SHAPE = 0.1, 0.3

NEG = -1e9                      # logit of an impossible action: finite, so entropy cannot go NaN


def mlp(n_in, n_out, out_gain):
    """Two hidden layers, orthogonal init, gain sqrt(2) inside and `out_gain` on the head.

    0.01 on the policy head starts it near-uniform over the legal actions, so the first rollouts
    explore instead of committing to whatever the random init liked.
    """
    net = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(),
                        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                        nn.Linear(HIDDEN, n_out))
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, 2 ** 0.5)
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(net[-1].weight, out_gain)
    return net


class PairNet(nn.Module):
    """[..., robots, width] -> [..., robots, width + 2 * PAIR]: each robot's row beside what the
    others mean to it.

    phi reads the pair [row_i, row_j] together, so its first layer can form the difference
    (where is j relative to me?) that no function of a fleet MEAN can. The answers are pooled over
    j by max ("is ANY robot threatening me?") and by mean ("how crowded is it?"). Pooling makes the
    result independent of the order and the number of robots, so a 4-robot checkpoint plays 6.

    `allow[..., i, j]` says whether robot i may know about robot j; a row with no allowed j (a
    lone robot, or central=0 with nobody in its rays) pools to zeros.
    """

    def __init__(self, width):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2 * width, PAIR), nn.ReLU(),
                                 nn.Linear(PAIR, PAIR), nn.ReLU())
        for layer in self.phi:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 2 ** 0.5)
                nn.init.zeros_(layer.bias)

    def forward(self, x, allow):
        n = x.shape[-2]
        mine = x.unsqueeze(-2).expand(*x.shape[:-1], n, x.shape[-1])        # [.., i, j, w] = row_i
        theirs = x.unsqueeze(-3).expand_as(mine)                           # [.., i, j, w] = row_j
        keep = allow.unsqueeze(-1).to(x.dtype)
        answers = self.phi(torch.cat([mine, theirs], -1)) * keep           # ReLU >= 0: masked -> 0
        mean = answers.sum(-2) / keep.sum(-2).clamp(min=1)
        return torch.cat([x, answers.amax(-2), mean], -1)


class Net(nn.Module):
    """Pairwise encoder followed by a two-hidden-layer head. One for the actor, one for the critic."""

    def __init__(self, width, n_out, out_gain):
        super().__init__()
        self.pair = PairNet(width)
        self.head = mlp(width + 2 * PAIR, n_out, out_gain)

    def forward(self, x, allow):
        return self.head(self.pair(x, allow))


def visible(obs, central):
    """[envs, robots, robots] bool: may robot i know about robot j?

    central=True: every other robot -- the central agent holds all the observations. central=False:
    only a robot one of i's own rays hit, which is what a decentralised robot could know.
    """
    n = len(obs[0])
    if central:
        return (~torch.eye(n, dtype=torch.bool)).expand(len(obs), n, n)
    out = torch.zeros(len(obs), n, n, dtype=torch.bool)
    for e, fleet in enumerate(obs):
        where = {(o.x, o.y): j for j, o in enumerate(fleet)}
        for i, o in enumerate(fleet):
            rays = ((o.front_d, o.front_t), (o.right_d, o.right_t),
                    (o.back_d, o.back_t), (o.left_d, o.left_t))
            for k, (dist, kind) in enumerate(rays):
                if kind == ROBOT:
                    dx, dy = HEADINGS[(o.heading + k) % 4]
                    j = where.get((o.x + dx * dist, o.y + dy * dist))
                    if j is not None and j != i:
                        out[e, i, j] = True
    return out


def masked_mean(values, live):
    """Mean over the rows whose action the env actually carried out (obs.ready)."""
    return (values * live).sum() / live.sum().clamp(min=1)


class Step(NamedTuple):
    """One vectorised step of a rollout; every field is a tensor with a leading [envs] axis."""
    x: torch.Tensor         # features of every robot
    mask: torch.Tensor      # which actions were legal
    acted: torch.Tensor
    logp: torch.Tensor      # of the action taken, when it was taken
    value: torch.Tensor
    after: torch.Tensor     # V(next obs), only filled where act_vec will not score it next
    reward: torch.Tensor    # what the net learns from (scaled, shaped)
    done: torch.Tensor      # the time limit was hit
    ready: torch.Tensor     # the env actually carried the action out
    allow: torch.Tensor     # [robots, robots]: which robots each robot's encoder may look at


class PPO(Policy):
    OBS = Features.OBS

    def __init__(self, spec, seed=0):
        super().__init__(spec, seed)
        self.rng = random.Random(seed)
        self.use_shaping, self.eval_greedy, self.central = True, False, True
        self.lr, self.ent = LR, ENT_COEF
        self.typed = set()      # settings given to configure(), which a checkpoint must not undo
        self.inputs = Features(spec.obs_features)
        self.inputs.use(spec.grid)
        self.steps, self.updates = 0, 0
        self.rollout, self.acting = [], None
        self.log = {"loss": 0.0, "entropy": 0.0, "clipfrac": 0.0, "explained_var": 0.0}
        self._build()

    # ------------------------------------------------------------------ setup

    def configure(self, bfs=None, shaping=None, eval_greedy=None, lr=None, ent=None, central=None,
                  **unknown):
        if unknown:
            raise ValueError(f"unknown setting {', '.join(sorted(unknown))}; this policy takes "
                             f"bfs, shaping, eval_greedy, lr, ent, central")
        # rl/cli.py configures before it loads a checkpoint; load() leaves these alone.
        self.typed |= {key for key, value in (("shaping", shaping), ("eval_greedy", eval_greedy),
                                              ("lr", lr), ("ent", ent), ("central", central))
                       if value is not None}
        if shaping is not None:
            self.use_shaping = flag(shaping, "shaping")
        if eval_greedy is not None:
            self.eval_greedy = flag(eval_greedy, "eval_greedy")
        if ent is not None:
            self.ent = _number(ent, "ent")
        if central is not None:
            self.central = flag(central, "central")
        rebuild = False
        if lr is not None and _number(lr, "lr") != self.lr:
            self.lr, rebuild = _number(lr, "lr"), True
        if bfs is not None and flag(bfs, "bfs") != self.inputs.use_bfs:
            self.inputs = Features(self.spec.obs_features, use_bfs=flag(bfs, "bfs"))
            self.inputs.use(self.spec.grid)
            rebuild = True
        if rebuild:
            self._build()

    def _build(self):
        torch.manual_seed(self.seed)
        width = self.inputs.width
        self.actor = Net(width, N_ACTIONS, 0.01)
        self.critic = Net(width, 1, 1.0)
        self.opt = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()],
                                    lr=self.lr)

    def reset(self):
        self.inputs.use(self.spec.grid)

    # ------------------------------------------------------------------ acting

    def act(self, obs):
        return self.act_vec([obs])[0]

    def act_vec(self, obs):
        """Every robot of every env through one forward pass.

        The encoder pools over the robot axis only, so envs never mix: correct, not merely faster.
        Evaluation samples rather than argmaxing -- see the deadlock argument in LEARNINGS.md.
        """
        x = self.inputs.batch(obs)
        mask = self.masks(obs)
        allow = self.reach(obs)
        with torch.no_grad():
            logits = self.actor(x, allow).masked_fill(~mask, NEG)
            value = self.critic(x, allow).squeeze(-1)
            if self.training or not self.eval_greedy:
                actions = torch.distributions.Categorical(logits=logits).sample()
            else:
                actions = logits.argmax(-1)
            logp = torch.log_softmax(logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        self.acting = (x, mask, actions, logp, value, allow)
        return actions.tolist()

    def values(self, obs):
        """[envs, robots] critic values for a list of observation lists (no grad)."""
        with torch.no_grad():
            return self.critic(self.inputs.batch(obs), self.reach(obs)).squeeze(-1)

    def sample(self, obs):
        """Sampled actions per env, like act_vec, but never records self.acting."""
        x, mask, allow = self.inputs.batch(obs), self.masks(obs), self.reach(obs)
        with torch.no_grad():
            logits = self.actor(x, allow).masked_fill(~mask, NEG)
            return torch.distributions.Categorical(logits=logits).sample().tolist()

    def reach(self, obs, central=None):
        """[envs, robots, robots] bool: which robots each robot's encoder may look at."""
        return visible(obs, self.central if central is None else central)

    def masks(self, obs):
        """[envs, robots, N_ACTIONS] of bools: which actions the env would actually carry out."""
        return torch.tensor([[legal_actions(self.spec.grid, o) for o in one] for one in obs],
                            dtype=torch.bool)

    # ------------------------------------------------------------------ learning

    def update(self, obs, actions, rewards, next_obs, done):
        self.update_vec([obs], [actions], [rewards], [next_obs], [done])

    def update_vec(self, obs, actions, rewards, next_obs, dones):
        x, mask, acted, logp, value, allow = self.acting
        if x.shape[0] != len(obs):
            raise RuntimeError(f"update_vec got {len(obs)} envs but act_vec ran on {x.shape[0]}")
        # V(next_obs) is only computed here where act_vec will not score it next: at an episode
        # boundary (next_obs is the state it ENDED on, never zeroed) and at the last step of a
        # rollout. Elsewhere _learn reuses the next step's value (LEARNINGS.md: Performance).
        last = len(self.rollout) + 1 >= ROLLOUT
        rows = list(range(len(obs))) if last else [i for i, d in enumerate(dones) if d]
        after = torch.zeros_like(value)
        if rows:
            with torch.no_grad():
                ended = [next_obs[i] for i in rows]
                after[rows] = self.critic(self.inputs.batch(ended), self.reach(ended)).squeeze(-1)
        self.rollout.append(Step(
            x, mask, acted, logp, value, after,
            torch.tensor([self.learning_rewards(*step) for step in zip(obs, rewards, next_obs)]),
            torch.tensor([float(d) for d in dones]),
            torch.tensor([[float(o.ready) for o in one] for one in obs]), allow))
        self.steps += len(obs)
        if len(self.rollout) >= ROLLOUT:
            self._learn()
            self.rollout.clear()

    def learning_rewards(self, obs, rewards, next_obs):
        """The per-robot reward the NET learns from: scaled, and optionally shaped.

        Nothing the env reports is touched, so train_log.csv and rl.bench are unaffected. The
        shaping is the PLAIN difference Phi(s') - Phi(s), deliberately not the discounted form,
        which pays a motionless robot here (LEARNINGS.md; TestShaping pins the consequences).
        """
        out = []
        for o, r, o2 in zip(obs, rewards, next_obs):
            if self.use_shaping and not target_changed(o, o2):
                r += SHAPE * (self.inputs.potential(o2) - self.inputs.potential(o))
            out.append(REWARD_SCALE * r)
        return out

    def _learn(self):
        x, mask, acted, logp, value, after, reward, done, ready, allow = (
            torch.stack(part) for part in zip(*self.rollout))
        # Fill in the bootstraps update_vec skipped: where an env did not end, it is t+1's value.
        carried = done[:-1].unsqueeze(-1).bool()
        after = torch.cat([torch.where(carried, after[:-1], value[1:]), after[-1:]])
        advantage = self._gae(reward, value, after, done)
        returns = advantage + value

        # Keep the robot axis: the encoder pools over one env's robots, so a minibatch is a
        # set of whole fleets, never of loose robots.
        flat = [t.flatten(0, 1) for t in (x, mask, acted, logp, advantage, returns, ready, allow)]
        groups = flat[0].shape[0]
        order = list(range(groups))
        size = max(1, groups // MINIBATCHES)
        for _ in range(EPOCHS):
            self.rng.shuffle(order)
            for start in range(0, groups, size):
                self._step([t[order[start:start + size]] for t in flat])
        self.updates += 1
        spread = returns.var()      # 0 means the critic is no better than predicting the mean
        self.log["explained_var"] = float(1 - (returns - value).var() / spread) if spread > 0 else 0.0

    @staticmethod
    def _gae(reward, value, after, done):
        """Advantages over a rollout that may span episode boundaries.

        `done` cuts the RECURSION but not the bootstrap: the env ran out of time rather than
        reaching a terminal state, so after[t] is a real estimate of what came next, while the
        state past the boundary belongs to an unrelated episode. Zeroing after[t] instead is the
        usual mistake and is invisible -- it teaches the policy the world ends every 400 steps.
        """
        advantage = torch.zeros_like(reward)
        running = torch.zeros_like(reward[0])
        for t in reversed(range(reward.shape[0])):
            delta = reward[t] + GAMMA * after[t] - value[t]
            running = delta + GAMMA * LAMBDA * running * (1.0 - done[t].unsqueeze(-1))
            advantage[t] = running
        return advantage

    def _step(self, batch):
        x, mask, acted, old_logp, advantage, returns, live, allow = batch
        logits = self.actor(x, allow).masked_fill(~mask, NEG)
        dist = torch.distributions.Categorical(logits=logits)
        ratio = (dist.log_prob(acted) - old_logp).exp()

        # Normalise over the rows that count, so the quarter of rows whose action the env
        # discarded (a slow robot acting on its off step) cannot shift the mean.
        counts = live > 0
        if counts.any():
            kept = advantage[counts]
            advantage = (advantage - kept.mean()) / (kept.std() + 1e-8)
        clipped = ratio.clamp(1 - CLIP, 1 + CLIP)
        policy_loss = -masked_mean(torch.min(ratio * advantage, clipped * advantage), live)
        entropy = masked_mean(dist.entropy(), live)

        # The value loss keeps every row, including the discarded ones: time still passed and a
        # reward still arrived, so the critic should predict that return. Only the POLICY has
        # nothing to learn from an action the env threw away.
        value_loss = ((self.critic(x, allow).squeeze(-1) - returns) ** 2).mean()
        loss = policy_loss + VF_COEF * value_loss - self.ent * entropy
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], GRAD_CLIP)
        self.opt.step()
        with torch.no_grad():
            self.log.update(loss=loss.item(), entropy=entropy.item(),
                            clipfrac=masked_mean(((ratio - 1).abs() > CLIP).float(), live).item())

    # ------------------------------------------------------------------ plumbing

    def stats(self):
        return dict(self.log)

    def save(self, folder):
        """Everything needed to carry on training, except the rollout in progress.

        That is at most ROLLOUT steps the next few seconds regenerate, and keeping it would make
        a checkpoint depend on where in a rollout it was taken.
        """
        torch.save(self.actor.state_dict(), folder / "actor.pt")
        torch.save(self.critic.state_dict(), folder / "critic.pt")
        torch.save(self.opt.state_dict(), folder / "opt.pt")
        state = {"bfs": self.inputs.use_bfs, "shaping": self.use_shaping,
                 "eval_greedy": self.eval_greedy, "lr": self.lr, "ent": self.ent,
                 "central": self.central, "encoder": "pair",
                 "steps": self.steps, "updates": self.updates, **self.log}
        (folder / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load(self, folder):
        """The weights, the optimiser and every setting configure() was not given.

        A typed setting wins over the saved one (--checkpoint X --policy-opt eval_greedy=1 must
        evaluate with argmax), except bfs: the checkpoint's decides how wide the weights are.
        """
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        if state.get("encoder") != "pair":
            raise ValueError("this checkpoint was trained by the mean-pool PPO, whose weights do not "
                             "fit the pairwise encoder: retrain, or check out git tag ppo-mean to play it")
        saved = {"shaping": "use_shaping", "eval_greedy": "eval_greedy", "lr": "lr", "ent": "ent",
                 "central": "central"}
        for key, attribute in saved.items():
            if key not in self.typed:
                setattr(self, attribute, state[key])
        if state["bfs"] != self.inputs.use_bfs:
            # The checkpoint's own setting wins: it decides how wide the saved weights are.
            self.inputs = Features(self.spec.obs_features, use_bfs=state["bfs"])
            self.inputs.use(self.spec.grid)
        self._build()
        self.actor.load_state_dict(torch.load(folder / "actor.pt", weights_only=True))
        self.critic.load_state_dict(torch.load(folder / "critic.pt", weights_only=True))
        self.opt.load_state_dict(torch.load(folder / "opt.pt", weights_only=True))
        for group in self.opt.param_groups:
            group["lr"] = self.lr       # the saved optimiser state brings its own lr back with it
        self.steps, self.updates = state["steps"], state["updates"]
        self.log = {key: state[key] for key in self.log}


def _number(value, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} wants a number, got '{value}'") from None

