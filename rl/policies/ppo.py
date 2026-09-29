"""Masked PPO, one set of weights for every robot, each input carrying a summary of the fleet.

IPPO/MAPPO in shape: every robot is scored by the same network from its own features plus the
fleet mean, and trained on its OWN advantage.

Why on-policy here, why the shaping takes the form it does, and why there are no schedules:
LEARNINGS.md. The short version is that six robots sharing one set of weights make a replay
buffer hold transitions whose dynamics no longer exist, and that a stochastic policy is the fix
for the deadlock parameter sharing causes -- which is why evaluation samples rather than argmaxes.

Ablations: --policy-opt bfs=0 drops the shortest-path features, shaping=0 the reward shaping,
eval_greedy=1 evaluates with argmax instead of sampling.
"""
import json
import random

import torch
from torch import nn

from env.warehouse import N_ACTIONS
from rl.features import Features, target_changed
from rl.nav import legal_actions
from rl.policy import Policy, flag

GAMMA, LAMBDA = 0.99, 0.95
ROLLOUT = 128                   # env steps per env before an update
EPOCHS, MINIBATCHES = 4, 4
CLIP, VF_COEF, GRAD_CLIP = 0.2, 0.5, 0.5
LR, ENT_COEF, HIDDEN = 3e-4, 0.01, 256

# A constant, not a running normaliser: nothing to keep in a checkpoint (LEARNINGS.md).
REWARD_SCALE, SHAPE = 0.1, 0.3

NEG = -1e9                      # logit of an impossible action: finite, so entropy cannot go NaN


def mlp(n_in, n_out, out_gain):
    """Two hidden layers, orthogonal init, gain sqrt(2) inside and `out_gain` on the head.

    0.01 on the policy head starts the policy almost uniform over the legal actions, so the first
    rollouts explore rather than commit to whatever the random init happened to like.
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


def pooled(x):
    """[..., robots, width] -> [..., robots, 2 * width]: own features beside the fleet mean.

    A mean, not a concatenation: the width is then independent of the robot count and the result
    independent of their order, which is what lets a 4-robot checkpoint play 6.
    """
    return torch.cat([x, x.mean(-2, keepdim=True).expand_as(x)], -1)


def masked_mean(values, live):
    """Mean over the rows whose action the env actually carried out (obs.ready)."""
    return (values * live).sum() / live.sum().clamp(min=1)


class PPO(Policy):
    OBS = Features.OBS

    def __init__(self, spec, seed=0):
        super().__init__(spec, seed)
        self.rng = random.Random(seed)
        self.use_shaping, self.eval_greedy = True, False
        self.lr, self.ent = LR, ENT_COEF
        self.inputs = Features(spec.obs_features)
        self.inputs.use(spec.grid)
        self.steps, self.updates = 0, 0
        self.rollout, self.acting = [], None
        self.log = {"loss": 0.0, "entropy": 0.0, "clipfrac": 0.0, "explained_var": 0.0}
        self._build()

    # ------------------------------------------------------------------ setup

    def configure(self, bfs=None, shaping=None, eval_greedy=None, lr=None, ent=None, **unknown):
        if unknown:
            raise ValueError(f"unknown setting {', '.join(sorted(unknown))}; this policy takes "
                             f"bfs, shaping, eval_greedy, lr, ent")
        if shaping is not None:
            self.use_shaping = flag(shaping, "shaping")
        if eval_greedy is not None:
            self.eval_greedy = flag(eval_greedy, "eval_greedy")
        if ent is not None:
            self.ent = _number(ent, "ent")
        rebuild = lr is not None and _number(lr, "lr") != self.lr
        if lr is not None:
            self.lr = _number(lr, "lr")
        if bfs is not None and flag(bfs, "bfs") != self.inputs.use_bfs:
            self.inputs = Features(self.spec.obs_features, use_bfs=flag(bfs, "bfs"))
            self.inputs.use(self.spec.grid)
            rebuild = True
        if rebuild:
            self._build()

    def _build(self):
        torch.manual_seed(self.seed)
        width = 2 * self.inputs.width
        self.actor = mlp(width, N_ACTIONS, 0.01)
        self.critic = mlp(width, 1, 1.0)
        self.opt = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()],
                                    lr=self.lr)

    def reset(self):
        self.inputs.use(self.spec.grid)

    # ------------------------------------------------------------------ acting

    def act(self, obs):
        return self.act_vec([obs])[0]

    def act_vec(self, obs):
        """Every robot of every env through one forward pass.

        pooled() averages over the robot axis, so each env keeps its own fleet summary and the
        envs never mix: this is correct, not merely faster.
        """
        x = self.inputs.batch(obs)
        mask = self.masks(obs)
        with torch.no_grad():
            logits = self.actor(pooled(x)).masked_fill(~mask, NEG)
            value = self.critic(pooled(x)).squeeze(-1)
            if self.training or not self.eval_greedy:
                # Sampling in evaluation too: see the deadlock argument in the module docstring.
                actions = torch.distributions.Categorical(logits=logits).sample()
            else:
                actions = logits.argmax(-1)
            logp = torch.log_softmax(logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        self.acting = (x, mask, actions, logp, value)
        return actions.tolist()

    def masks(self, obs):
        """[envs, robots, N_ACTIONS] of bools: which actions the env would actually carry out."""
        return torch.tensor([[legal_actions(self.spec.grid, o) for o in one] for one in obs],
                            dtype=torch.bool)

    # ------------------------------------------------------------------ learning

    def update(self, obs, actions, rewards, next_obs, done):
        self.update_vec([obs], [actions], [rewards], [next_obs], [done])

    def update_vec(self, obs, actions, rewards, next_obs, dones):
        x, mask, acted, logp, value = self.acting
        if x.shape[0] != len(obs):
            raise RuntimeError(f"update_vec got {len(obs)} envs but act_vec ran on {x.shape[0]}")
        with torch.no_grad():
            # next_obs is the real successor, so at an episode boundary it is the state the
            # episode ENDED on. The env only ever runs out of time, it never reaches a terminal
            # state, so that value is a real estimate of what came next and is not zeroed.
            after = self.critic(pooled(self.inputs.batch(next_obs))).squeeze(-1)
        self.rollout.append((
            x, mask, acted, logp, value, after,
            torch.tensor([self.learning_rewards(*step) for step in zip(obs, rewards, next_obs)]),
            torch.tensor([float(d) for d in dones]),
            torch.tensor([[float(o.ready) for o in one] for one in obs])))
        self.steps += len(obs)
        if len(self.rollout) >= ROLLOUT:
            self._learn()
            self.rollout.clear()

    def learning_rewards(self, obs, rewards, next_obs):
        """The per-robot reward the NET learns from: scaled, and optionally shaped.

        Every number the env reports, and so every number in train_log.csv and rl.bench, is
        untouched by this.

        The shaping is the PLAIN difference Phi(s') - Phi(s), deliberately not Ng, Harada and
        Russell's discounted form, which is policy-invariant but pays a motionless robot here.
        LEARNINGS.md has the sign argument; tests/test_ppo.py TestShaping pins the consequences.
        """
        out = []
        for o, r, o2 in zip(obs, rewards, next_obs):
            if self.use_shaping and not target_changed(o, o2):
                r += SHAPE * (self.inputs.potential(o2) - self.inputs.potential(o))
            out.append(REWARD_SCALE * r)
        return out

    def _learn(self):
        x, mask, acted, logp, value, after, reward, done, ready = (
            torch.stack(part) for part in zip(*self.rollout))
        advantage = self._gae(reward, value, after, done)
        returns = advantage + value

        # Flatten time and env together but keep the robot axis: pooled() has to average over
        # one env's robots, so a minibatch is a set of whole fleets, never of loose robots.
        flat = [t.flatten(0, 1) for t in (x, mask, acted, logp, advantage, returns, ready)]
        groups = flat[0].shape[0]
        order = list(range(groups))
        size = max(1, groups // MINIBATCHES)
        for _ in range(EPOCHS):
            self.rng.shuffle(order)
            for start in range(0, groups, size):
                self._step([t[order[start:start + size]] for t in flat])
        self.updates += 1
        self.log["explained_var"] = _explained(returns, value)

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
        x, mask, acted, old_logp, advantage, returns, live = batch
        inputs = pooled(x)
        logits = self.actor(inputs).masked_fill(~mask, NEG)
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
        value_loss = 0.5 * ((self.critic(inputs).squeeze(-1) - returns) ** 2).mean()
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
                 "steps": self.steps, "updates": self.updates, **self.log}
        (folder / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load(self, folder):
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        self.use_shaping = state["shaping"]
        self.eval_greedy, self.lr, self.ent = state["eval_greedy"], state["lr"], state["ent"]
        if state["bfs"] != self.inputs.use_bfs:
            # The checkpoint's own setting wins: it decides how wide the saved weights are.
            self.inputs = Features(self.spec.obs_features, use_bfs=state["bfs"])
            self.inputs.use(self.spec.grid)
        self._build()
        self.actor.load_state_dict(torch.load(folder / "actor.pt", weights_only=True))
        self.critic.load_state_dict(torch.load(folder / "critic.pt", weights_only=True))
        self.opt.load_state_dict(torch.load(folder / "opt.pt", weights_only=True))
        self.steps, self.updates = state["steps"], state["updates"]
        self.log = {key: state[key] for key in self.log}


def _number(value, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} wants a number, got '{value}'") from None


def _explained(returns, value):
    """1 - Var(returns - value) / Var(returns): 0 means the critic is no better than the mean."""
    spread = returns.var()
    if spread <= 0:
        return 0.0
    return float(1 - (returns - value).var() / spread)
