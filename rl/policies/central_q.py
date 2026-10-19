"""Centralised tabular Q-learning: one table, one joint action for the whole fleet.

A teaching baseline, not a model to tune. The state is every robot's observation together and
the action is one of 6**n joint actions, so the table grows as (34*4*17)**n states x 6**n
actions on the default map: rl/qtable_report.py measures where that stops working (2, 3, 4 robots).

    state         tuple of every robot's (x, y, heading, carrying, has_task, target_x, target_y)
    joint action  an int in 0 .. 6**n - 1, robot 0 is the most significant base-6 digit
    mask          only joint actions whose every robot action is legal (rl.nav.legal_actions)
    update        Q[s][a] += lr * (sum(rewards) + gamma * max_legal Q[s'] - Q[s][a])

A row is created only in update(), never in act(); once the table would pass max_mb, new
states are no longer stored and the rejected updates are counted (stats()["dropped"]).
"""
import json
import random
from itertools import product

import numpy as np

from env.warehouse import N_ACTIONS
from rl.nav import legal_actions
from rl.policy import Policy

MB = 2 ** 20


class CentralQ(Policy):
    OBS = ("position", "heading", "task")

    def __init__(self, spec, seed=0):
        self.n_robots = spec.n_robots
        super().__init__(spec, seed)
        self.n_joint = N_ACTIONS ** self.n_robots
        self.rng = random.Random(seed)
        self.lr, self.gamma = 0.1, 0.95
        self.eps_start, self.eps_end, self.eps_steps = 1.0, 0.05, 500_000
        self.max_mb = 1500.0
        self.q = {}             # state -> np.float32[6**n]
        self.steps = 0          # training steps, for the epsilon schedule
        self.updates = self.rejected = 0

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, spec):
        # The loop sets a fresh spec every episode: the joint action space is fixed at construction.
        if spec.n_robots != getattr(self, "n_robots", spec.n_robots):
            raise ValueError(f"central_q was built for {self.n_robots} robots, the env has "
                             f"{spec.n_robots}: its joint actions do not transfer, train one per fleet size")
        self._spec = spec

    # -- joint actions -------------------------------------------------------------------------

    def encode(self, actions):
        """One action per robot -> joint action (robot 0 = most significant digit)."""
        joint = 0
        for a in actions:
            joint = joint * N_ACTIONS + a
        return joint

    def decode(self, joint):
        """Joint action -> one action per robot."""
        actions = []
        for _ in range(self.n_robots):
            joint, a = divmod(joint, N_ACTIONS)
            actions.append(a)
        return actions[::-1]

    def legal_joint(self, obs):
        """The joint actions whose every robot action is legal, ascending."""
        self._check(obs)
        per_robot = [[a for a, ok in enumerate(legal_actions(self.spec.grid, o)) if ok] for o in obs]
        return np.array([self.encode(c) for c in product(*per_robot)], dtype=np.int64)

    def state(self, obs):
        self._check(obs)
        return tuple(tuple(int(v) for v in o) for o in obs)

    # -- acting and learning -------------------------------------------------------------------

    @property
    def epsilon(self):
        frac = min(1.0, self.steps / self.eps_steps) if self.eps_steps > 0 else 1.0
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def act(self, obs):
        legal = self.legal_joint(obs)
        if self.training and self.rng.random() < self.epsilon:
            return self.decode(int(legal[self.rng.randrange(len(legal))]))
        row = self.q.get(self.state(obs))
        if row is None:
            return self.decode(int(legal[0]))       # unknown state: first legal joint action
        return self.decode(int(legal[np.argmax(row[legal])]))

    def update(self, obs, actions, rewards, next_obs, done):
        # done is a time limit, never a terminal state: always bootstrap from next_obs.
        self.steps += 1
        self.updates += 1
        s = self.state(obs)
        row = self.q.get(s)
        if row is None:
            if (len(self.q) + 1) * self.n_joint * 4 > self.max_mb * MB:
                self.rejected += 1
                return
            row = self.q[s] = np.zeros(self.n_joint, dtype=np.float32)
        nxt = self.q.get(self.state(next_obs))
        best_next = 0.0 if nxt is None else float(nxt[self.legal_joint(next_obs)].max())
        a = self.encode(actions)
        target = float(sum(rewards)) + self.gamma * best_next
        row[a] += self.lr * (target - row[a])

    def _check(self, obs):
        if len(obs) != self.n_robots:
            raise ValueError(f"central_q was built for {self.n_robots} robots, got {len(obs)} observations")

    # -- runner plumbing -----------------------------------------------------------------------

    def configure(self, lr=None, gamma=None, eps_start=None, eps_end=None, eps_steps=None,
                  max_mb=None, **unknown):
        if unknown:
            raise ValueError(f"unknown setting {', '.join(sorted(unknown))}; this policy takes "
                             f"lr, gamma, eps_start, eps_end, eps_steps, max_mb")
        for name, value, cast in (("lr", lr, float), ("gamma", gamma, float),
                                  ("eps_start", eps_start, float), ("eps_end", eps_end, float),
                                  ("eps_steps", eps_steps, int), ("max_mb", max_mb, float)):
            if value is not None:
                setattr(self, name, cast(value))

    def stats(self):
        return {"states": len(self.q),
                "table_mb": round(len(self.q) * self.n_joint * 4 / MB, 3),
                "epsilon": round(self.epsilon, 4),
                "dropped": round(self.rejected / self.updates, 4) if self.updates else 0.0}

    def _settings(self):
        return {"n_robots": self.n_robots, "lr": self.lr, "gamma": self.gamma,
                "eps_start": self.eps_start, "eps_end": self.eps_end, "eps_steps": self.eps_steps,
                "max_mb": self.max_mb, "steps": self.steps, "updates": self.updates,
                "rejected": self.rejected}

    def save(self, folder):
        width = self.n_robots * len(self.spec.obs_fields)
        keys = np.array([[v for o in s for v in o] for s in self.q], dtype=np.int16).reshape(-1, width)
        q = np.array(list(self.q.values()), dtype=np.float32).reshape(-1, self.n_joint)
        np.save(folder / "keys.npy", keys)
        np.save(folder / "q.npy", q)
        (folder / "central_q.json").write_text(json.dumps(self._settings(), indent=2, sort_keys=True),
                                               encoding="utf-8")

    def load(self, folder):
        settings = json.loads((folder / "central_q.json").read_text(encoding="utf-8"))
        if settings["n_robots"] != self.n_robots:
            raise ValueError(f"checkpoint is a central_q for {settings['n_robots']} robots, the env has "
                             f"{self.n_robots}: use the same --robots as in training")
        keys, q = np.load(folder / "keys.npy"), np.load(folder / "q.npy")
        per_robot = len(self.spec.obs_fields)
        self.q = {tuple(tuple(int(v) for v in k[i:i + per_robot]) for i in range(0, len(k), per_robot)):
                  row.copy() for k, row in zip(keys, q)}
        # The settings come back through configure() (the checkpoint's policy_opts); only the
        # schedule position and the counters are restored here.
        for name in ("steps", "updates", "rejected"):
            setattr(self, name, int(settings[name]))
