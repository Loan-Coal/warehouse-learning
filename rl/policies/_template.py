"""TEMPLATE: copy me, rename the class, register it. Full guide: README.md, "Writing an algorithm".

    copy rl\\policies\\_template.py rl\\policies\\myalgo.py

    then in rl/policies/__init__.py:
        from rl.policies.myalgo import MyAlgo
        POLICIES = {..., "myalgo": shared(MyAlgo)}      (or independent(MyAlgo))

This is a small working tabular Q-learner for ONE robot (a RobotPolicy): keep what you need,
replace the rest. For a controller that decides for all robots at once, subclass
rl.policy.Policy instead and write act(obs_list) -> list[int].
"""
import json
import random

from env.warehouse import N_ACTIONS
from rl.policy import RobotPolicy


class MyAlgo(RobotPolicy):
    def __init__(self, spec, seed=0, lr=0.1, gamma=0.95, epsilon=0.1):
        super().__init__(spec, seed)
        self.rng = random.Random(seed)
        self.lr, self.gamma, self.epsilon = lr, gamma, epsilon
        self.q = {}     # obs -> one value per action; obs is a tuple, so it works as a key

    def act_one(self, obs):
        # Explore only while training: evaluation and the Isaac demo play the best known action.
        if self.training and self.rng.random() < self.epsilon:
            return self.rng.randrange(N_ACTIONS)
        values = self._values(obs)
        return values.index(max(values))

    def update_one(self, obs, action, reward, next_obs, done):
        # done is only a time limit, never a real end: always bootstrap from next_obs.
        target = reward + self.gamma * max(self._values(next_obs))
        values = self.q.setdefault(obs, [0.0] * N_ACTIONS)
        values[action] += self.lr * (target - values[action])

    def save(self, folder):
        # Plain JSON, never pickle: it still loads after you rename or change this class.
        rows = [[list(obs), values] for obs, values in self.q.items()]
        (folder / "q.json").write_text(json.dumps(rows), encoding="utf-8")

    def load(self, folder):
        rows = json.loads((folder / "q.json").read_text(encoding="utf-8"))
        self.q = {tuple(obs): values for obs, values in rows}

    def _values(self, obs):
        return self.q.get(obs, [0.0] * N_ACTIONS)
