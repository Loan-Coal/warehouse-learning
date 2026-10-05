"""Q-learning tabulaire centralisé : un seul contrôleur choisit l'action jointe de tous les robots.

Portage du prototype (notebook) sur l'environnement du dépôt.
État : pour chaque robot (x, y, carrying, has_task, target_x, target_y).
Action jointe : un entier dans 0 .. 6**n - 1, décodé en une action par robot.
"""
import pickle
import random
from collections import defaultdict

import numpy as np

N_ACTIONS = 6


class CentralQPolicy:
    def __init__(self, n_robots, lr=0.5, gamma=0.95, epsilon=0.0, seed=0):
        self.n_robots = n_robots
        self.n_joint = N_ACTIONS ** n_robots
        self.lr, self.gamma, self.epsilon = lr, gamma, epsilon
        self.rng = random.Random(seed)
        self.Q = defaultdict(lambda: np.zeros(self.n_joint, dtype=np.float32))

    def state(self, obs):
        return tuple(o[:6] for o in obs)

    def decode(self, a):
        """Action jointe -> une action par robot (robot 0 = chiffre de poids fort)."""
        actions = []
        for _ in range(self.n_robots):
            actions.append(a % N_ACTIONS)
            a //= N_ACTIONS
        return actions[::-1]

    def encode(self, actions):
        a = 0
        for x in actions:
            a = a * N_ACTIONS + x
        return a

    def act_all(self, obs):
        if self.rng.random() < self.epsilon:
            a = self.rng.randrange(self.n_joint)          # exploration
        else:
            a = int(np.argmax(self.Q[self.state(obs)]))   # exploitation
        return self.decode(a)

    def update_all(self, obs, actions, rewards, next_obs, done):
        s, s2, a = self.state(obs), self.state(next_obs), self.encode(actions)
        r = sum(rewards)                                  # récompense d'équipe
        # done = simple troncature (pas d'état terminal) : on continue de bootstrapper
        self.Q[s][a] += self.lr * (r + self.gamma * self.Q[s2].max() - self.Q[s][a])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"n_robots": self.n_robots, "Q": dict(self.Q)}, f)

    @classmethod
    def load(cls, path, **kwargs):
        with open(path, "rb") as f:
            data = pickle.load(f)
        policy = cls(data["n_robots"], **kwargs)
        policy.Q.update(data["Q"])
        return policy
