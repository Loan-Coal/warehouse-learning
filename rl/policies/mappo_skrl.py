import os
import torch
import torch.nn as nn
import numpy as np
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from rl.policies.idqn_skrl import get_state, N_ACTIONS, OBS_DIM

N_ROBOTS = 2
STATE_DIM = OBS_DIM * N_ROBOTS  # 20D (Concaténation des états des 2 robots)


class Actor(CategoricalMixin, Model):
    """Réseau de politique (Acteur) : observation locale (10D) -> distribution d'actions."""
    def __init__(self, observation_space=None, action_space=None, device="cpu"):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)

        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, N_ACTIONS)
        )
        self.to(device)

    def compute(self, inputs, role=""):
        obs = inputs.get("observations")
        if obs is None:
            obs = inputs.get("states")
        return self.net(obs), {}


class Critic(DeterministicMixin, Model):
    """Réseau de valeur (Critique Centralisé) : état global joint (20D) -> V(S)."""
    def __init__(self, observation_space=None, action_space=None, device="cpu"):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.to(device)

    def compute(self, inputs, role=""):
        states = inputs.get("states")
        if states is None:
            states = inputs.get("observations")
        return self.net(states), {}


class SKRLMAPPOPolicy:
    """Classe d'inférence pour charger et exécuter l'Acteur sous Isaac Sim."""
    def __init__(self, weights_path=None, device="cpu"):
        self.device = torch.device(device)
        self.model = Actor(device=self.device)
        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def act(self, obs) -> int:
        st_vec = torch.tensor(np.array([get_state(obs)], dtype=np.float32), device=self.device)
        with torch.no_grad():
            q_vals, _ = self.model.compute({"observations": st_vec})
            return int(torch.argmax(q_vals, dim=1).item())