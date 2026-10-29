import os
import torch
import torch.nn as nn
import numpy as np
from skrl.models.torch import CategoricalMixin, Model

N_ACTIONS = 6
OBS_DIM = 10  # [x, y, carrying, has_task, dx_target, dy_target, n_up, n_down, n_left, n_right]

def get_state(obs):
    """Retourne un vecteur d'observation de taille 10 extrait du dictionnaire ou du tuple d'observation."""
    if isinstance(obs, dict):
        x, y = obs["position"]
        carrying = 1.0 if obs.get("carrying", False) else 0.0
        has_task = 1.0 if obs.get("has_task", False) else 0.0
        tx, ty = obs.get("target", (x, y))
        neighbors = obs.get("neighbors", [0, 0, 0, 0])
        n_vec = [float(n) / 5.0 for n in neighbors]
    else:
        x, y = int(obs[0]), int(obs[1])
        carrying = float(obs[2])
        has_task = float(obs[3])
        tx, ty = int(obs[4]), int(obs[5])

        # Extraction des 4 rayons (haut, bas, gauche, droite) du tuple 14D de warehouse.py
        # obs[6..13] = [d0, t0, d1, t1, d2, t2, d3, t3]
        n_vec = []
        for k in range(4):
            dist = obs[6 + 2 * k]
            type_id = obs[7 + 2 * k]
            # Si un obstacle est sur la case immédiatement adjacente (dist == 1)
            n_vec.append(float(type_id) / 5.0 if dist == 1 else 0.0)

    # Direction relative normalisée vers la cible (-1.0 à 1.0)
    dx = (tx - x) / 20.0
    dy = (ty - y) / 20.0

    vec = [x / 20.0, y / 20.0, carrying, has_task, dx, dy] + n_vec
    return np.array(vec, dtype=np.float32)


class QNet(CategoricalMixin, Model):
    def __init__(self, observation_space=None, action_space=None, device="cpu"):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)

        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, N_ACTIONS)
        )

    def compute(self, inputs, role=""):
        obs = inputs.get("states")
        if obs is None:
            obs = inputs.get("observations")
        return self.net(obs), {}


class SKRLIDQNPolicy:
    def __init__(self, weights_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNet(device=self.device)
        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def act(self, obs) -> int:
        st_vec = torch.tensor(np.array([get_state(obs)], dtype=np.float32), device=self.device)
        with torch.no_grad():
            q_vals, _ = self.model.compute({"states": st_vec})
            return int(torch.argmax(q_vals, dim=1).item())