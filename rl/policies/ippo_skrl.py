import os
import torch
import torch.nn as nn
import numpy as np
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model

N_ACTIONS = 6
OBS_DIM = 11  # [x, y, heading, carrying, has_task, dx_target, dy_target, n_up, n_down, n_left, n_right]
N_ROBOTS = 2
STATE_DIM = OBS_DIM * N_ROBOTS  # 22D (Concaténation des états des 2 robots)

NEIGHBOR_MAP = {"AISLE": 0, "WALL": 1, "SHELF_EMPTY": 2, "SHELF_PACKAGE": 3, "DOCK": 4}


def get_state(obs):
    """Extraction 11D : x, y, heading, carrying, has_task, dx, dy + 4 voisins/rayons."""
    if isinstance(obs, dict):
        x, y = obs.get("position", (0, 0))
        heading = float(obs.get("heading", 0)) / 3.0
        carrying = bool(obs.get("carrying", False))
        has_task = 1.0 if obs.get("has_task", False) else 0.0

        # CIBLE DYNAMIQUE
        if not carrying and "crate_position" in obs:
            tx, ty = obs["crate_position"]
        elif carrying and "dock_position" in obs:
            tx, ty = obs["dock_position"]
        elif "target_x" in obs and "target_y" in obs:
            tx, ty = obs["target_x"], obs["target_y"]
        else:
            tx, ty = obs.get("target", (x, y))

        neighbors = obs.get("neighbors", ["AISLE"] * 4)
        n_vec = [float(NEIGHBOR_MAP.get(n, 0) if isinstance(n, str) else n) / 4.0 for n in neighbors]

    else:
        x, y = int(obs[0]), int(obs[1])
        carrying = bool(obs[2])
        has_task = float(obs[3])
        tx, ty = int(obs[4]), int(obs[5])
        heading = float(obs[14]) / 3.0 if len(obs) > 14 else 0.0

        n_vec = []
        for k in range(4):
            dist = obs[6 + 2 * k] if len(obs) > 6 + 2 * k else 0
            type_id = obs[7 + 2 * k] if len(obs) > 7 + 2 * k else 0
            n_vec.append(float(type_id) / 5.0 if dist == 1 else 0.0)

    dx = (tx - x) / 20.0
    dy = (ty - y) / 20.0

    vec = [x / 20.0, y / 20.0, heading, float(carrying), has_task, dx, dy] + n_vec
    return np.array(vec, dtype=np.float32)


def get_action_mask(obs):
    """Masque d'actions strict."""
    mask = [True] * N_ACTIONS
    mask[1] = False

    if isinstance(obs, dict):
        neighbors = obs.get("neighbors", ["AISLE"] * 4)
        carrying = obs.get("carrying", False)
        front_cell = neighbors[0] if len(neighbors) > 0 else "AISLE"

        if front_cell in ["WALL", 1, "SHELF_EMPTY", 2, "SHELF_PACKAGE", 3]:
            mask[0] = False
        if carrying or front_cell not in ["SHELF_PACKAGE", 3]:
            mask[4] = False
        if not carrying or front_cell not in ["DOCK", 4]:
            mask[5] = False

    return mask


class Actor(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device="cpu"):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)

        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS)
        )
        self.to(device)

    def compute(self, inputs, role=""):
        obs = inputs["observations"]
        return self.net(obs), {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device="cpu"):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.to(device)

    def compute(self, inputs, role=""):
        obs = inputs["observations"]
        return self.net(obs), {}


class SKRLIPPOPolicy:
    def __init__(self, weights_path=None, device="cpu", observation_space=None, action_space=None):
        self.device = torch.device(device)
        self.model = Actor(observation_space=observation_space, action_space=action_space, device=self.device)

        if weights_path is not None and os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def act(self, raw_obs) -> int:
        obs_tensor = torch.tensor(np.array([get_state(raw_obs)], dtype=np.float32), device=self.device)
        mask = get_action_mask(raw_obs)
        with torch.no_grad():
            logits, _ = self.model.compute({"observations": obs_tensor})
            logits = logits.squeeze(0).clone()
            for act_idx, is_valid in enumerate(mask):
                if not is_valid:
                    logits[act_idx] = -1e9
            return int(torch.argmax(logits, dim=-1).item())