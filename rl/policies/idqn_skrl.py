import os
import torch
import torch.nn as nn
import numpy as np
from skrl.models.torch import CategoricalMixin, Model

N_ACTIONS = 6
OBS_DIM = 10  # [x, y, carrying, has_task, dx_target, dy_target, n_up, n_down, n_left, n_right]

NEIGHBOR_MAP = {"AISLE": 0, "WALL": 1, "SHELF_EMPTY": 2, "SHELF_PACKAGE": 3, "DOCK": 4}


def get_state(obs):
    """Extraction robuste du vecteur d'état à 10 dimensions."""
    if isinstance(obs, dict):
        x, y = obs["position"]
        carrying = 1.0 if obs.get("carrying", False) else 0.0
        has_task = 1.0 if obs.get("has_task", False) else 0.0
        tx, ty = obs.get("target", (x, y))
        neighbors = obs.get("neighbors", ["AISLE"] * 4)

        n_vec = [float(NEIGHBOR_MAP.get(n, 0) if isinstance(n, str) else n) / 4.0 for n in neighbors]
    else:
        x, y = int(obs[0]), int(obs[1])
        carrying = float(obs[2])
        has_task = float(obs[3])
        tx, ty = int(obs[4]), int(obs[5])

        n_vec = []
        for k in range(4):
            dist = obs[6 + 2 * k] if len(obs) > 6 + 2 * k else 0
            type_id = obs[7 + 2 * k] if len(obs) > 7 + 2 * k else 0
            n_vec.append(float(type_id) / 5.0 if dist == 1 else 0.0)

    dx = (tx - x) / 20.0
    dy = (ty - y) / 20.0

    vec = [x / 20.0, y / 20.0, carrying, has_task, dx, dy] + n_vec
    return np.array(vec, dtype=np.float32)


def get_action_mask(obs):
    """Masque d'actions strict :
    0: FORWARD, 1: BACKWARD (Désactivé), 2: TURN_LEFT, 3: TURN_RIGHT, 4: PICKUP, 5: DROP
    """
    mask = [True] * N_ACTIONS

    # Interdire systématiquement de reculer (supprime les collisions arrière)
    mask[1] = False

    # --- FORMAT DICTIONNAIRE (Isaac Sim / Démo) ---
    if isinstance(obs, dict):
        neighbors = obs.get("neighbors", ["AISLE"] * 4)
        carrying = obs.get("carrying", False)
        front_cell = neighbors[0] if len(neighbors) > 0 else "AISLE"

        # FORWARD : Interdit face à un mur ou une étagère
        if front_cell in ["WALL", 1, "SHELF_EMPTY", 2, "SHELF_PACKAGE", 3]:
            mask[0] = False

        # PICKUP : Autorisé UNIQUEMENT si on ne porte rien ET que la case en face contient un colis
        if carrying or front_cell not in ["SHELF_PACKAGE", 3]:
            mask[4] = False

        # DROP : Autorisé UNIQUEMENT si on porte un colis ET que la case en face est un quai (DOCK)
        if not carrying or front_cell not in ["DOCK", 4]:
            mask[5] = False

    # --- FORMAT TABLEAU / TUPLE (Gym / Entraînement) ---
    elif isinstance(obs, (list, tuple, np.ndarray)):
        carrying = bool(obs[2])
        front_dist = obs[6] if len(obs) > 6 else 0
        front_type = obs[7] if len(obs) > 7 else 0

        # Bloquer FORWARD si obstacle sur la case adjacente
        if front_dist == 1 and front_type in [1, 2, 3, "WALL", "SHELF_EMPTY", "SHELF_PACKAGE"]:
            mask[0] = False

        # PICKUP autorisé uniquement si adjacent à un colis
        is_package_front = (front_dist == 1 and front_type in [3, "SHELF_PACKAGE"])
        if carrying or not is_package_front:
            mask[4] = False

        # DROP autorisé uniquement si adjacent à un quai
        is_dock_front = (front_dist == 1 and front_type in [4, "DOCK"])
        if not carrying or not is_dock_front:
            mask[5] = False

    return mask


class QNet(CategoricalMixin, Model):
    def __init__(self, observation_space=None, action_space=None, device="cpu"):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)

        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS),
        )

    def compute(self, inputs, role=""):
        obs = inputs.get("states")
        if obs is None:
            obs = inputs.get("observations")
        return self.net(obs), {}


class SKRLIDQNPolicy:
    def __init__(self, weights_path=None, device="cpu"):
        self.device = torch.device(device)
        self.model = QNet(device=self.device)
        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def act(self, obs) -> int:
        st_vec = torch.tensor(np.array([get_state(obs)], dtype=np.float32), device=self.device)
        mask = get_action_mask(obs)

        with torch.no_grad():
            q_vals, _ = self.model.compute({"states": st_vec})
            logits = q_vals.squeeze(0).clone()

            # Masquage : -1e9 sur les actions physiquement invalides
            for act_idx, is_valid in enumerate(mask):
                if not is_valid:
                    logits[act_idx] = -1e9

            return int(torch.argmax(logits).item())