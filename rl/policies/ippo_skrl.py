import os
import torch
import torch.nn as nn
import numpy as np
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model

N_ACTIONS = 6
OBS_DIM = 10  # [x, y, carrying, has_task, dx_target, dy_target, n_up, n_down, n_left, n_right]
N_ROBOTS = 2
STATE_DIM = OBS_DIM * N_ROBOTS  # 20D (Concatenation of the states of 2 robots)

def get_state(obs):
    """ Returns a observation vector of size 10 extracted from the observation dictionary or tuple. """
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

        # Extract the 4 rays (up, down, left, right) from the 14D tuple of warehouse.py
        # obs[6..13] = [d0, t0, d1, t1, d2, t2, d3, t3]
        # If the distance if greater than 1 then the neighbor is empty
        # If the distance is 1 then the neighbor is occupied by an object of type t
        n_vec = []
        for k in range(4):
            dist = obs[6 + 2 * k]
            type_id = obs[7 + 2 * k]
            # If an obstacle is on the immediately adjacent cell (dist == 1)
            n_vec.append(float(type_id) / 5.0 if dist == 1 else 0.0)

    dx = (tx - x) / 20.0
    dy = (ty - y) / 20.0

    vec = [x / 20.0, y / 20.0, carrying, has_task, dx, dy] + n_vec
    return np.array(vec, dtype=np.float32)

class Actor(CategoricalMixin, Model):
    """Network for the policy (Actor): local observation (10D) -> action distribution."""
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
    """Value network: local observation 10D -> scalar state value"""
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
    """ Loads one trained IPPO actor for warehous inference"""
    def __init__(self, weights_path=None, device="cpu", observation_space=None, action_space=None):
        self.device = torch.device(device)

        self.model = Actor(observation_space=observation_space, action_space=action_space, device=self.device)

        if weights_path is not None:
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(weights_path)

            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)

        self.model.eval()

    
    def act(self, raw_obs) -> int:
        obs_tensor = torch.tensor(np.array([get_state(raw_obs)], dtype=np.float32), device=self.device)
        with torch.no_grad():
            logits, _ = self.model.compute({"observations": obs_tensor})
            return int(torch.argmax(logits, dim=-1).item())