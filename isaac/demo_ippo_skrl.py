
import os
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import torch
from gymnasium import spaces

from env.warehouse import Warehouse
from env.layout import maze_layout
from rl.policies.ippo_skrl import SKRLIPPOPolicy, OBS_DIM, N_ACTIONS

N_ROBOTS = 2
DEVICE = "cpu"

obs_space = spaces.Box(
    low=-1.00,
    high=1.00,
    shape=(OBS_DIM,),
    dtype=np.float32,
)
act_space = spaces.Discrete(N_ACTIONS)

env = Warehouse(
    layout=maze_layout(42, n_robots=N_ROBOTS),
    n_robots=N_ROBOTS,
    max_steps=500,
)

policies = []

for i in range(N_ROBOTS):
    weights_path = f"weights/skrl_ippo_robot_{i}.pt"

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Missing trained policy: {weights_path}"
        )

    policy = SKRLIPPOPolicy(
        weights_path=weights_path,
        device=DEVICE,
        observation_space=obs_space,
        action_space=act_space,
    )
    policies.append(policy)

obs = env.reset(seed=100)
done = False

step_count = 0
total_deliveries = 0
total_invalid = 0

print("--- Starting IPPO demonstration ---")

try:
    while simulation_app.is_running() and not done:
        actions = [
            policies[i].act(obs[i])
            for i in range(N_ROBOTS)
        ]

        next_obs, rewards, done, info = env.step(actions)

        # Render only in this demonstration.
        env.render()

        for i, reward in enumerate(rewards):
            if reward >= 5.0:
                total_deliveries += 1
            elif reward == -1.0:
                total_invalid += 1

        obs = next_obs
        step_count += 1

finally:
    simulation_app.close()

print(
    f"Episode ended: steps={step_count}, "
    f"deliveries={total_deliveries}, "
    f"invalid actions={total_invalid}"
)
