import os
import torch
import numpy as np

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory

from env.warehouse import Warehouse
from env.layout import maze_layout
from rl.policies.mappo_skrl import Actor, Critic, STATE_DIM
from rl.policies.idqn_skrl import get_state, N_ACTIONS, OBS_DIM

N_ROBOTS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Warehouse(layout=maze_layout(42, n_robots=N_ROBOTS), n_robots=N_ROBOTS, max_steps=500)
obs_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
state_space = spaces.Box(low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)
act_space = spaces.Discrete(N_ACTIONS)

cfg = PPO.default_config.copy() if hasattr(PPO, "default_config") else {}
cfg["rollouts"] = 500
cfg["learning_rate"] = 5e-4
cfg["mini_batches"] = 4
cfg["discount_factor"] = 0.99
cfg["gae_lambda"] = 0.95
cfg["learning_rate_scheduler"] = None

agents = []
for i in range(N_ROBOTS):
    memory = RandomMemory(memory_size=500, num_envs=1, device=DEVICE)
    ag = PPO(
        models={
            "policy": Actor(obs_space, act_space, DEVICE),
            "value": Critic(state_space, act_space, DEVICE)
        },
        memory=memory,
        cfg=cfg,
        observation_space=obs_space,
        state_space=state_space,
        action_space=act_space,
        device=DEVICE
    )
    ag.init()
    agents.append(ag)

print(f"CUDA disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU utilisé : {torch.cuda.get_device_name(0)}")
print("Entraînement MAPPO (Centralized Critic PPO) avec skrl 2.1.0...")

global_step = 0
TOTAL_TIMESTEPS = 750000

def dist_to_target(o):
    if isinstance(o, dict):
        x, y = o["position"]
        tx, ty = o.get("target", (x, y))
    else:
        x, y = int(o[0]), int(o[1])
        tx, ty = int(o[4]), int(o[5])
    return abs(tx - x) + abs(ty - y)

for episode in range(1500):
    obs, done = env.reset(seed=42 + episode), False
    while not done:
        states = [get_state(obs[i]) for i in range(N_ROBOTS)]
        joint_state = np.concatenate(states, axis=0)

        actions = []
        actions_t = []

        for i in range(N_ROBOTS):
            st_tensor = torch.tensor(np.array([states[i]], dtype=np.float32), device=DEVICE)
            joint_tensor = torch.tensor(np.array([joint_state], dtype=np.float32), device=DEVICE)
            
            # Dépaquetage corrigé (2 valeurs retournées)
            act, _ = agents[i].act(states=joint_tensor, observations=st_tensor, timestep=global_step, timesteps=TOTAL_TIMESTEPS)
            actions.append(act.item())
            actions_t.append(act)

        next_obs, rewards, done, info = env.step(actions)
        next_states = [get_state(next_obs[i]) for i in range(N_ROBOTS)]
        next_joint_state = np.concatenate(next_states, axis=0)

        for i in range(N_ROBOTS):
            r = rewards[i]
            if r == -1.0:
                shaped_r = -2.0
            elif r > 0:
                shaped_r = r * 2.0
            else:
                d_before = dist_to_target(obs[i])
                d_after = dist_to_target(next_obs[i])
                progress = d_before - d_after
                shaped_r = 0.2 * progress - 0.05

            s_t = torch.tensor(np.array([states[i]], dtype=np.float32), device=DEVICE)
            ns_t = torch.tensor(np.array([next_states[i]], dtype=np.float32), device=DEVICE)
            joint_t = torch.tensor(np.array([joint_state], dtype=np.float32), device=DEVICE)
            next_joint_t = torch.tensor(np.array([next_joint_state], dtype=np.float32), device=DEVICE)
            rw_t = torch.tensor([[shaped_r]], dtype=torch.float32, device=DEVICE)
            d_t = torch.tensor([[done]], dtype=torch.bool, device=DEVICE)

            agents[i].record_transition(
                states=joint_t,
                observations=s_t,
                actions=actions_t[i],
                rewards=rw_t,
                next_states=next_joint_t,
                next_observations=ns_t,
                terminated=d_t,
                truncated=d_t,
                infos={},
                timestep=global_step,
                timesteps=TOTAL_TIMESTEPS
            )

            agents[i].pre_interaction(timestep=global_step, timesteps=TOTAL_TIMESTEPS)
            agents[i].post_interaction(timestep=global_step, timesteps=TOTAL_TIMESTEPS)

        obs = next_obs
        global_step += 1

    if (episode + 1) % 100 == 0:
        print(f"Épisode MAPPO {episode + 1}/1500 terminé (global_step: {global_step})")

os.makedirs("weights", exist_ok=True)
for i in range(N_ROBOTS):
    torch.save(agents[i].models["policy"].state_dict(), f"weights/skrl_mappo_robot_{i}.pt")
print("Sauvegarde MAPPO terminée dans 'weights/'.")