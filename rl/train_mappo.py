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
from rl.policies.idqn_skrl import get_state, get_action_mask, N_ACTIONS, OBS_DIM

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

agents = []
for i in range(N_ROBOTS):
    memory = RandomMemory(memory_size=500, num_envs=1, device=DEVICE)
    ag = PPO(
        models={"policy": Actor(obs_space, act_space, DEVICE), "value": Critic(state_space, act_space, DEVICE)},
        memory=memory, cfg=cfg, observation_space=obs_space, state_space=state_space, action_space=act_space, device=DEVICE
    )
    ag.init()
    agents.append(ag)

print("Démarrage entraînement MAPPO avec Action Masking...")

global_step = 0
TOTAL_TIMESTEPS = 500000

def dist_to_target(o):
    if isinstance(o, dict):
        x, y = o["position"]
        tx, ty = o.get("target", (x, y))
    else:
        x, y = int(o[0]), int(o[1])
        tx, ty = int(o[4]), int(o[5])
    return abs(tx - x) + abs(ty - y)

for episode in range(1200):
    obs, done = env.reset(seed=42 + episode), False
    while not done:
        states = [get_state(obs[i]) for i in range(N_ROBOTS)]
        joint_state = np.concatenate(states, axis=0)

        actions = []
        actions_t = []

        for i in range(N_ROBOTS):
            st_tensor = torch.tensor(np.array([states[i]], dtype=np.float32), device=DEVICE)
            mask = get_action_mask(obs[i])
            
            with torch.no_grad():
                out, _ = agents[i].models["policy"].compute({"observations": st_tensor})
                logits = out.squeeze(0).clone()
                for idx, is_valid in enumerate(mask):
                    if not is_valid:
                        logits[idx] = -1e9
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                chosen_act = dist.sample().item()

            actions.append(chosen_act)
            actions_t.append(torch.tensor([chosen_act], device=DEVICE))

        next_obs, rewards, done, info = env.step(actions)
        next_states = [get_state(next_obs[i]) for i in range(N_ROBOTS)]
        next_joint_state = np.concatenate(next_states, axis=0)

        for i in range(N_ROBOTS):
            r = rewards[i]
            if r > 0:
                shaped_r = r * 2.0
            else:
                d_before = dist_to_target(obs[i])
                d_after = dist_to_target(next_obs[i])
                progress = d_before - d_after
                shaped_r = 0.3 * progress - 0.02

            s_t = torch.tensor(np.array([states[i]], dtype=np.float32), device=DEVICE)
            ns_t = torch.tensor(np.array([next_states[i]], dtype=np.float32), device=DEVICE)
            joint_t = torch.tensor(np.array([joint_state], dtype=np.float32), device=DEVICE)
            next_joint_t = torch.tensor(np.array([next_joint_state], dtype=np.float32), device=DEVICE)
            rw_t = torch.tensor([[shaped_r]], dtype=torch.float32, device=DEVICE)
            d_t = torch.tensor([[done]], dtype=torch.bool, device=DEVICE)

            agents[i].record_transition(
                states=joint_t, observations=s_t, actions=actions_t[i], rewards=rw_t,
                next_states=next_joint_t, next_observations=ns_t, terminated=d_t, truncated=d_t,
                infos={}, timestep=global_step, timesteps=TOTAL_TIMESTEPS
            )

            agents[i].pre_interaction(timestep=global_step, timesteps=TOTAL_TIMESTEPS)
            agents[i].post_interaction(timestep=global_step, timesteps=TOTAL_TIMESTEPS)

        obs = next_obs
        global_step += 1

    if (episode + 1) % 100 == 0:
        print(f"Épisode MAPPO {episode + 1}/1200 terminé (global_step: {global_step})")

os.makedirs("weights", exist_ok=True)
for i in range(N_ROBOTS):
    torch.save(agents[i].models["policy"].state_dict(), f"weights/skrl_mappo_robot_{i}.pt")
print("Modèles MAPPO enregistrés.")