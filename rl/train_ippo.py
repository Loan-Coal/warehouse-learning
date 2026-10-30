import os
import torch
import numpy as np

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.ippo import IPPO, IPPO_CFG

from env.warehouse import Warehouse
from env.layout import maze_layout
from rl.policies.ippo_skrl import Actor, Critic, get_state, N_ACTIONS, OBS_DIM

N_ROBOTS = 2
AGENT_IDS = [f"robot_{i}" for i in range(N_ROBOTS)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Warehouse(
    layout=maze_layout(42, n_robots=N_ROBOTS),
    n_robots=N_ROBOTS,
    max_steps=500,
)

obs_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

act_space = spaces.Discrete(N_ACTIONS)

observation_spaces = {agent_id: obs_space for agent_id in AGENT_IDS}
state_spaces = {agent_id: obs_space for agent_id in AGENT_IDS}
action_spaces = {agent_id: act_space for agent_id in AGENT_IDS}

models = {}
memories = {}

for agent_id in AGENT_IDS:
    models[agent_id] = {
        "policy": Actor(
            observation_space=observation_spaces[agent_id],
            action_space=action_spaces[agent_id],
            device=DEVICE,
        ),
        "value": Critic(
            observation_space=observation_spaces[agent_id],
            action_space=action_spaces[agent_id],
            device=DEVICE,
        ),
    }

    memories[agent_id] = RandomMemory(
        memory_size=500,
        num_envs=1,
        device=DEVICE,
    )

cfg = IPPO_CFG()

cfg.rollouts = 500
cfg.learning_epochs = 8
cfg.mini_batches = 4
cfg.discount_factor = 0.99
cfg.gae_lambda = 0.95
cfg.learning_rate = 3e-4
cfg.ratio_clip = 0.2
cfg.value_clip = 0.2
cfg.value_loss_scale = 1.0
cfg.entropy_loss_scale = 0.01
cfg.grad_norm_clip = 0.5
cfg.learning_rate_scheduler = None

cfg.learning_rate_scheduler_kwargs = {
    agent_id: {}
    for agent_id in AGENT_IDS
}

cfg.observation_preprocessor_kwargs = {
    agent_id: {}
    for agent_id in AGENT_IDS
}

cfg.state_preprocessor_kwargs = {
    agent_id: {}
    for agent_id in AGENT_IDS
}

cfg.value_preprocessor_kwargs = {
    agent_id: {}
    for agent_id in AGENT_IDS
}

agent = IPPO(
    possible_agents=AGENT_IDS,
    models=models,
    memories=memories,
    cfg=cfg,
    observation_spaces=observation_spaces,
    state_spaces=state_spaces,
    action_spaces=action_spaces,
    device=DEVICE,
)

agent.init()

global_step = 0
TOTAL_TIMESTEPS = 750000

#TOTAL_TIMESTEPS = 500

os.makedirs("weights", exist_ok=True)


def tensor_obs(raw_observations):
    """Convert the list of robot observations to skrl dictionaries."""
    result = {}

    for i, agent_id in enumerate(AGENT_IDS):
        state = np.asarray(
            get_state(raw_observations[i]),
            dtype=np.float32,
        )
        result[agent_id] = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).reshape(1, -1)

    return result


def tensor_scalars(values, dtype=torch.float32):
    """Convert one scalar per robot to a dictionary of tensors."""
    return {
        agent_id: torch.tensor(
            [[values[i]]],
            dtype=dtype,
            device=DEVICE,
        )
        for i, agent_id in enumerate(AGENT_IDS)
    }

def dist_to_target(obs):
    """Manhattan distance between a robot and its target."""
    if isinstance(obs, dict):
        x, y = obs["position"]
        tx, ty = obs["target"]
    else:
        x, y = int(obs[0]), int(obs[1])
        tx, ty = int(obs[4]), int(obs[5])

    return abs(tx - x) + abs(ty - y)


def shape_rewards(obs, next_obs, rewards):
    """Apply progress-based reward shaping to each robot."""
    shaped_rewards = []

    for i in range(N_ROBOTS):
        reward = float(rewards[i])

        if reward == -1.0:
            shaped_reward = -2.0

        elif reward > 0:
            shaped_reward = reward * 2.0

        else:
            d_before = dist_to_target(obs[i])
            d_after = dist_to_target(next_obs[i])

            progress = d_before - d_after
            shaped_reward = 0.2 * progress - 0.05

        shaped_rewards.append(shaped_reward)

    return shaped_rewards


episode = 0

while global_step < TOTAL_TIMESTEPS:
    # Your current Warehouse API appears to return observations only.
    obs, done = env.reset(seed=42 + episode), False

    episode_return = np.zeros(N_ROBOTS, dtype=np.float32)

    while not done and global_step < TOTAL_TIMESTEPS:
        observations = tensor_obs(obs)

        # For IPPO, each critic uses its own same local observation
        # as its corresponding policy.
        states = {
            agent_id: observations[agent_id]
            for agent_id in AGENT_IDS
        }

        agent.pre_interaction(
            timestep=global_step,
            timesteps=TOTAL_TIMESTEPS,
        )

        actions_dict, _ = agent.act(
            observations=observations,
            states=states,
            timestep=global_step,
            timesteps=TOTAL_TIMESTEPS,
        )

        # Convert actions back to the list expected by Warehouse.step()
        actions = [
            int(actions_dict[agent_id].reshape(-1)[0].item())
            for agent_id in AGENT_IDS
        ]

        next_obs, rewards, done, info = env.step(actions)

        shaped_rewards = shape_rewards(obs, next_obs, rewards)

        rewards_dict = tensor_scalars(shaped_rewards)

        next_observations = tensor_obs(next_obs)
        next_states = {
            agent_id: next_observations[agent_id]
            for agent_id in AGENT_IDS
        }

        terminated = {
            agent_id: torch.tensor(
                [[bool(done)]],
                dtype=torch.bool,
                device=DEVICE,
            )
            for agent_id in AGENT_IDS
        }
        truncated = {
            agent_id: torch.zeros(
                (1, 1),
                dtype=torch.bool,
                device=DEVICE,
            )
            for agent_id in AGENT_IDS
        }

        agent.record_transition(
            observations=observations,
            states=states,
            actions=actions_dict,
            rewards=rewards_dict,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos={
                agent_id: {}
                for agent_id in AGENT_IDS
            },
            timestep=global_step,
            timesteps=TOTAL_TIMESTEPS,
        )

        agent.post_interaction(
            timestep=global_step,
            timesteps=TOTAL_TIMESTEPS,
        )

        episode_return += np.asarray(rewards, dtype=np.float32)
        obs = next_obs
        global_step += 1

    episode += 1

    if episode % 10 == 0:
        print(
            f"Episode {episode} | "
            f"Steps {global_step} | "
            f"Returns {episode_return.tolist()}"
        )

# Save the full skrl checkpoint, then also export the individual actors
agent.save("weights/skrl_ippo_checkpoint.pt")

for agent_id in AGENT_IDS:
    policy = agent.models[agent_id]["policy"]
    robot_index = AGENT_IDS.index(agent_id)

    torch.save(
        policy.state_dict(),
        f"weights/skrl_ippo_robot_{robot_index}.pt",
    )

print("Training finished. Weights saved in weights/.")
