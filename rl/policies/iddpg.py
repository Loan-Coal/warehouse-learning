"""Independent DDPG-style policy for one warehouse robot."""

import copy
import random
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class PolicyNN(nn.Module):
    def __init__(self, obs_dim=14, action_dim=6, hidden_dim=128):
        super(PolicyNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Initialize the weights and bias of BatchNorm1d
        #self.model[0].weight.data.fill_(1)
        #self.model[0].bias.data.fill_(0)
    
    def forward(self, state):
        return self.model(state)
    
class CriticNN(nn.Module):
    def __init__(self, obs_dim = 14, action_dim=6, hidden_dim=128):
        super(CriticNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize the weights and bias of BatchNorm1d
        #self.model[0].weight.data.fill_(1)
        #self.model[0].bias.data.fill_(0)
    
    def forward(self, obs, action):
        state_action = torch.cat((obs, action), dim=1)
        return self.model(state_action)

class DDPGAgent(object):
    def __init__(self, obs_dim=14, action_dim=6, hidden_dim=128, gamma=0.95, tau=0.01, lr=0.001, batch_size=64, buffer_size=10000, epsilon=0.1, discrete_action=False):

        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.policy_nn = PolicyNN(obs_dim, action_dim, hidden_dim)
        self.critic_nn = CriticNN(obs_dim, action_dim, hidden_dim)
        self.target_policy_nn = copy.deepcopy(self.policy_nn)
        self.target_critic_nn = copy.deepcopy(self.critic_nn)

        self.policy_optim = torch.optim.Adam(params=self.policy_nn.parameters(), lr = lr)
        self.critic_optim = torch.optim.Adam(params=self.critic_nn.parameters(), lr = lr)

        #if Config.gpu_process == 0:
            #self.policy_nn, self.policy_optim = ipex.optimize(self.policy_nn, optimizer=self.policy_optim, dtype=torch.bfloat16)
            #self.critic_nn, self.critic_optim = ipex.optimize(self.critic_nn, optimizer=self.critic_optim, dtype=torch.bfloat16)
            #self.policy_nn, self.policy_optim = ipex.optimize(self.policy_nn, optimizer=self.policy_optim, dtype=torch.float32)
            #self.critic_nn, self.critic_optim = ipex.optimize(self.critic_nn, optimizer=self.critic_optim, dtype=torch.float32)

        self.models = {
            'policy_nn': self.policy_nn,
            'target_policy_nn': self.target_policy_nn,
            'critic_nn': self.critic_nn,
            'target_critic_nn': self.target_critic_nn
        }

        # Each agent has its own experience buffer.
        self.replay_buffer = deque(maxlen=buffer_size)

    def act(self, obs):
        """Choose one of the six discrete actions."""

        # Explore: occasionally try a random action.
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        obs_tensor = torch.as_tensor(
            np.asarray(obs, dtype=np.float32)
        ).unsqueeze(0)

        with torch.no_grad():
            scores = self.policy_nn(obs_tensor)

        return int(scores.argmax(dim=1).item())

    def update(self, obs, action, reward, next_obs, done):
        """Store an experience and train from a random batch."""

        self.replay_buffer.append(
            (
                np.asarray(obs, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                bool(done),
            )
        )

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_b, actions_b, rewards_b, next_obs_b, dones_b = zip(*batch)

        obs_t = torch.as_tensor(np.asarray(obs_b), dtype=torch.float32)
        next_obs_t = torch.as_tensor(
            np.asarray(next_obs_b), dtype=torch.float32
        )
        actions_t = torch.as_tensor(actions_b, dtype=torch.long)
        rewards_t = torch.as_tensor(
            rewards_b, dtype=torch.float32
        ).unsqueeze(1)

        # Represent each discrete action as a one-hot vector.
        actions_onehot = F.one_hot(
            actions_t, num_classes=self.action_dim
        ).float()

        # ----- Update critic -----

        with torch.no_grad():
            next_scores = self.target_policy_nn(next_obs_t)
            next_actions = F.one_hot(
                next_scores.argmax(dim=1),
                num_classes=self.action_dim,
            ).float()

            next_q = self.target_critic_nn(next_obs_t, next_actions)

            # done means time-limit truncation in this project.
            # Continue bootstrapping from the next observation.
            target_q = rewards_t + self.gamma * next_q

        current_q = self.critic_nn(obs_t, actions_onehot)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_nn.parameters(), max_norm=0.5
        )
        self.critic_optim.step()

        # ----- Update actor -----

        scores = self.policy_nn(obs_t)

        # Hard one-hot actions in the forward pass, with an
        # approximate gradient for discrete-action learning.
        proposed_actions = F.gumbel_softmax(
            scores, tau=1.0, hard=True, dim=1
        )

        actor_loss = -self.critic_nn(obs_t, proposed_actions).mean()

        self.policy_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_nn.parameters(), max_norm=0.5
        )
        self.policy_optim.step()

        # ----- Update target networks -----

        self._soft_update(self.target_policy_nn, self.policy_nn)
        self._soft_update(self.target_critic_nn, self.critic_nn)

    def _soft_update(self, target, source):
        with torch.no_grad():
            for target_param, source_param in zip(
                target.parameters(), source.parameters()
            ):
                target_param.mul_(1.0 - self.tau)
                target_param.add_(self.tau * source_param)


class IDDPG(object):
    def __init__(self, num_agents, **agent_kwargs):
        """
        Inputs:
            num_agents (int): number of agents in the environment
            agent_kwargs: keyword arguments for DDPGAgent constructor
        """
        # Each list entry creates new networks and a new replay buffer.
        self.agents = [
            DDPGAgent(**agent_kwargs)
            for _ in range(num_agents)
        ]

    def act(self, agent_i, obs):
        return self.agents[agent_i].act(obs)

    def update(self, agent_i, obs, action, reward, next_obs, done):
        self.agents[agent_i].update(
            obs, action, reward, next_obs, done
        )

    def make_policies(self):
        """Create one warehouse-compatible policy per robot."""
        return [
            IDDPGPolicy(self, agent_i)
            for agent_i in range(len(self.agents))
        ]


    def act_all(self, observations):
        """Choose one action for each robot."""
        return [
            agent.act(obs)
            for agent, obs in zip(self.agents, observations)
        ]

    def update_all(self, observations, actions, rewards, next_observations, done):
        """Update each robot using its own transition."""
        for i, agent in enumerate(self.agents):
            agent.update(
                observations[i],
                actions[i],
                rewards[i],
                next_observations[i],
                done,
            )

    def save(self, path):
        """Save all agents and their learned networks."""
        torch.save(
            {
                "num_agents": len(self.agents),
                "agents": [
                    {
                        "actor": agent.policy_nn.state_dict(),
                        "critic": agent.critic_nn.state_dict(),
                        "target_actor": agent.target_policy_nn.state_dict(),
                        "target_critic": agent.target_critic_nn.state_dict(),
                    }
                    for agent in self.agents
                ],
            },
            path,
        )


class IDDPGPolicy:
    def __init__(self, iddpg, agent_i):
        self.iddpg = iddpg
        self.agent_i = agent_i

    def act(self, obs):
        return self.iddpg.act(self.agent_i, obs)

    def update(self, obs, action, reward, next_obs, done):
        self.iddpg.update(
            self.agent_i, obs, action, reward, next_obs, done
        )

    def reset(self):
        # The agent has no episode-specific hidden state.
        pass
