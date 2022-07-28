import torch
import torch.nn as nn
import torch.optim as optim
# from dqn.replay import ReplayMemory
# from dqn.replay import Transition
from itertools import count
import numpy as np
# import torch.nn.functional as F
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_network(nn.Module):
    def __init__(self, n_obs, n_action):
        super(DQN_network, self).__init__()
        self.dqn = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_action)
        )

    def forward(self, x):
        return self.dqn(x)

class DQN_agent():
    def __init__(self, env):
        self.env = env
        self.dqn_policy = DQN_network(2, env.num_actions())
        self.dqn_target = DQN_network(2, env.num_actions())
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.memory = ReplayMemory(10000)
        self.batch_size = 32
        self.target_update = 10
        self.gamma = 0.999
        self.optimizer = optim.Adam(self.dqn_policy.parameters())

        self.device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def choose_action(self, obs):
        return torch.randint(0, 2, (1,), device = self.device)

    def train(self, num_episodes = 1):
        env = self.env
        for i_episode in range(num_episodes):
            obs = env.reset()
            for t in count():
                # Select and perform an action
                action = self.choose_action(obs)
                obs_next, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device = self.device)
                # Store the transition in memory
                self.memory.push(obs, action, obs_next, reward)
                # Move to the next state
                obs = obs_next
                # Perform one step of the optimization (on the policy network)
                self.step_train()
                if done:
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        print('Complete')

    def step_train(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
            batch.next_state)), device = self.device, dtype = torch.bool)
        non_final_next_states = torch.tensor([s for s in batch.next_state
            if s is not None], device = self.device, dtype = torch.float)
        state_batch = torch.tensor(batch.state, device = self.device, dtype = torch.float)
        action_batch = torch.tensor(batch.action, device  = self.device)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        v_lastlayer = self.dqn_policy(state_batch)
        state_action_values = torch.gather(v_lastlayer, dim = 1, index = action_batch.view(-1,1))

        next_state_values = torch.zeros(self.batch_size, device = self.device)
        next_state_values[non_final_mask] = self.dqn_target(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()