# https://medium.com/@hkabhi916/mastering-deep-q-learning-with-pytorch-a-comprehensive-guide-a7e690d644fc

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym


# Define Deep Q-Network (DQN) Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define DQN Agent with Experience Replay Buffer
class DQNAgent:
    def __init__(
        self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, buffer_size
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.model(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (
                    reward
                    + self.gamma
                    * torch.max(
                        self.model(torch.tensor(next_state, dtype=torch.float32))
                    ).item()
                )
            target_f = self.model(torch.tensor(state, dtype=torch.float32)).numpy()
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(
                torch.tensor(target_f),
                self.model(torch.tensor(state, dtype=torch.float32)),
            )
            loss.backward()
            self.optimizer.step()
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay


# Initialize environment and agent with Experience Replay Buffer
env = gym.make("Breakout-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(
    state_dim,
    action_dim,
    lr=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    buffer_size=10000,
)

# Train the DQN agent with Experience Replay Buffer
batch_size = 32
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.replay(batch_size)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Evaluate the trained agent
total_rewards = []
num_episodes_eval = 10
for _ in range(num_episodes_eval):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    total_rewards.append(total_reward)
print(f"Average Total Reward (Evaluation): {np.mean(total_rewards)}")
