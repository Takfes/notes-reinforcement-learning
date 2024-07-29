import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Set the environment
env = gym.make("CartPole-v0")
env.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, env.action_space.n)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the DDQNAgent
class DDQNAgent:
    def __init__(
        self,
        env,
        replay_buffer,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    ):
        self.env = env
        self.replay_buffer = replay_buffer
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Neural networks for Q and target Q
        self.q_network = QNetwork(env)
        self.target_network = QNetwork(env)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set target network to eval mode

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            self.q_network.train()
            return action_values.max(1)[1].item()  # Exploit

    def train_step(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Get max predicted Q values from target model
        Q_targets_next = (
            self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        )
        # Compute Q targets
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from current model
        Q_expected = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


# Initialize components
replay_buffer = ReplayBuffer(10000)  # Buffer capacity
agent = DDQNAgent(env, replay_buffer)


# Define the training loop
def train_ddqn(agent, env, n_episodes=500, max_t=1000, batch_size=64):
    scores = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        for t in range(max_t):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.train_step(batch_size)

            if done:
                break

        scores.append(total_reward)
        if i_episode % 20 == 0:
            agent.update_target_network()
            print(
                f"Episode {i_episode}/{n_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}"
            )

    return scores


# Train the agent
scores = train_ddqn(agent, env)
