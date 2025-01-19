# Notwendige Bibliotheken importieren
import gymnasium as gym
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Q-Network-Definition
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        # Netzwerk mit 3 Schichten definieren
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        # Vorwärtsdurchlauf durch das Netzwerk
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN-Agent
class DQNAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, lr=1e-3, batch_size=32, memory_size=10000, log_dir="./dqn_logs"):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # Haupt- und Zielnetzwerk initialisieren
        self.model = QNetwork(env.observation_space.shape[0], env.action_space.n)
        self.target_model = QNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.update_target_network()  # Zielnetzwerk synchronisieren
        self.writer = SummaryWriter(log_dir=log_dir)  # TensorBoard-Logger

    def update_target_network(self):
        # Zielnetzwerk aktualisieren
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        # Epsilon-greedy Policy
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        # Übergang im Replay-Buffer speichern
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # Erfahrungsspeicher trainieren
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, total_timesteps=10000):
        # Training für gegebene Anzahl an Schritten
        timestep = 0
        episode_rewards = []
        avg_rewards = []

        while timestep < total_timesteps:
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.replay()
                state = next_state
                episode_reward += reward
                timestep += 1

            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_reward)
            self.writer.add_scalar("AvgReward", avg_reward, timestep)
            print(f"Step: {timestep}, AvgReward: {avg_reward:.2f}")

            if timestep % 1000 == 0:
                self.update_target_network()

        self.writer.close()
        self.plot_training_results(episode_rewards, avg_rewards)

    def plot_training_results(self, rewards, avg_rewards):
        # Ergebnisse mit matplotlib plotten
        plt.figure(figsize=(12, 6))
        plt.plot(rewards, label="Episodenbelohnung", alpha=0.6)
        plt.plot(avg_rewards, label="100-Episoden Durchschnitt", linewidth=2)
        plt.title("Trainingsbelohnungen über Zeit")
        plt.xlabel("Episode")
        plt.ylabel("Belohnung")
        plt.legend()
        plt.grid()
        plt.show()

# Umgebung definieren und Agent trainieren
env = gym.make("CartPole-v1")
agent = DQNAgent(env)
agent.train(total_timesteps=50000)
