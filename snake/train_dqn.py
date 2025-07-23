import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from snake_env import SnakeEnv
from dqn_model import DQN
from replay_buffer import ReplayBuffer
import random
import os


EPISODES = 1000
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

env = SnakeEnv()
state_dim = 13
action_dim = 4  # ['UP', 'DOWN', 'LEFT', 'RIGHT']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer()

start_episode = 0
epsilon = EPS_START

if os.path.exists("../checkpoint.pth"):
    checkpoint = torch.load("../checkpoint.pth")
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epsilon = checkpoint['epsilon']
    start_episode = checkpoint['episode'] + 1
    print(f"Loading checkpoint from episode {checkpoint['episode']}")
else:
    target_net.load_state_dict(policy_net.state_dict())

def select_action(state):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return q_values.argmax().item()

def train():
    if len(memory) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = F.mse_loss(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for episode in range(start_episode, EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = select_action(state)
        next_state, reward, done = env.step(action)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        train()

    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    print(f"Episode {episode}: Reward {total_reward:.2f}, Epsilon {epsilon:.3f}")

    torch.save({
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
        'episode': episode
    }, "../checkpoint.pth")


torch.save(policy_net.state_dict(), "../dqn_snake.pth")
