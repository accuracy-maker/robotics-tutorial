import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import imageio
import tensorboard
from torch.utils.tensorboard import SummaryWriter

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
action = env.action_space.sample()
print(f'observation: {observation}')
print(f'action: {action}')

print(f'observation space: {env.observation_space}')
print(f'action space: {env.action_space.n}')

# network architecture
class Actor(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float,
                 device: torch.device):
        super(Actor,self).__init__()

        self.fc1 = nn.Linear(state_dim,256)
        self.fc2 = nn.Linear(256,256)
        self.fc_out = nn.Linear(256,action_dim)
        
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.to(device)
        
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        action_probs = F.softmax(self.fc_out(x),dim=-1)
        return action_probs
    
    def sample_action(self,state):
        action_probs = self(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(),log_prob.detach()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
actor = Actor(state_dim,action_dim,lr,device)
action_probs = actor(torch.tensor(observation).float().to(device))
print(action_probs)

class Critic(nn.Module):
    def __init__(self,
                 state_dim: int,
                 lr: float,
                 device):
        
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(state_dim + 1, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.to(device)
        
    def forward(self,state,action):
        if state.dim() == 1:
          state = state.unsqueeze(0)

        # Unsqueeze the action tensor for the same reason
        if action.dim() == 1:
          action = action.unsqueeze(-1)

        x = torch.cat([state,action],dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        
        return value
    
# Ensure observation is a tensor
if not isinstance(observation, torch.Tensor):
    observation = torch.tensor(observation, dtype=torch.float32, device=device)

# Ensure action is a tensor and has the correct shape
if not isinstance(action, torch.Tensor):
    action = torch.tensor([action], dtype=torch.float32, device=device)  # Enclose action in a list to create a 1D tensor

critic = Critic(state_dim,lr,device)
value = critic(observation, action)
print(f'value: {value}')

# bulid the Agent: init, learn, predict
class Agent():
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 actor_lr: float,
                 critic_lr: float,
                 gamma: float,
                 tau: float,
                 device: torch.device):
        
        self.actor = Actor(state_dim,action_dim,actor_lr,device)
        self.critic1 = Critic(state_dim,critic_lr,device)
        self.critic2 = Critic(state_dim,lr,device)
        
        self.target_critic1 = Critic(state_dim,lr,device)
        self.target_critic2 = Critic(state_dim,lr,device)
        
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Initialize target critic networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)
            
    
    def predict(self,state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample_action(state_tensor)
        return action.detach().cpu().numpy()[0]           
    
    def learn(self,state,action,reward,next_state,done):
        # Convert states, actions, rewards, and dones to tensors
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        
        with torch.no_grad():
            next_action, next_log_prob = actor.sample_action(next_state)
            target_Q1 = self.target_critic1(next_state,next_action)
            target_Q2 = self.target_critic2(next_state,next_action)
            # Sum log probabilities across action dimensions
            next_log_prob = next_log_prob.reshape(-1,1)
            target_V = torch.min(target_Q1,target_Q2) - next_log_prob
            target_Q = reward + (1 - done) * self.gamma * target_V
            
        current_Q1 = self.critic1(state,action)
        current_Q2 = self.critic2(state,action)
        
        # critic loss
        critic1_loss = F.mse_loss(current_Q1,target_Q)
        critic2_loss = F.mse_loss(current_Q2,target_Q)
        
        # update critic
        self.critic1.optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1.optimizer.step()
        
        self.critic2.optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2.optimizer.step()
        
        new_action, log_prob = self.actor.sample_action(state)
        Q1_new = self.critic1(state, new_action)
        Q2_new = self.critic2(state, new_action)
        actor_loss = (log_prob - torch.min(Q1_new, Q2_new)).mean()

        # Update actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
# Example usage
buffer_capacity = 10000  # Size of the replay buffer
replay_buffer = ReplayBuffer(buffer_capacity)
print(replay_buffer)
# great work!

# Function to store transitions
def store_transition(state, action, reward, next_state, done):
    replay_buffer.store_transition(state, action, reward, next_state, done)


def should_update(replay_buffer, batch_size):
    return len(replay_buffer) >= batch_size

def sample_batch(replay_buffer, batch_size):
    return replay_buffer.sample(batch_size)

def train(agent,env,num_episodes,batch_size):
    episode_rewards = []
    
    # loop over each episode
    for episode in range(num_episodes):
        state,info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.predict(state)
            
            next_state,reward,terminated,truncated,info = env.step(action)
            done = terminated or truncated
            store_transition(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if should_update(replay_buffer,batch_size):
                states,actions,rewards,next_states,dones =  sample_batch(replay_buffer,batch_size)
                agent.learn(states,actions,rewards,next_states,dones)
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}")
    
    return episode_rewards

actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.99
tau = 0.005

CartPoleAgent = Agent(state_dim,action_dim,actor_lr,critic_lr,gamma,tau,device)

num_episodes = 10
batch_size = 128
train(CartPoleAgent,env,num_episodes,batch_size)