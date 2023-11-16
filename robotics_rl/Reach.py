import gymnasium as gym
import numpy as np
# these are new packages in this file
import panda_gym
import stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
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

env = gym.make("PandaReachDense-v3")
observation, info = env.reset(seed=42)
action = env.action_space.sample()
print(f'observation: {observation}')
print(f'action: {action}')

print(f'observation space: {env.observation_space}')
print(f'action space: {env.observation_space}')

def observation_preprocessing(obs) -> np.array:
    obs = np.concatenate([observation['observation'], observation['desired_goal'], observation['achieved_goal']])
    obs_min = np.min(obs)
    obs_max = np.max(obs)
    obs = (obs - obs_min) / (obs_max - obs_min)
    
    return obs

# test the observation_preprocessing function
obs,_ = env.reset()
print(f'obs: {obs}')
preprocessed_obs = observation_preprocessing(obs)
print(f'preprocessed obs: {preprocessed_obs}')
print(f'preprocessed obs length: {len(preprocessed_obs)}')
# well done

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
        self.mu = nn.Linear(256,action_dim)
        self.sigma = nn.Linear(256,action_dim)
        
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.to(device)
        
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # note that the activation function is `tanh` rather than `relu`
        mu = torch.tanh(self.mu(x))
        sigma = torch.exp(torch.clamp(self.sigma(x),min=-20,max=2))
        return mu, sigma
    
    def sample_action(self,state):
        mu,sigma = self(state)
        dist = torch.distributions.Normal(mu,sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(),log_prob.detach()
    
# test Actor Network
state_dim = len(preprocessed_obs)
action_dim = env.action_space.shape[0]
print(f'state dim {state_dim} action dim {action_dim}')
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
actor = Actor(state_dim,action_dim,lr,device)
print(actor)
mu,sigma = actor(torch.from_numpy(preprocessed_obs).float().unsqueeze(0).to(device))
# print(torch.from_numpy(preprocessed_obs).float().unsqueeze(0).shape)
print(f'mu: {mu}')
print(f'sigma: {sigma}')

action, log_prob = actor.sample_action(torch.from_numpy(preprocessed_obs).float().unsqueeze(0).to(device))
print(f'action: {action}')
print(f'log prob: {log_prob}')
# good job!!

class Critic(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float,
                 device):
        
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.to(device)
        
    def forward(self,state,action):
        x = torch.cat([state,action],dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        
        return value
    
# test Critic Network
action = env.action_space.sample()
critic = Critic(state_dim,action_dim,lr,device)
print(critic)
value = critic(torch.from_numpy(preprocessed_obs).float().unsqueeze(0).to(device),torch.tensor(action).float().unsqueeze(0).to(device))
# print(torch.tensor(action).float().unsqueeze(0).shape)
print(f'value: {value}')

# I am the best coder!

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
        self.critic1 = Critic(state_dim,action_dim,critic_lr,device)
        self.critic2 = Critic(state_dim,action_dim,lr,device)
        
        self.target_critic1 = Critic(state_dim,action_dim,lr,device)
        self.target_critic2 = Critic(state_dim,action_dim,lr,device)
        
        
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
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        
        with torch.no_grad():
            next_action, next_log_prob = actor.sample_action(next_state)
            target_Q1 = self.target_critic1(next_state,next_action)
            target_Q2 = self.target_critic2(next_state,next_action)
            # Sum log probabilities across action dimensions
            next_log_prob_sum = next_log_prob.sum(dim=1, keepdim=True)  
            target_V = torch.min(target_Q1,target_Q2) - next_log_prob_sum
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
            action = agent.predict(observation_preprocessing(state))
            
            next_state,reward,terminated,truncated,info = env.step(action)
            done = terminated or truncated
            state_ = observation_preprocessing(state)
            next_state_ = observation_preprocessing(next_state)
            store_transition(state_, action, reward, next_state_, done)
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

RobotAgent = Agent(state_dim,action_dim,actor_lr,critic_lr,gamma,tau,device)
print(RobotAgent)

num_episodes = 10
batch_size = 128
train(RobotAgent,env,num_episodes,batch_size)