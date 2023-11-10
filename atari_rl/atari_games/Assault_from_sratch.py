import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple,deque
import random

# define the DQN Network
class DQN_cnn(nn.Module):
    def __init__(self,
                 height: int,
                 width: int,
                 num_frames: int,
                 num_actions: int,
                 device):
        super(DQN_cnn,self).__init__()
        
        self.height = height
        self.width = width
        
        # Layer
        self.cnn1 = nn.Conv2d(
            in_channels=num_frames,
            out_channels=16,
            kernel_size=8,
            stride=4,
            padding=2
        )
        
        self.cnn2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
        )
        
        self.fc1 = nn.Linear(
            in_features= 16640,
            out_features=256
        )
        
        self.fc2 = nn.Linear(
            in_features=256,
            out_features=num_actions
        )
        
        # Activation Functions
        self.relu = nn.ReLU()
        
        self.device = device
        self.to(self.device)
        
    def flatten(self,x):
        batch_size = x.size()[0]
        x = x.view(batch_size,-1)
        return x
    
    def forward(self,x):
        # we use the "grayscale observation"
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def transform_to_grayscale(x):
    # x: [height, weight, channel]
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(x)
    # convert the PIL image to grayscale
    to_gray = transforms.Grayscale(num_output_channels=1)
    gray_image_pil = to_gray(image_pil)
    
    # to tensor
    to_tensor = transforms.ToTensor()
    gray_image_tensor = to_tensor(gray_image_pil)
    return gray_image_tensor

def transform_batch_to_grayscale(batch):
    gray_images_list = [transform_to_grayscale(image) for image in batch]
    gray_images_tensor = torch.stack(gray_images_list)
    return gray_images_tensor

Transition = namedtuple('Transition',('obs','action','next_obs','reward','done'))
class Replay_Buffer(object):
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)
        
    def push(self,*args):
        self.memory.append(Transition(*args))
    
    # there is a little difference from normal sample function
    # Because images should not be shuffled due to time propoerities
    def sample(self,batch_size):
        if len(self.memory) < batch_size:
            raise ValueError("Not enough data in the replay buffer to sample from.")

        start_index = random.randint(0, len(self.memory) - batch_size)
        return [self.memory[i] for i in range(start_index, start_index + batch_size)]
    
    def __len__(self):
        return len(self.memory)

class agent():
    def __init__(
        self,
        env,
        # Parameters of model
        height: int,
        width: int,
        num_frames: int,
        num_actions: int,
        # Parameters of RL
        gamma: float,
        tau: float,
        batch_size: int,
        min_epsilon: float,
        max_epsilon: float,
        decay_rate: float,
        buffer_size: int,
        # Parameters of training
        n_training_episodes: int,
        n_eval_episodes: int,
        max_step: int,
        device
    ):
        self.env = env
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_actions = num_actions
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = decay_rate
        self.n_training_episodes = n_training_episodes
        self.n_eval_episodes = n_eval_episodes
        self.max_step = max_step
        self.device = device
        
        # DQN_CNN
        self.model = DQN_cnn(height=self.height,
                             width=self.width,
                             num_frames=self.num_frames,
                             num_actions=self.num_actions,
                             device=self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        self.loss = nn.MSELoss()
        
        # Buffer
        self.memory = Replay_Buffer(buffer_size)
        
        
    def act(self,state,ep = 0):
        # choose action given a state
        p = random.random()
        gray_obs = transform_to_grayscale(state)
        
        if p > ep:
             action_value = self.model(gray_obs)
             action = torch.argmax(action_value).item()
             return action
        else:
            return self.env.action_space.sample()
        
    def step(self,state,ep = 0):
        action = self.act(state,ep)
        next_state,reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.memory.push(state,action,next_state,reward,done)
        return next_state,reward,terminated,truncated

    def update(self):
        if self.memory.__len__() >= self.batch_size:
            self.optimizer.zero_grad()
            
            transition = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transition))
            
            b_states = transform_batch_to_grayscale(list(batch.obs)).to(self.device)
            b_actions = np.array(batch.action)
            b_rewards = torch.tensor(batch.reward).view(self.batch_size,-1).to(self.device)
            b_next_states = transform_batch_to_grayscale(list(batch.next_obs)).to(self.device)
            b_dones = torch.tensor(batch.done).to(self.device)
            
            batch_idx = np.arange(self.batch_size,dtype=np.int32)
            pred_q_value = self.model(b_states)[batch_idx,b_actions]    
            next_max_q_value = self.model(b_next_states)
            next_max_q_value[b_dones] = 0.0
            
            # a = torch.max(next_max_q_value,dim = 1)[0]
            target_q_value = b_rewards.reshape(-1) + self.gamma * torch.max(next_max_q_value,dim = 1)[0]
            loss = self.loss(pred_q_value,target_q_value).to(self.device)
          
            loss.backward()
            self.optimizer.step()
        else:
            return
        
    def train(self):
        scores,avg_scores = [],[]
        for episode in range(self.n_training_episodes):
            # we should update the epsilon at very episode firstly
            ep = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
            print(f'episode: {episode} | epsilon: {ep}')
            
            state,info = self.env.reset()
            step = 0
            score = 0
            terminated = False
            truncated = False
            
            for step in range(self.max_step):
                next_state,reward,terminated,truncated = self.step(state,ep = ep)
                self.update()
                score += reward
                state = next_state
                # scores.append(score)
                # scores_window.append(score)
                
                if terminated or truncated:
                    break
                
                # state = next_state
            
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            
            if episode % 1 == 0:
                # print(f'episode: {episode} | average score: {np.mean(scores_window)}')
                print(f'episode: {episode} | score: {score} | avg score: {avg_score}')
                
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        fig.suptitle('Training Progress')

        ax1.plot(scores, label='Episode Scores', color='b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('scores', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2.plot(avg_scores, label='Episode Average Score', color='r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Score', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.show()
            
if __name__ == "__main__":
    env_id = 'Assault-v4'
    env = gym.make(env_id)
    # obs_space = env.observation_space
    # act_space = env.action_space
    # print(f'obs_space: {obs_space}')
    # print(f'act_space: {act_space}')
    # # TODO: grayscale
    obs,info = env.reset()
    # print(f'observation shape: {obs.shape}')
    # gray_obs = transform_to_grayscale(obs)
    height,width = obs.shape[0],obs.shape[1]
    num_frame = 1
    num_actions = env.action_space.n
    
    # model = DQN_cnn(
    #     height=height,
    #     width=width,
    #     num_frames=num_frame,
    #     num_actions=num_actions
    # )
    # print(model)
    # y = model(gray_obs.unsqueeze(0))
    # print(y.shape)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    n_training_episodes = 300
    n_eval_episodes = 100
    lr = 0.001
    max_step = 100000
    gamma = 0.95
    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.005
    buffer_size = 10000
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    Agent = agent(env,
                  height,
                  width,
                  num_frame,
                  num_actions,
                  gamma,
                  lr,
                  batch_size,
                  min_epsilon,
                  max_epsilon,
                  decay_rate,
                  buffer_size,
                  n_training_episodes,
                  n_eval_episodes,
                  max_step,
                  device)
    
    Agent.train()