import os
import torch as T
import torch.nn.functional as F
import numpy as np
from replay_memory import ReplyBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self,alpha=0.0003,beta=0.0003,input_dims=[8],
                 env=None,gamma=0.99,actions_dim=(3,),max_size=1000000,tau=0.005,
                 layer1_size=256,layer2_size=256,batch_size=256,reward_scale=2):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplyBuffer(max_size,input_dims,actions_dim)
        self.batch_size = batch_size
        self.n_actions = actions_dim
        
        self.actor = ActorNetwork(alpha,input_dims,actions_dim=actions_dim,max_action=env.action_space.high[0],name='actor')
        self.critic1 = CriticNetwork(beta,input_dims,actions_dim=actions_dim,name='critic_1')
        self.critic2 = CriticNetwork(beta,input_dims,actions_dim=actions_dim,name='critic_2')
        self.value = ValueNetwork(beta,input_dims,name='value')
        self.target_value = ValueNetwork(beta,input_dims,name='target_value')
        
        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        
    def choose_action(self,observation):
        state = T.tensor([observation]).to(self.actor.device)
        actions,_ = self.actor.sample_normal(state,reparameterize=False)
        
        return actions.cpu().detach().numpy()[0]
    
    def remember(self,state,action,reward,new_state,done):
        self.memory.store_transition(state,action,reward,new_state,done)
        
    def update_network_parameters(self,tau=None):
        if tau is None:
            tau = self.tau
        
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)
        
        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                (1-tau)*target_value_state_dict[name].clone()
                
        self.target_value.load_state_dict(value_state_dict)
        
    def save_models(self):
        print(".... saving models ....")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        
    def load_models(self):
        print(".... loading models ....")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state,action,reward,new_state,done = self.memory.sample_buffer(self.batch_size)
        
        reward = T.tensor(reward,dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state,dtype=T.float).to(self.actor.device)
        state = T.tensor(state,dtype=T.float).to(self.actor.device)
        action = T.tensor(action,dtype=T.float).to(self.actor.device)
        
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0
        
        actions,log_probs = self.actor.sample_normal(state,reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state,actions)
        q2_new_policy = self.critic2.forward(state,actions)
        critic_value = T.min(q1_new_policy,q2_new_policy)
        critic_value = critic_value.view(-1)
        
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value,value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()
        
        actions, log_probs = self.actor.sample_normal(state,reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state,actions)
        q2_new_policy = self.critic2.forward(state,actions)
        critic_value = T.min(q1_new_policy,q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value # change
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()
        
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic1.forward(state,action).view(-1)
        q2_old_policy = self.critic2.forward(state,action).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q1_old_policy,q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy,q_hat)
        
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()
        
        self.update_network_parameters()
        
# Test the Agent
if __name__ == '__main__':
    import panda_gym
    import gymnasium as gym
    from tqdm import tqdm
    
    env = gym.make("PandaReachDense-v3")

    max_action = env.action_space.high[0]  

    env = gym.make("PandaReachDense-v3")
    observation, info = env.reset(seed=42)
    action = env.action_space.sample()
    # print("Max action:", max_action)
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
    print(f'preprocessed obs shape: {preprocessed_obs.shape}')
    
    agent = Agent(input_dims=preprocessed_obs.shape,env=env,actions_dim=env.action_space.shape)
    print(agent)
        
        
        
        
        