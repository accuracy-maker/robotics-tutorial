import gymnasium as gym
import numpy as np
import panda_gym
from sac_torch import Agent
from utils import plot_learning_curve,observation_preprocessing
import os

if __name__ == '__main__':
    env = gym.make('PandaReachDense-v3')
    obs,_ = env.reset()
    obs = observation_preprocessing(obs)
    agent = Agent(input_dims=obs.shape,
                  env = env,
                  actions_dim=env.action_space.shape
                  )
    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists('tmp/sac'):
        os.makedirs('tmp/sac')
    n_games = 500
    filename = 'PandaReachDense-v3.png'
    figure_file = 'plots/' + filename
    
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        # env.render(mode='human')

    for i in range(n_games):
        obs,_ = env.reset()
        done = False
        score = 0
        while not done:
            p_obs = observation_preprocessing(obs)
            action = agent.choose_action(p_obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            p_obs_ = observation_preprocessing(obs_)
            score += reward
            agent.remember(p_obs, action, reward, p_obs_, done)
            if not load_checkpoint:
                agent.learn()
            obs = obs_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)