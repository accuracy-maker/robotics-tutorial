import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    
def observation_preprocessing(obs) -> np.array:
    obs = np.concatenate([obs['observation'], obs['desired_goal'], obs['achieved_goal']])
    obs_min = np.min(obs)
    obs_max = np.max(obs)
    obs = (obs - obs_min) / (obs_max - obs_min)
    
    return obs