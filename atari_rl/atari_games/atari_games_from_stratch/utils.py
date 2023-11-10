
import datetime
import warnings
from typing import List

import gymnasium as gym
import torch as T
from torch import nn
from torchvision import transforms

from atari_games_from_stratch.base_model import BaseModel



class SimpleCrop(T.nn.Module):
    """
    Crops an image (deterministically) using the transforms.functional.crop function. (No simple crop can be found in
    the torchvision.transforms library
    """

    def __init__(self, top: int, left: int, height: int, width: int) -> None:
        """
        See transforms.functional.crop for parameter descriptions
        """
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img: T.Tensor) -> None:
        """
        Forward pass for input img

        :param img: image tensor
        """
        return transforms.functional.crop(img, self.top, self.left, self.height, self.width)

def annealed_epsilon(step: int, epsilon_start: float, epsilon_stop: float, anneal_finished_step: int) -> float:
    """
    Linear annealed epsilon. See crude ASCII diagram below for depiction of plot of epsilon vs step number.

            |  .
         e  |    .
         p  |      .
         s  |        .
            |          .
            |            . . . . . .
            |_______________________
                   step

    :param step: current step number, to calculate corresponding epsilon value
    :param epsilon_start: starting value for epsilon
    :param epsilon_stop: final value for epsilon
    :param anneal_finished_step: step at which annealing is done; the first step that corresponds to epsilon_stop
    :return: annealed epsilon value
    """
    return epsilon_start + (epsilon_stop - epsilon_start) * min(1, step / anneal_finished_step)


def evaluate_model(model: BaseModel, env: gym.Env, num_episodes: int = 10, max_steps: int = int(1e6)) -> List[float]:
    """
    Evaluate model by rolling out episodes in the env and reporting the episode rewards (returns).

    :param model: model to run
    :param env: environment to run rollouts in
    :param num_episodes: number of episode rollouts to run in env
    :param max_steps: max number of steps for each rollout. If exceeded, then will terminate and return the episode
        reward up to that point
    :return: list of episode rewards (returns)
    """
    with T.no_grad():
        ep_rews = []
        obs = env.reset()
        for _ in range(num_episodes):
            ep_rew = 0
            done = False
            for _ in range(max_steps):
                action = model.predict(obs)
                obs, reward, done, _ = env.step(action)
                ep_rew += reward
                if done:
                    obs = env.reset()
                    break

            ep_rews.append(ep_rew)

            if not done:
                obs = env.reset()
                warnings.warn(
                    f"While evaluating the model, reached max_steps ({max_steps}) before reaching terminal "
                    f"state in env. Terminating it at max_steps."
                )

    return ep_rews

def basic_mlp_network(n_inputs: int, n_outputs: int) -> nn.Module:
    """
    Builds a basic MLP NN with 3 fully connected hidden layers of 64 hidden units each.

    :param n_inputs: number of input units
    :param n_outputs: number of output units
    :return: the constructed neural net
    """
    net = nn.Sequential(
        nn.Linear(n_inputs, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, n_outputs),
    )
    return net


def datetime_string() -> str:
    """
    :return: formatted datetime, as a string
    """
    return (datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")