"""
resizing,cropping,action repeating
"""
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch as T
from torchvision import transforms

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


ATARI_OBS_SHAPE = (210, 160, 3)
OBS_SEQUENCE_LENGTH = 4  # number of env observations (i.e. frames) to count as one "preprocessed obs" for RL algo
MOD_OBS_SHAPE = (OBS_SEQUENCE_LENGTH, 84, 84)
# Need different image cropping (roughly capturing the playing area of screen) for each env; starting row for crop
CROP_START_ROW = {
    "PongNoFrameskip-v4": 18, "BreakoutNoFrameskip-v4": 18, "BoxingNoFrameskip-v4": 15, "FreewayNoFrameskip-v4": 15
}
NO_OP_ACTION = 0


class PreprocessedAtariEnv(gym.Env):
    """
    Wrapper around Atari gym envs to preprocess the observations (resizing, cropping, action repeat, etc)
    """

    def __init__(
        self, env: gym.Env, action_repeat: int = 4, clip_reward: bool = False, device: T.device = None
    ) -> None:
        """
        :param env: gym environment
        :param action_repeat: repeat action this many times, as one step. See `step` method for details.
        :param clip_reward: whether to clip reward at -1 and +1. This limits scale of errors (potentially better
            training stability), but reduces ability to differentiate actions for large/small rewards. See paper.
        :param device: torch device (cpu or cuda). If None, will automatically select one
        """
        super().__init__()
        self.env = env
        self.clip_reward = clip_reward
        
        if action_repeat >= 1:
            self.action_repeat = action_repeat
            
        else:
            raise ValueError("action repeat must be greater than 1")
        
        if device is None:
            self.device = T.device("cuda" if T.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        game = env.spec.id
        self.preprocess_transform = create_preprocessing_transform(CROP_START_ROW[game])
        
        self.prev_obs: np.ndarray
        self.lastest_obs_maxed_seq = List[np.ndarray] = []
        self.has_been_reset = False
        
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(0,255,MOD_OBS_SHAPE,dtype=np.uint8)
        self.initial_num_lives = initial_num_lives(deepcopy(self.env))
        
    def reset(self) -> np.ndarray:
        """
        Resets the env to initial state, returning the first observation (which is an obs_maxed sequence with a
        preprocessing step; see `step` method for details).

        :return: observation (obs_maxed sequence after preprocessing step)
        """        
        obs = self.env.reset()
        obs_maxed = obs
        self.lastest_obs_maxed_seq = [obs_maxed] * OBS_SEQUENCE_LENGTH
        
        self.prev_obs = obs
        self.has_been_reset = True
        mod_obs = preprocess_obs_maxed_seq(self.lastest_obs_maxed_seq,self.preprocess_transform)
        return mod_obs
    
    def step(self,action: int) -> Tuple[np.ndarray, float, bool, Dict[str,Any]]:
        """
        Takes `action_repeat` steps in the raw env, and returns the effective obs, rew, done, and info

        :return: Tuple of (obs, rew, done, info). `obs` is the sequence of obs_maxed (maxed pixels between this frame
            and previous), after undergoing the preprocessing transformation (grayscale, resize, and crop). `reward`
            is total accrued reward. `done` is whether terminated at any point during the repeat sequence. `info` is
            latest info dictionary. Note: reward is clipped if self.clip_reward is True
        """
        if not self.has_been_reset:
            raise EnvOperationError("Env must be reset before user can call `step`.")
        
        total_rew = 0
        assert self.action_repeat >= 1
        for _ in range(self.action_repeat):
            obs2, rew, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            obs_maxed = np.maximum(self.prev_obs,obs2)
            
            if self.clip_reward:
                rew = np.clip(rew,-1,1)
            total_rew += rew
            
            if info['ale.lives'] != self.initial_num_lives:
                done = True
            
            if done:
                break
            
            self.prev_obs = obs2
        
        self.latest_obs_maxed_seq.pop(0)
        self.latest_obs_maxed_seq.append(obs_maxed)
        mod_obs = preprocess_obs_maxed_seq(self.latest_obs_maxed_seq, self.preprocess_transform, self.device)
        return mod_obs, total_rew, done, info
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the env visually
        
        :param mode: human
        
        """
        self.env.render(mode)
        
def preprocess_obs_maxed_seq(
    obs_maxed_seq: List[np.ndarray], preprocess_transform: Any, device: Optional[T.device] = None
) -> np.ndarray:
    """
    Preprocess the sequence of obs_maxed's, using `preprocess_transform`. Will first load onto `device`, to apply the transformation
    
    :param obs_maxed_seq: sequence of obs_maxed's, which is altogether effectively interpretated as one observation
    :param preprocess_transform: a transform (like from torchvision) that operates on an image
    :param device: torch device to load data on, for applying the transformation. If None, will choose automatically
    :return: modified obs (i.e. obs_maxed_seq after preprocessing)
    """
    
    if len(obs_maxed_seq) != OBS_SEQUENCE_LENGTH:
        raise ValueError("`obs_maxed_seq` must have `OBS_SEQUENCE_LENGTH` number of elements.")
    
    for a in obs_maxed_seq:
        if a.shape != ATARI_OBS_SHAPE:
            raise ValueError("Items in `obs_maxed_seq` must have shape of `ATARI_OBS_SHAPE`.")
    
    if device is None:
        device = T.device("cuda" if T.cuda.is_available() else "cpu")

    obs_maxed_seq_arr = np.array(obs_maxed_seq)
    
    result = T.tensor(obs_maxed_seq)
    orig_device = result.device
    result = result.to(device)
    result = result.permute(0,3,1,2)
    result = preprocess_transform(result)
    result = result.squeeze(1)
    result = result.to(orig_device)
    return np.array(result)

def create_preprocessing_transform(crop_start_row: int) -> Any:
    """
    :param crop_start_row: starting (top-most) row for crop
    :return: preprocessing transform
    """
    return transforms.Compose([transforms.Grayscale(), transforms.Resize((110, 84)), SimpleCrop(crop_start_row, 0, 84, 84)])

    
def initial_num_lives(env: gym.Env) -> int:
    """
    Return the initial number of lives in the env.
    :param env: the Atari env
    :return: initial number of lives
    """
    env.reset()
    action = env.action_space.sample()
    _,_,_,_,info = env.step(action)
    num_lives = info["ale.lives"]
    return num_lives
    
class ReorderedObsAtariEnv(gym.Env):
    """
    Reorder the dimensions of the obs from the Atari env.
    """

    def __init__(self, env, new_ordering=(2, 0, 1)) -> None:
        """
        Reorder the dimensions of the obs, with `new_ordering`.

        :param env: gym env
        :param new_ordering: desired ordering of dimensions of the obs. Default is to order the channels dimensions
            as first (preferred order for torch CNNs), from the default Atari env order (where channels are last).
        """
        super().__init__()
        self.env = env
        self.new_ordering = new_ordering

        orig_obs_shape = self.env.observation_space.shape
        mod_obs_shape = tuple(np.array(orig_obs_shape)[(self.new_ordering,)])

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(0, 255, mod_obs_shape, dtype=np.uint8)

    def reset(self) -> np.ndarray:
        """
        Resets the env, ensuring the obs is reordered.

        :return: reordered observation
        """
        obs = self.env.reset()
        mod_obs = obs.transpose(*self.new_ordering)
        return mod_obs

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Steps in env, ensuring obs is reordered
        :param action: action to take
        :return: tuple (obs, rew, done, info). obs is reordered.
        """
        obs2, rew, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        mod_obs2 = obs2.transpose(*self.new_ordering)
        return mod_obs2, rew, done, info

    def render(self, mode: str = "human") -> None:
        """
        Render the env visually

        :param mode: see gym.Env
        """
        self.env.render(mode)


class EnvOperationError(Exception):
    """
    Exception for when user tries to do an env operation that's not allowed, like step without having reset the env
    """

    pass