from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import NoopResetEnv
from stable_baselines3 import PPO


# !pip install stable-baselines3
# !apt-get install swig cmake ffmpeg
# !pip install gymnasium[atari]
# !pip install gymnasium[accept-rom-license]

env_id = "Assault-v4"
video_folder = "logs/videos/"
video_length = 2000

vec_env = make_atari_env(env_id, n_envs=1, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

state_space = vec_env.observation_space
action_space = vec_env.action_space
print(f"state space: {state_space}")
print(f'action space: {action_space}')

model = PPO("CnnPolicy", vec_env, verbose=1)
model.learn(total_timesteps=50000)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'mean: {mean_reward}, std: {std_reward}')

record_env = VecVideoRecorder(vec_env,
                              video_folder,
                              record_video_trigger=lambda x: vec_env.get_total_reward() >= 500,
                              video_length=video_length)


obs = record_env.reset()
for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, done, _ = record_env.step(action)
    if done.all():
        break

record_env.close()

# TODO: implement this from scratch
"""
1. atri processing
2. CNN/MLP network
3. the PPO algorithms
"""
