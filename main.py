from hvac_data import get_hvac_data
import sys
import gym
from custom_envs import *
import time
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from joblib import load
import os


data = get_hvac_data()
ml_model = load('ml_models/Building01_LinearRegression.joblib')
scaler_X = load('scalers/Building01_LinearRegression_Scaler_X.joblib')
scaler_y = load('scalers/Building01_LinearRegression_Scaler_y.joblib')
feature_order = load('feature_order/Building01_LinearRegression_Features.joblib')
target_order = load('feature_order/Building01_LinearRegression_Targets.joblib')


env = gym.make('CustomEnv-v0')
env.pass_df(data)
env.pass_model_details(ml_model, scaler_X, scaler_y, feature_order, target_order)

print(env.observation_space)
print(env.action_space)
print(env.spec)
print(env.spec.max_episode_steps)
print(env.spec.reward_threshold)


state = env.reset()
for _ in range(10):  
    action = env.action_space.sample()  # Replace with your model's action
    next_state, reward, done, info = env.step(action)
    env.render()
    if done:
        break


# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = DDPG(MlpPolicy, env, action_noise=action_noise, verbose=1, tensorboard_log="./ddpg_hvac_tensorboard/")
# model.learn(total_timesteps=10000)

# model.save("ddpg_models/ddpg_hvac_control")
# del model

# model = DDPG.load("ddpg_models/ddpg_hvac_control")

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward}, Std Reward: {std_reward}")

# state = env.reset()
# dones = True
# while dones:
#     action, _states = model.predict(state)
#     state, rewards, done, info = env.step(action)
#     env.render()