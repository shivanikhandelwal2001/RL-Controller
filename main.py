import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from joblib import load
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.ddpg.policies import MlpPolicy

from datadrivenHVACEnv import *
from hvac_data import get_hvac_data

print(torch.cuda.is_available())


data = get_hvac_data()
ml_model = load("simulation/BuildingA/Building01_linear_regression.joblib")
scaler_X = load("simulation/BuildingA/Building01_linear_regression_scaler_X.joblib")
scaler_y = load("simulation/BuildingA/Building01_linear_regression_scaler_y.joblib")
features_order = load(
    "simulation/BuildingA/Building01_linear_regression_features.joblib"
)

# env = gym.make('CustomEnv-v0')
# env.pass_df(data)
# env.pass_model_details(ml_model, scaler_X, scaler_y)

# state = env.reset()
# for _ in range(6000):
#     action = env.action_space.sample()  # Replace with your model's action
#     next_state, reward, _, done, info = env.step(action)
#     env.render()
#     if done:
#         break

env = gym.make("CustomEnv-v0")
env = Monitor(env)
env.unwrapped.pass_model_details(ml_model, scaler_X, scaler_y, features_order)
state = env.reset()

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)

model = DDPG(
    MlpPolicy,
    env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./ddpg_logs/",
    device="cuda",
    learning_rate=1e-5,
    batch_size=128,
)

eval_callback = EvalCallback(
    env,
    best_model_save_path="./ddpg_logs/best_model",
    log_path="./ddpg_logs/",
    eval_freq=8000,
    deterministic=True,
    render=False,
)
checkpoint_callback = CheckpointCallback(
    save_freq=20000, save_path="./ddpg_logs/", name_prefix="ddpg_checkpoint"
)

model.learn(total_timesteps=1000000, callback=[eval_callback, checkpoint_callback])

model.save("ddpg_models/ddpg_hvac_control")
del model

model = DDPG.load("ddpg_models/ddpg_hvac_control")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std Reward: {std_reward}")

# state = env.reset()
# done = True
# while done:
#     action, _states = model.predict(state)
#     state, rewards, done, info = env.step(action)
#     env.render()
