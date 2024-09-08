import os
import warnings

import gymnasium as gym
import numpy as np
import torch
from joblib import load
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import wandb
from datadrivenHVACEnv import *
from hvac_data import get_hvac_data

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

print(torch.cuda.is_available())

building_id = ["BuildingA", "Building01"]
env_id = "CustomEnv-v0"
num_envs = 10

log_dir = f"ddpg_logs_{building_id[0]}"
model_dir = f"ddpg_models_{building_id[0]}"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

df = get_hvac_data(file_path=f"simulation/{building_id[0]}/{building_id[1]}_Processed.pkl")
ml_model = load(f"simulation/{building_id[0]}/{building_id[1]}_linear_regression.joblib")
scaler_X = load(f"simulation/{building_id[0]}/{building_id[1]}_linear_regression_scaler_X.joblib")
scaler_y = load(f"simulation/{building_id[0]}/{building_id[1]}_linear_regression_scaler_y.joblib")
features_order = load(
    f"simulation/{building_id[0]}/{building_id[1]}_linear_regression_features.joblib"
)

model_config = {
    "learning_rate": 1e-3,
    "batch_size": 32,
    "total_timesteps": 1000000,
    "policy": "MlpPolicy",
    "action_noise_sigma": 0.4,
    "action_noise_theta": 0.25,
    "gamma": 0.999,
    "device": "cuda",
}

env_config = {
    "df": df,
    "n_days": 28,
    "beta": 0.5,
    "setpoint_temp": 21,
    "setpoint_humi": 50,
    "render_episodes": True,
    "controller": "rl",
}

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


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id, **env_config)
        env = Monitor(env)
        env.unwrapped.pass_model_details(ml_model, scaler_X, scaler_y, features_order)
        env.unwrapped.seed(seed + rank)  # Set different seed for each environment
        return env

    return _init


if __name__ == "__main__":
    # env = gym.make(env_id, **env_config)
    # env = Monitor(env)
    # env.unwrapped.pass_model_details(ml_model, scaler_X, scaler_y, features_order)

    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])

    wandb.init(
        project=building_id[0],
        entity="shivani_2001",
        config=model_config,
        id="shi1",
        name="V10",
        sync_tensorboard=True,
    )

    n_actions = env.action_space.shape[-1]
    
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=model_config["action_noise_sigma"] * np.ones(n_actions),
    )

    ou_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=model_config["action_noise_sigma"] * np.ones(n_actions),
        theta=model_config["action_noise_theta"],
    )

    model = DDPG(
        model_config["policy"],
        env,
        action_noise=ou_noise,
        verbose=1,
        tensorboard_log=log_dir,
        device=model_config["device"],
        learning_rate=model_config["learning_rate"],
        batch_size=model_config["batch_size"],
        gamma=model_config["gamma"],
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=log_dir, name_prefix="ddpg_checkpoint"
    )

    wandb_callback = WandbCallback(
        model_save_path=log_dir, model_save_freq=10000, verbose=2
    )

    model.learn(
        total_timesteps=model_config["total_timesteps"],
        callback=[eval_callback, wandb_callback],
    )

    model.save(f"{model_dir}/ddpg_hvac_control")
    del model

    model = DDPG.load(f"{model_dir}/ddpg_hvac_control")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}, Std Reward: {std_reward}")

    wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})
    wandb.finish()

    # state = env.reset()
    # done = True
    # while done:
    #     action, _states = model.predict(state)
    #     state, rewards, done, info = env.step(action)
    #     env.render()
