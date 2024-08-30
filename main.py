import warnings

import gymnasium as gym
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
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


config = {
    "learning_rate": 1e-3,
    "batch_size": 32,
    "total_timesteps": 10000000,
    "policy": "MlpPolicy",
    "action_noise_sigma": 0.1,
    "device": "cuda",
    "step_size": 1000, 
    "gamma": 0.9,
}

data = get_hvac_data()
ml_model = load("simulation/BuildingA/Building01_linear_regression.joblib")
scaler_X = load("simulation/BuildingA/Building01_linear_regression_scaler_X.joblib")
scaler_y = load("simulation/BuildingA/Building01_linear_regression_scaler_y.joblib")
features_order = load(
    "simulation/BuildingA/Building01_linear_regression_features.joblib"
)

env_id = "CustomEnv-v0"
num_envs = 10


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


def make_env(env_id, rank, ml_model, scaler_X, scaler_y, features_order, seed=0):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.unwrapped.pass_model_details(ml_model, scaler_X, scaler_y, features_order)
        env.unwrapped.seed(seed + rank)  # Set different seed for each environment
        return env

    return _init


if __name__ == "__main__":
    # env = gym.make(env_id)
    # env = Monitor(env)
    # env.unwrapped.pass_model_details(ml_model, scaler_X, scaler_y, features_order)

    env = SubprocVecEnv(
        [
            make_env(env_id, i, ml_model, scaler_X, scaler_y, features_order)
            for i in range(num_envs)
        ]
    )

    wandb.init(
        project="RL-Controller2",
        entity="shivani_2001",
        config=config,
        id="shi4",
        name="V4",
        resume="allow",
        sync_tensorboard=True,
    )

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=config["action_noise_sigma"] * np.ones(n_actions),
    )

    ou_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions), theta=0.15
    )

    model = DDPG(
        config["policy"],
        env,
        action_noise=ou_noise,
        verbose=1,
        tensorboard_log="ddpg_logs",
        device=config["device"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
    )
    
    # Get the optimizers from the model
    actor_optimizer = model.actor.optimizer
    critic_optimizer = model.critic.optimizer

    # Initialize the learning rate schedulers
    actor_scheduler = StepLR(actor_optimizer, step_size=config["step_size"], gamma=config["gamma"])
    critic_scheduler = StepLR(critic_optimizer, step_size=config["step_size"], gamma=config["gamma"])

    eval_callback = EvalCallback(
        env,
        best_model_save_path="ddpg_logs/best_model",
        log_path="ddpg_logs",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path="ddpg_logs", name_prefix="ddpg_checkpoint"
    )

    wandb_callback = WandbCallback(
        model_save_path="ddpg_logs", model_save_freq=10000, verbose=2
    )

    for _ in range(int(config["total_timesteps"] // config["step_size"])):
        model.learn(
            total_timesteps=config["step_size"],
            callback=[eval_callback, wandb_callback],
        )
        
        # Step the scheduler after every interval
        actor_scheduler.step()
        critic_scheduler.step()

    model.save("ddpg_models/ddpg_hvac_control")
    del model

    model = DDPG.load("ddpg_models/ddpg_hvac_control")

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
