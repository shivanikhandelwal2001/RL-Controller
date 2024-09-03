import os
import warnings

import gymnasium as gym
import numpy as np
from joblib import load

from datadrivenHVACEnv import *
from hvac_data import get_hvac_data

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

class BangBangController:
    """
    A simple Bang-Bang Controller for HVAC systems that toggles gas volume on or off
    based on the deviation of indoor temperature from the setpoint.
    """

    def __init__(self, setpoint_temp=21, tolerance=1):
        """
        Initialize the Bang-Bang Controller.

        Parameters:
        - setpoint_temp (float): The desired temperature setpoint.
        - tolerance (float): The acceptable deviation from the setpoint.
        - max_gas_volume (float): The maximum gas volume (fully on).
        - min_gas_volume (float): The minimum gas volume (fully off).
        """
        self.setpoint_temp = setpoint_temp
        self.tolerance = tolerance
        self.action = 0

    def compute_action(self, current_temp):
        """
        Compute the gas volume action based on the current temperature.

        Parameters:
        - current_temp (float): The current indoor temperature.

        Returns:
        - action (float): The gas volume to be used (either max or min).
        """
        if (current_temp < (self.setpoint_temp - self.tolerance)) and (
            self.action == 0
        ):
            self.action = 1
        elif (current_temp > (self.setpoint_temp + self.tolerance)) and (
            self.action == 1
        ):
            self.action = 0

        return self.action


df = get_hvac_data(file_path="simulation/BuildingB/Building08_Processed.pkl")
ml_model = load("simulation/BuildingB/Building08_linear_regression.joblib")
scaler_X = load("simulation/BuildingB/Building08_linear_regression_scaler_X.joblib")
scaler_y = load("simulation/BuildingB/Building08_linear_regression_scaler_y.joblib")
features_order = load(
    "simulation/BuildingB/Building08_linear_regression_features.joblib"
)

env_id = "CustomEnv-v0"
bang_bang_controller = BangBangController()

env_config = {
    "df": df,
    "n_days": 28,
    "beta": 0.5,
    "setpoint_temp": 21,
    "setpoint_humi": 50,
    "render_episodes": True,
    "controller": "bang_bang",
}

env = gym.make(env_id, **env_config)
env.unwrapped.pass_model_details(ml_model, scaler_X, scaler_y, features_order)

state = env.reset()
print(state[1]["info"])

for _ in range(6000):
    action = bang_bang_controller.compute_action(state[0])
    state, reward, _, done, info = env.step([action])
    if done:
        break
