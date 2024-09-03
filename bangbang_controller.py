import os
import warnings

import gymnasium as gym
import numpy as np
from joblib import load
from datadrivenHVACEnv import *


class BangBangController:
    """
    A simple Bang-Bang Controller for HVAC systems that toggles gas volume on or off
    based on the deviation of indoor temperature from the setpoint.
    """

    def __init__(self, setpoint_temp=21, tolerance=2, max_gas_volume=1.0, min_gas_volume=0.0):
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
        self.max_gas_volume = max_gas_volume
        self.min_gas_volume = min_gas_volume

    def compute_action(self, current_temp):
        """
        Compute the gas volume action based on the current temperature.

        Parameters:
        - current_temp (float): The current indoor temperature.

        Returns:
        - action (float): The gas volume to be used (either max or min).
        """
        if current_temp < (self.setpoint_temp - self.tolerance):
            # Temperature is too low; turn heating fully on
            return self.max_gas_volume
        elif current_temp > (self.setpoint_temp + self.tolerance):
            # Temperature is too high; turn heating off
            return self.min_gas_volume
        else:
            # Within the acceptable range; maintain current gas volume
            return 0.5 # or max_gas_volume, depending on your setup


ml_model = load("simulation/BuildingB/Building08_linear_regression.joblib")
scaler_X = load("simulation/BuildingB/Building08_linear_regression_scaler_X.joblib")
scaler_y = load("simulation/BuildingB/Building08_linear_regression_scaler_y.joblib")
features_order = load(
    "simulation/BuildingB/Building08_linear_regression_features.joblib"
)

log_dir = "bb_logs"
os.makedirs(log_dir, exist_ok=True)

env_id = "CustomEnv-v0"
bang_bang_controller = BangBangController()

env = gym.make(env_id)
env.unwrapped.pass_model_details(ml_model, scaler_X, scaler_y, features_order)

state = env.reset()

for _ in range(6000):
    action = bang_bang_controller.compute_action(state)
    state, reward, _, done, info = env.step([action])
    if done:
        break
