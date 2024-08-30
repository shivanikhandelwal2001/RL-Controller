import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from sklearn.preprocessing import MinMaxScaler

from .test_and_plot import plot_predictions, plot_results


class HVACEnv(gym.Env):
    """
    A simple HVAC control environment for reinforcement learning.
    """

    def __init__(self, df):
        super(HVACEnv, self).__init__()
        # Assuming the temperature ranges from 0 to 50 degrees Celsius and humidity from 0% to 100%
        self.df = df
        self.scaler = MinMaxScaler()

        self.beta = 0.4
        self.setpoint_temp = 21
        self.range_around = 2
        self.setpoint_temp_range = list(
            range(
                self.setpoint_temp - self.range_around,
                self.setpoint_temp + self.range_around + 1,
            )
        )
        self.ideal_humidity_lower = 40
        self.ideal_humidity_upper = 60
        self.render_episodes = True

        self.temp_columns = [
            col for col in df.columns if col.endswith("Air_temperature")
        ]
        self.humi_columns = [
            col for col in df.columns if col.endswith("Relative_humidity")
        ]

        self.action_variables = ["Gas_volume"]
        self.state_variables = [
            col
            for col in df.columns
            if (col.endswith("Air_temperature")) or col.endswith("Relative_humidity")
        ]
        self.control_variables = [
            col
            for col in df.columns
            if col not in self.state_variables + self.action_variables
        ]
        self.lagged_features = {
            "lags_1": [col for col in df.columns if col.endswith("_1")],
            "lags_2": [col for col in df.columns if col.endswith("_2")],
        }
        
        self.scaler.fit(self.df[self.control_variables])

        self.lags = 2
        self.current_step = 0
        self.episode = 0

        self.actuals_df = pd.DataFrame(
            columns=self.state_variables + self.action_variables
        )
        self.predictions_df = pd.DataFrame(
            columns=self.state_variables + self.action_variables
        )

        default_obs_bounds = {
            "Air_temperature": (0, 50),
            "Relative_humidity": (0, 100),
            "Surface_temperature": (0, 70),
            "Wind_speed": (0, 50),
            "Total_rainfall": (0, 500),
            "Outside_air_temperature": (-30, 50),
            "Outside_relative_humidity": (0, 100),
        }

        observations_dict = {}
        for col in self.control_variables:
            if "Air_temperature" in col:
                obs_type = "Air_temperature"
            elif "Relative_humidity" in col:
                obs_type = "Relative_humidity"
            elif "Surface_temperature" in col:
                obs_type = "Surface_temperature"
            elif "Wind_speed" in col:
                obs_type = "Wind_speed"
            elif "Total_rainfall" in col:
                obs_type = "Total_rainfall"
            elif "Outside_air_temperature" in col:
                obs_type = "Outside_air_temperature"
            elif "Outside_relative_humidity" in col:
                obs_type = "Outside_relative_humidity"
            else:
                continue

            observations_dict[col] = default_obs_bounds.get(obs_type, (0, 1))

        self.observations_vars = list(observations_dict.keys())
        self.lower_obs_bounds = [
            observations_dict[obs][0] for obs in self.observations_vars
        ]
        self.upper_obs_bounds = [
            observations_dict[obs][1] for obs in self.observations_vars
        ]

        self.action_space = spaces.Box(
            low=np.array([0]), high=np.array([1]), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array(self.lower_obs_bounds),
            high=np.array(self.upper_obs_bounds),
            dtype=np.float32,
        )

    def pass_model_details(self, ml_model, scaler_X, scaler_y, features_order):
        self.ml_model = ml_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.features_order = features_order

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def advance(self, action):
        action_gas_volume = action[0]

        observations_df = self.observations.copy()
        observations_df["Gas_volume"] = action_gas_volume
        observations_df = observations_df[self.features_order]

        actual_next_state = self.df.loc[self.current_step, self.state_variables + self.action_variables].values

        observations_df_scaled = self.scaler_X.transform(observations_df)
        next_state_scaled = self.ml_model.predict(observations_df_scaled)

        predicted_next_state = pd.DataFrame(
            self.scaler_y.inverse_transform(next_state_scaled),
            columns=self.state_variables,
        )
        predicted_next_state["Gas_volume"] = action_gas_volume

        self.actuals.append(actual_next_state)
        self.predictions.append(predicted_next_state.iloc[0].values)

        return predicted_next_state

    def get_observations(self, res):
        current_control_df = self.df.loc[self.current_step, self.control_variables]
        
        if not hasattr(self, 'observations'):
            self.observations = pd.DataFrame([current_control_df])

        if self.current_step % 48 == 0:
            self.observations.loc[0, self.control_variables] = current_control_df.values
        else:
            self.observations.loc[0, [col + "_2" for col in self.state_variables]] = self.observations.loc[0, [col + "_1" for col in self.state_variables]].values
            self.observations.loc[0, [col + "_1" for col in self.state_variables]] = res[self.state_variables].values

            rem_cols = [col for col in self.control_variables if col not in self.lagged_features["lags_1"] + self.lagged_features["lags_2"]]
            self.observations.loc[0, rem_cols] = current_control_df[rem_cols].values
        
        observations = self.observations.values.astype(np.float32)
        observations_norm = self.scaler.transform(observations)[0]

        return observations_norm

    def normalize_discomfort(self, discomfort):
        min_val = np.min(discomfort)
        max_val = np.max(discomfort)

        if max_val == min_val:
            return np.zeros_like(discomfort)

        return (discomfort - min_val) / (max_val - min_val)

    def get_reward(self, res):
        temp_values = res[self.temp_columns].values
        humi_values = res[self.humi_columns].values

        temp_discomfort = np.maximum(
            self.setpoint_temp_range[0] - temp_values,
            temp_values - self.setpoint_temp_range[1],
        )
        temp_discomfort = np.maximum(temp_discomfort, 0) ** 2

        humi_discomfort = np.maximum(40 - humi_values, humi_values - 60)
        humi_discomfort = np.maximum(humi_discomfort, 0) ** 2

        temp_discomfort_normalized = self.normalize_discomfort(temp_discomfort)
        humi_discomfort_normalized = self.normalize_discomfort(humi_discomfort)

        discomfort = np.sum(temp_discomfort_normalized) + np.sum(
            humi_discomfort_normalized
        )

        cost = res["Gas_volume"].iloc[-1]

        reward = -self.beta * cost - (1 - self.beta) * discomfort

        return reward

    def compute_terminated(self):
        terminated = False

        return terminated

    def compute_truncated(self):
        truncated = bool(self.current_step >= len(self.df) - 1)

        return truncated

    def step(self, action):
        res = self.advance(action)
        self.current_step += 1

        reward = self.get_reward(res)
        self.episode_rewards.append(reward)

        terminated = self.compute_terminated()
        truncated = self.compute_truncated()

        info = {
            "info": "Step at step {}".format(self.current_step),
        }

        observations_norm = self.get_observations(res)

        if (terminated or truncated) and self.render_episodes:
            self.render()

        return observations_norm, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.episode += 1

        self.observations = self.df.loc[[self.current_step], self.control_variables]
        
        observations = self.observations.values.astype(np.float32)
        observations_norm = self.scaler.transform(observations)[0]

        self.actuals = []
        self.predictions = []

        reset_info = {
            "info": f"Reset at step {self.current_step}",
        }

        self.objective_integrand = 0
        self.episode_rewards = []

        return observations_norm, reset_info

    def render(self, mode="episodes"):
        if self.episode % 10 == 0:
            plot_predictions(
                self,
                self.episode,
                self.actuals,
                self.predictions,
                self.state_variables,
                self.action_variables,
                save_to_file=True,
            )
        plot_results(self, self.episode, self.episode_rewards)
