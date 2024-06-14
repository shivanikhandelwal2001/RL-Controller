import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

beta = 0.6
setpoint_temp = 23
range_around = 2
setpoint_temp_range = list(range(setpoint_temp-range_around, setpoint_temp+range_around + 1))
ideal_humidity_lower = 40
ideal_humidity_upper = 60

class HVACEnv(gym.Env):
    """
    A simple HVAC control environment for reinforcement learning.
    """
    def __init__(self, df):
        self.seed()
        super(HVACEnv, self).__init__()
        # Assuming the temperature ranges from 0 to 50 degrees Celsius and humidity from 0% to 100%
        self.df = df
        
        temp_columns = [col for col in df.columns if col.endswith("Air_temperature")]
        humi_columns = [col for col in df.columns if col.endswith("Relative_humidity")]
        n_temp, n_humi = len(temp_columns), len(humi_columns)
        
        self.state_variables = [col for col in df.columns if (col.endswith("Air_temperature")) or col.endswith("Relative_humidity")]
        self.control_variables = [col for col in df.columns if col not in self.state_variables]
        
        self.max_lag = 2
        self.rolling_window_sizes = [12, 48]
        self.current_step = 3
        # self.buffer_df = pd.DataFrame(columns=self.state_variables)
        # initial_rows = min(self.current_step, len(df))
        self.refresh_step = 0
        self.max_refresh_step = 6
        
        self.buffer_df = self.df[self.state_variables].head(self.current_step)
        
        self.actuals_df = pd.DataFrame(columns=self.state_variables+["Gas_volume"])
        self.predictions_df = pd.DataFrame(columns=self.state_variables+["Gas_volume"])
        
        low_bounds = np.array([0] * n_temp + [0] * n_humi)
        high_bounds = np.array([50] * n_temp + [100] * n_humi)
        
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([2]), dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        
        self.reward_range=(-10,10)
        self.state = self.observation_space.sample()
    
        
    def pass_df(self, df):
        self.df = df
        print(self.df.head())    
    
        
    def pass_model_details(self, ml_model, scaler_X, scaler_y, feature_order, target_order):
        self.ml_model = ml_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.feature_order = feature_order
        self.target_order = target_order    
    
    
    def _take_action(self):
        self.control = self.df[self.control_variables].iloc[self.current_step]
        gas_volume = self.action[0]
        updated_features = self.generate_features_for_next_state()
        
        for feature, value in updated_features.items():
            self.control[feature] = value
        
        actual_gas_volume = self.control['Gas_volume']
        self.control['Gas_volume'] = gas_volume
        self.control = pd.DataFrame([self.control[self.feature_order]])
        print(f"Control Variables for ML Model Prediction {self.control}")
        
        actual_next_state = self.df[self.state_variables].iloc[self.current_step]
        actual_next_state["Gas_volume"] = actual_gas_volume
        
        control_scaled = self.scaler_X.transform(self.control)
        
        predicted_next_state = self.ml_model.predict(control_scaled)
        predicted_next_state = self.scaler_y.inverse_transform(predicted_next_state)
        
        self.predicted_next_state_series = pd.Series(predicted_next_state.flatten(), index=self.target_order)
        self.predicted_next_state_series["Gas_volume"] = gas_volume
        
        self.actuals_df = pd.concat([self.actuals_df, pd.DataFrame([actual_next_state])], ignore_index=True)
        self.predictions_df = pd.concat([self.predictions_df, pd.DataFrame([self.predicted_next_state_series])], ignore_index=True)
        
        # if self.refresh_step < self.max_refresh_step:
        #     self.refresh_step += 1
        self.update_buffer(self.predicted_next_state_series)
            
        # else:
        #     self.refresh_step = 0
        #     self.update_buffer(self.df[self.state_variables].iloc[self.current_step])
    
    
    def _calculate_reward(self):
        temp_comfort_scores = []
        humidity_comfort_scores = []

        predicted_temps = self.predicted_next_state_series.filter(like='Air_temperature').values
        predicted_humidities = self.predicted_next_state_series.filter(like='Relative_humidity').values
        
        for predicted_temp in predicted_temps:
            if predicted_temp in setpoint_temp_range:
                temp_comfort = 2 - abs(setpoint_temp - predicted_temp)
            else:
                temp_comfort = -(abs(min(setpoint_temp_range, key=lambda x: abs(x - predicted_temp)) - predicted_temp))
            temp_comfort_scores.append(temp_comfort)
        
        for predicted_humidity in predicted_humidities:
            if ideal_humidity_lower <= predicted_humidity <= ideal_humidity_upper:
                humidity_comfort = 1 
            else:
                if predicted_humidity < ideal_humidity_lower:
                    humidity_comfort = -(ideal_humidity_lower - predicted_humidity) / 20 
                else:
                    humidity_comfort = -(predicted_humidity - ideal_humidity_upper) / 20 
            humidity_comfort_scores.append(humidity_comfort)
    
        avg_temp_comfort = sum(temp_comfort_scores) / len(temp_comfort_scores)
        avg_humidity_comfort = sum(humidity_comfort_scores) / len(humidity_comfort_scores)
    
        # print(f"Avg Temp and Humi for Comfort {avg_temp_comfort} {avg_humidity_comfort}")
        combined_comfort = (avg_temp_comfort + avg_humidity_comfort) / 2

        self.reward = -(beta * self.action) + (1 - beta) * combined_comfort
        
        
    def _calculate_reward2(self):
        total_temp_discomfort = 0
        total_humidity_discomfort = 0

        predicted_temps = self.predicted_next_state_series.filter(like='Air_temperature').values
        predicted_humidities = self.predicted_next_state_series.filter(like='Relative_humidity').values
        
        for predicted_temp in predicted_temps:
            temp_discomfort = (predicted_temp - setpoint_temp) ** 2
            total_temp_discomfort += temp_discomfort
        
        for predicted_humidity in predicted_humidities:
            if predicted_humidity < ideal_humidity_lower:
                humidity_discomfort = (ideal_humidity_lower - predicted_humidity) ** 2
            elif predicted_humidity > ideal_humidity_upper:
                humidity_discomfort = (predicted_humidity - ideal_humidity_upper) ** 2
            else:
                humidity_discomfort = 0 
            total_humidity_discomfort += humidity_discomfort
    
        normalized_temp_discomfort = total_temp_discomfort / len(predicted_temps)
        normalized_humidity_discomfort = total_humidity_discomfort / len(predicted_humidities)
    
        print(f"Avg Temp and Humi for Discomfort {normalized_temp_discomfort} {normalized_humidity_discomfort}")
        overall_discomfort = normalized_temp_discomfort + normalized_humidity_discomfort

        self.reward = -(beta * self.action) + (1 - beta) * overall_discomfort
    
            
    def step(self, action):
        self.action = action
        
        self._take_action()
        self._calculate_reward()
            
        # done = (self.reward>2.5) or (self.current_step >= len(self.df.index) - 1)
        done = (self.reward>2.5) or (self.current_step >= 20)
        next_state = self.extract_next_state_from_buffer()
        self.state = next_state
        
        print(f"Step: {self.current_step}")
        print(f"State: {self.state}")
        print(f"Action: {self.action}")
        print(f"Reward: {self.reward}")
        
        self.current_step += 1
        
        return next_state, self.reward, done, {}
    
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed] 
    
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 3
        self.state = self.df[self.state_variables].iloc[self.current_step].values
        self.buffer_df = self.df[self.state_variables].head(self.current_step)
        
        return self.state
    
    
    def update_buffer(self, new_row):
        # Update the buffer with new observations or predictions
        if isinstance(new_row, dict):
            new_row = pd.Series(new_row)
            
        self.buffer_df = pd.concat([self.buffer_df, pd.DataFrame([new_row[self.state_variables]])], ignore_index=True)
        
        max_size = max(self.max_lag, max(self.rolling_window_sizes))
        max_size = self.max_lag
        if len(self.buffer_df) > max_size:
            self.buffer_df.drop(self.buffer_df.index[:len(self.buffer_df)-max_size], inplace=True)

    
    def extract_next_state_from_buffer(self):
    # Extracts the most recent observations from the buffer to form the next state.
        next_state = self.buffer_df.iloc[-1][self.state_variables].values
        return next_state
    

    def generate_features_for_next_state(self):
        # Generate lag and rolling features based on the current buffer.
        features = {}
        
        for col in self.state_variables:
            for lag in range(1, self.max_lag+1):
                if len(self.buffer_df) >= lag:
                    features[f"{col}_{lag}"] = self.buffer_df[col].shift(lag).iloc[-1]
                else:
                    features[f"{col}_{lag}"] = self.buffer_df[col].iloc[-1] if not self.buffer_df[col].empty else 0
        
        # for col in self.state_variables:
        #     for window_size in self.rolling_window_sizes:
        #         if len(self.buffer_df) >= window_size:
        #             features[f"{col}_rolling_{window_size}_mean"] = self.buffer_df[col].shift(1).rolling(window=window_size).mean().iloc[-1]
        #             features[f"{col}_rolling_{window_size}_std"] = self.buffer_df[col].shift(1).rolling(window=window_size).std().iloc[-1]
        #         else:
        #             mean_value = self.buffer_df[col].mean() if not self.buffer_df[col].empty else 0
        #             std_value = self.buffer_df[col].std() if not self.buffer_df[col].empty else 0
        #             features[f"{col}_rolling_{window_size}_mean"] = mean_value
        #             features[f"{col}_rolling_{window_size}_std"] = std_value

        return features
    
    
    def render(self, mode='console', close=False):
        if mode == 'console':
            print(f"Step: {self.current_step}")
            print(f"State: {self.state}")
            print(f"Action: {self.action}")


    def plot_predictions(self):
        self.actuals_df.to_csv("actual_df.csv")
        self.predictions_df.to_csv("predictions_df.csv")
        for col in self.state_variables+["Gas_volume"]:
            fig, ax = plt.subplots(figsize=(20, 12))
            actual_values = self.actuals_df[col].tolist()
            predicted_values = self.predictions_df[col].tolist()

            ax.plot(actual_values, label='Actual')
            ax.plot(predicted_values, label='Predicted', linestyle='--')
            ax.set_title(f'Actual vs Predicted Values for {col}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Values')
            ax.legend()

            plt.savefig(f'plots/{col}_plot.png')
            plt.close(fig) 