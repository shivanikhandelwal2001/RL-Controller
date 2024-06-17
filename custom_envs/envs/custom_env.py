import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

beta = 0.6
setpoint_temp = 20
range_around = 2
setpoint_temp_range = list(range(setpoint_temp-range_around, setpoint_temp+range_around + 1))
ideal_humidity_lower = 40
ideal_humidity_upper = 60

class HVACEnv(gym.Env):
    """
    A simple HVAC control environment for reinforcement learning.
    """
    def __init__(self, df):
        super(HVACEnv, self).__init__()
        # Assuming the temperature ranges from 0 to 50 degrees Celsius and humidity from 0% to 100%
        self.df = df
        
        temp_columns = [col for col in df.columns if col.endswith("Air_temperature")]
        humi_columns = [col for col in df.columns if col.endswith("Relative_humidity")]
        n_temp, n_humi = len(temp_columns), len(humi_columns)
        
        self.state_variables = [col for col in df.columns if (col.endswith("Air_temperature")) or col.endswith("Relative_humidity")]
        self.control_variables = [col for col in df.columns if col not in self.state_variables]
        self.lagged_features = {"lags_1": [col for col in df.columns if col.endswith("_1")],
                                "lags_2": [col for col in df.columns if col.endswith("_2")]}
        
        self.lags = 2
        self.current_step = 0    
        self.refresh_step = 0
        self.max_refresh_step = 49
        self.buffer_df = pd.DataFrame(columns=self.state_variables)
        
        self.actuals_df = pd.DataFrame(columns=self.state_variables+["Gas_volume"])
        self.predictions_df = pd.DataFrame(columns=self.state_variables+["Gas_volume"])
        
        low_bounds = np.array([0] * n_temp + [0] * n_humi)
        high_bounds = np.array([50] * n_temp + [100] * n_humi)
        
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([2]), dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        
        self.state = self.observation_space.sample()
        self.reward = 0
    
        
    def pass_df(self, df):
        self.df = df 
    
        
    def pass_model_details(self, ml_model, scaler_X, scaler_y):
        self.ml_model = ml_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
    
    
    def _take_action(self):
        self.control = self.df[self.control_variables].iloc[[self.current_step]]
        actual_gas_volume = self.control['Gas_volume']
        
        action_gas_volume = self.action[0]
        self.control['Gas_volume'] = action_gas_volume
        self.update_lagged_features()
        
        actual_next_state = self.df[self.state_variables].iloc[[self.current_step]]
        actual_next_state["Gas_volume"] = actual_gas_volume
        
        control_scaled = self.scaler_X.transform(self.control)
        
        next_state_scaled = self.ml_model.predict(control_scaled)
        self.predicted_next_state = pd.DataFrame(self.scaler_y.inverse_transform(next_state_scaled), columns=self.state_variables)
        self.predicted_next_state["Gas_volume"] = action_gas_volume
        
        self.actuals_df = pd.concat([self.actuals_df, actual_next_state], ignore_index=True)
        self.predictions_df = pd.concat([self.predictions_df, self.predicted_next_state], ignore_index=True)
        
        if self.refresh_step < self.max_refresh_step:
            self.refresh_step += 1
            self.update_buffer(new_row=self.predicted_next_state)   
        else:
            self.refresh_step = 0
            self.update_buffer(reset=True)
    
    
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
        # self._calculate_reward()
            
        done = bool(self.current_step >= len(self.df)-1)
        self.state = self.predicted_next_state
        self.current_step += 1
        
        return self.state, self.reward, False, done, {}
    
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.refresh_step = 0
        self.state = self.df[self.state_variables].iloc[self.current_step].values
        self.buffer_df = pd.DataFrame(columns=self.state_variables)
        
        return self.state
    
    
    def update_buffer(self, new_row=None, reset=False):
        # Update the buffer with new observations or predictions
        if reset:
            self.buffer_df = pd.DataFrame(columns=self.state_variables)            
        else:
            self.buffer_df = pd.concat([self.buffer_df, new_row[self.state_variables]], ignore_index=True)        
    

    def update_lagged_features(self):
        # Generate lag and rolling features based on the current buffer.
        for l in range(1, self.lags + 1):
            if self.refresh_step - l >= 0:
                self.control.loc[self.control.index[0], self.lagged_features[f"lags_{l}"]] = self.buffer_df.iloc[(self.refresh_step - l), :].values
    
    
    def render(self, mode='console', close=False):
        if mode == 'console':
            print(f"Step: {self.current_step}")
            # print(f"State: {self.state}")
            print(f"Action: {self.action}")
            print(f"Reward: {self.reward}")


    def plot_predictions(self):
        self.actuals_df.to_csv("actual_df.csv")
        self.predictions_df.to_csv("predictions_df.csv")
        
        for col in self.state_variables+["Gas_volume"]:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(self.actuals_df.index, self.actuals_df[col], label="Actual", color="tab:blue")
            ax.plot(
                self.actuals_df.index,
                self.predictions_df[col],
                label="Predicted",
                linestyle="--",
                color="tab:orange",
            )
            ax.set_title(f'Actual vs Predicted Values for {col}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Values')
            ax.legend()

            plt.savefig(f'plots/{col}_plot.png')
            plt.close(fig) 