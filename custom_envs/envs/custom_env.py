import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

num_spaces = 4
beta = 0.6
setpoint_temp = 23
range_around = 2
setpoint_temp_range = list(range(setpoint_temp-range_around, setpoint_temp+range_around + 1))

class HVACEnv(gym.Env):
    """
    A simple HVAC control environment for reinforcement learning.
    """
    def __init__(self):
        self.seed()
        super(HVACEnv, self).__init__()
        # Define the observation space
        # Assuming the temperature ranges from 0 to 50 degrees Celsius and humidity from 0% to 100%
        low = np.array([[0]*num_spaces, [0]*num_spaces])  # Temperature lows followed by humidity lows
        high = np.array([[50]*num_spaces, [100]*num_spaces]) # Temperature highs followed by humidity highs

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.reward_range=(-10,10)
        self.state = None
        
    def pass_df(self,df, gas_df):
        self.df = df
        self.gas_df = gas_df
        print(self.df.head())    
    
    def _next_observation(self):
        # self.state = np.array([np.array(self.df.iloc[self.current_step: self.current_step+4, :num_spaces].values, dtype=np.float32), 
        # np.array(self.df.iloc[self.current_step: self.current_step+4, :num_spaces].values, dtype=np.float32)])

        self.state = np.array([np.array(self.df.iloc[self.current_step].values, dtype=np.float32), 
        np.array(self.df.iloc[self.current_step].values, dtype=np.float32)])

        return self.state
    
    def _take_action(self,action):
        self.gas_consumption = self.gas_df.iloc[self.current_step].values
        self.temp = self.state[0, 0, :]
        self.humid = self.state[1, 0, :]

        if action == 0:
            self.temp -= 2
            self.humid *= (self.temp/100)
            
        elif action == 1:
            self.temp += 2 
            self.humid *= (self.temp/100)
            self.gas_consumption += 0.2
            
        self.comfort = [2 - abs(setpoint_temp_range[2] - t) if t in setpoint_temp_range else -(abs(min(setpoint_temp_range, key=lambda x: abs(x - t)) - t)) for t in self.temp]
        self.comfort = sum(self.comfort) / len(self.comfort)
            
    def step(self,action):
        self._take_action(action)
        
        reward = -(beta * self.gas_consumption) + (1 - beta) * self.comfort
        
        self.current_step += 1
        if self.current_step > len(self.df)-4:
            self.current_step = 0
            
        done=reward>2.5
        
        obs = self._next_observation()
        return obs, reward, done, {}
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed] 
    
    def reset(self):
        # self.temp = np.random.uniform(0, 50, num_spaces)
        # self.humid = np.random.uniform(0, 100, num_spaces)
        self.current_step = np.random.randint(0, len(self.df)-4)
        return self._next_observation()
    
    def _render(self, mode='human', close=False):
        pass
