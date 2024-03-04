from hvac_data import get_hvac_data
import sys
import gym
from custom_envs import *
import time
from Q_Learning import Q_Learning
import numpy as np

data, gas_data = get_hvac_data()

env = gym.make('CustomEnv-v0')
env.pass_df(data, gas_data)

print(env.observation_space)
print(env.action_space)
print(env.spec)
print(env.spec.max_episode_steps)
print(env.spec.reward_threshold)


# here define the parameters for state discretization
upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
print(upperBounds, lowerBounds)
numberOfBinsTemp = 50
numberOfBinsHumid = 50
numberOfBins=[numberOfBinsTemp, numberOfBinsHumid]

# define the parameters
alpha = 0.1
gamma = 1
epsilon = 0.2
timeSteps = 99
numSpaces = 4
episodeNumber = 25000


# Simulation Training with Q-Matrix
env = gym.make('CustomEnv-v0')
env.pass_df(data, gas_data)

Q1 = Q_Learning(env, alpha, gamma, epsilon, episodeNumber, timeSteps, numSpaces, numberOfBins, lowerBounds, upperBounds)
Q1.simulateEpisodes()

# Simulation Evaluation with Learned Policy
env1 = gym.make('CustomEnv-v0')
env1.pass_df(data, gas_data)

episode_rewards = []

for episodeIndex in range(10):
    total_rewards_ep = 0
    state = env1.reset()
    print("Episode no. ", episodeIndex)
    
    for timeIndex in range(timeSteps):
        # select greedy actions
        greedy_action = Q1.greedy_policy(state)   
        state, reward, terminated, info = env1.step(greedy_action)

        total_rewards_ep += reward
        
        if terminated:
            break    

    print(f"Episode {episodeIndex} Reward {total_rewards_ep}")
    episode_rewards.append(total_rewards_ep)
    
print(f"Final Episode Rewards: {episode_rewards}")


# Simulation Evaluation with Random Actions 
env2 = gym.make('CustomEnv-v0')
env2.pass_df(data, gas_data)

episode_rewards = []

for episodeIndex in range(10):
    total_rewards_ep = 0
    initial_state = env2.reset()
    print("Episode no. ", episodeIndex)
    
    for timeIndex in range(timeSteps):
        random_action = env2.action_space.sample()
        state, reward, terminated, info = env2.step(random_action)
        total_rewards_ep += reward
        
        if terminated:
            break    

    print(f"Episode {episodeIndex} Reward {total_rewards_ep}")
    episode_rewards.append(total_rewards_ep)
    
print(f"Final Episode Rewards: {episode_rewards}")