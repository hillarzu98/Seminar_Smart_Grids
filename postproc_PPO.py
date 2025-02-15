from pathlib import Path

from ray.rllib.algorithms.algorithm import Algorithm
import gym
import pandas as pd
from ray.rllib.algorithms.ppo import PPOConfig
from pymgrid.envs import DiscreteMicrogridEnv
from ray.tune.registry import register_env

class MyEnv(gym.Env):
    def __init__(self, env_config):
        # Get predefined Microgrid from python-microgrid library
        self.env = DiscreteMicrogridEnv.from_scenario(microgrid_number=1)
        # Get action and observation space
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        # Returns the initial observation
        return self.env.reset()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

def env_creator(env_config):
    return MyEnv(env_config)  # return an env instance

register_env("my_env", env_creator)

algo = Algorithm.from_checkpoint("results/PPO_2025-01-05_16-11-28/PPO_my_env_565b8_00000_0_2025-01-05_16-11-28/checkpoint_000417")

df_list = []
# Currently only for microgrid 1 if all 25 need to be run then range(0,25,1)
# this is however not possible with the current setup
for i in range(0,25,1):
    env = DiscreteMicrogridEnv.from_scenario(microgrid_number=i)
    
    
    episode_reward = 0
    done = False
    obs = env.reset()
    try:
        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
        
        df = env.log
        print(df["balance"][0]["reward"].mean())
        print(df["balance"][0]["reward"].sum())
        print(df["balance"][0]["reward"].std())
        try:
            # if file.name in "mpc_24.csv":
            #     print(file.name)
            if df['grid'][0]["grid_status_current"].eq(1).all():
                #print("NO Weak Grid")
                df["weak_grid"] = 0
            else:
                #print("Weak Grid")
                df["weak_grid"] = 1
        except:
            #print("NO Grid")
            df["weak_grid"] = pd.NA
        df["grid_nr"] = i
        df_list.append(df)
    except:
        df_list.append(pd.DataFrame())
        print("PASSED_________________________________________________________")
        pass

df = pd.concat(df_list)
df.to_excel("PPO_Results.xlsx")

df.groupby('grid_nr').agg({('balance', 0, 'reward'): 'mean'})
df.groupby('grid_nr').agg({('balance', 0, 'reward'): 'sum'})
df.groupby('grid_nr').agg({('balance', 0, 'reward'): 'std'})

#All Grids x25
all_df = df["balance"][0]["reward"]
print("Mean Cost:" + str(all_df.mean()))
print("Total Cost:" + str(all_df.sum()))
print("Std:" + str(all_df.std()))

