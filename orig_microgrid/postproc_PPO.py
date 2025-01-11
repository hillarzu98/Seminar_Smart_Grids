from pathlib import Path

from ray.rllib.algorithms.algorithm import Algorithm
import gym
import pandas as pd
from ray.rllib.algorithms.ppo import PPOConfig
from pymgrid.envs import DiscreteMicrogridEnv
from ray.tune.registry import register_env

class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.env = DiscreteMicrogridEnv.from_scenario(microgrid_number=1)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.episode_length = 24  # 24-hour episode
    def reset(self):
        return self.env.reset()
    def step(self, action):
        #print("Action:", action)
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

def env_creator(env_config):
    return MyEnv(env_config)  # return an env instance

df_list = []
# Currently only for microgrid 1 if all 25 need to be run then range(0,25,1)
# this is however not possible with the current setup
for i in range(1,2,1):
    env = DiscreteMicrogridEnv.from_scenario(microgrid_number=i)
    
    register_env("my_env", env_creator)

    algo = Algorithm.from_checkpoint("C:/Users/arin1/Seminar_Smart_Grids/results/PPO_2025-01-05_14-05-59/PPO_my_env_cf0c0_00000_0_2025-01-05_14-06-00/checkpoint_001000/")

    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = algo.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
    
    df = env.log
    print(df["balance"][0]["reward"].std())
    #print(df["balance"][0]["reward"].sum())
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
    df_list.append(df)

df = pd.concat(df_list)

#All Grids x25
all_df = df["balance"][0]["reward"]
print("Mean Cost:" + str(all_df.mean()))
print("Total Cost:" + str(all_df.sum()))

#----- Only needed when it is possible to run multiple grids with a agent
# Grid Only x7
grid_only = df[pd.notna(df['grid'][0]["reward"]) & pd.isna(df['genset'][0]["reward"])]
print("Grid Only Mean:" + str(grid_only["balance"][0]["reward"].mean()))
print("Grid Only Std:" + str(grid_only["balance"][0]["reward"].std()))
print("Grid Only Sum:" + str(grid_only["balance"][0]["reward"].sum()/7))
len(df[pd.notna(df['grid'][0]["reward"]) & pd.isna(df['genset'][0]["reward"])])/8758
# Genset Only x10
genset_only = df[pd.isna(df['grid'][0]["reward"]) & pd.notna(df['genset'][0]["reward"])]
print("Genset Only Mean:" + str(genset_only["balance"][0]["reward"].mean()))
print("Genset Only Std:" + str(genset_only["balance"][0]["reward"].std()))
print("Genset Only Sum:" + str(genset_only["balance"][0]["reward"].sum()/10))
len(df[pd.isna(df['grid'][0]["reward"]) & pd.notna(df['genset'][0]["reward"])])/8758
# Grid + Genset x4
grid_genset = df[pd.notna(df['grid'][0]["reward"]) & pd.notna(df['genset'][0]["reward"]) & df["weak_grid"].eq(0)]
print("Grid + Genset Mean:" + str(grid_genset["balance"][0]["reward"].mean()))
print("Grid + Genset Std:" + str(grid_genset["balance"][0]["reward"].std()))
print("Grid + Genset Sum:" + str(grid_genset["balance"][0]["reward"].sum()/4))
len(df[pd.notna(df['grid'][0]["reward"]) & pd.notna(df['genset'][0]["reward"]) & df["weak_grid"].eq(0)] )/8758
# Genset + Weak Grid x4
grid_genset_weak = df[pd.notna(df['grid'][0]["reward"]) & pd.notna(df['genset'][0]["reward"]) & df["weak_grid"].eq(1)]
print("Genset + Weak Grid Mean:" + str(grid_genset_weak["balance"][0]["reward"].mean()))
print("Genset + Weak Grid Std:" + str(grid_genset_weak["balance"][0]["reward"].std()))
print("Genset + Weak Grid Sum:" + str(grid_genset_weak["balance"][0]["reward"].sum()/4))
len(df[pd.notna(df['grid'][0]["reward"]) & pd.notna(df['genset'][0]["reward"]) & df["weak_grid"].eq(1)] )/8758
