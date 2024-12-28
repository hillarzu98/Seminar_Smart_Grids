import pandas as pd
import numpy as np
from pymgrid.envs import DiscreteMicrogridEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
import matplotlib.pyplot as plt
import os



#env = gym.make("CartPole-v1", render_mode="rgb_array")
env = DiscreteMicrogridEnv.from_scenario(microgrid_number=11)
#env = DiscreteMicrogridEnv.from_microgrid(microgrid)
#check_env(env)
#env = Monitor(env, log_dir)


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_testlog/")
model.learn(total_timesteps=100, tb_log_name="first_run")
vec_env = model.get_env()
obs = vec_env.reset()

actions_lst = []
reward_lst = []
load_lst = []
pv_lst = []
grid_lst = []
battery_lst = []
battery_abs_lst = []
genset_lst = []

num = 300
for i in range(num):
    action, _state = model.predict(obs, deterministic=True)
    print(env.convert_action(action[0]))
    #obs, reward, done, info = vec_env.step(action)
    obs, rewards, terminations, truncations, infos = convert_to_terminated_truncated_step_api(vec_env.step(action), is_vector_env=True)
    actions_lst.append(action)
    reward_lst.append(rewards)
    load_lst.append(infos[0]['load'][0]['absorbed_energy'])
    pv_lst.append(infos[0]['pv'][0]['provided_energy'])
    try:
        battery_lst.append(infos[0]['battery'][0]['provided_energy'])
    except:
        battery_lst.append(0)
    try:
        battery_abs_lst.append(infos[0]['battery'][0]['absorbed_energy'])
    except:
        battery_abs_lst.append(0)
    try:
        grid_lst.append(infos[0]['grid'][0]['provided_energy'])
    except:
        grid_lst.append(0)
    try:
        genset_lst.append(infos[0]['genset'][0]['provided_energy'])
    except:
        genset_lst.append(0)
    
    print(rewards,infos)

plt.subplot(411)
plt.plot(np.linspace(0,100,num),reward_lst)
plt.subplot(412)
plt.plot(np.linspace(0,100,num),actions_lst)
plt.subplot(413)
plt.plot(np.linspace(0,100,num),load_lst)
plt.plot(np.linspace(0,100,num),pv_lst)
plt.plot(np.linspace(0,100,num),battery_lst)
plt.plot(np.linspace(0,100,num),grid_lst)
plt.plot(np.linspace(0,100,num),battery_abs_lst)
plt.plot(np.linspace(0,100,num),genset_lst)
plt.legend(["load","pv","battery","grid","battery_absorbed","genset"])
plt.subplot(414)
plt.plot(np.linspace(0,100,num),battery_lst)
plt.plot(np.linspace(0,100,num),battery_abs_lst)
plt.legend(["battery","battery_absorbed"])
plt.show()
    #vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()