import pandas as pd
import numpy as np
from pymgrid.envs import DiscreteMicrogridEnv
from stable_baselines3 import A2C
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
import matplotlib.pyplot as plt

# Try to create own Microgrid
from pymgrid import Microgrid
from pymgrid.modules import (
    BatteryModule,
    LoadModule,
    RenewableModule,
    GridModule)
small_battery = BatteryModule(min_capacity=10,
                              max_capacity=100,
                              max_charge=50,
                              max_discharge=50,
                              efficiency=0.9,
                              init_soc=0.5)

large_battery = BatteryModule(min_capacity=10,
                              max_capacity=1000,
                              max_charge=10,
                              max_discharge=10,
                              efficiency=0.7,
                              init_soc=0.5)
load_ts = 100+100*np.random.rand(24*90) # random load data in the range [100, 200].
pv_ts = 200*np.random.rand(24*90) # random pv data in the range [0, 200].

load = LoadModule(time_series=load_ts)

pv = RenewableModule(time_series=pv_ts)

grid_ts = [0.2, 0.1, 0.5] * np.ones((24*90, 3))

grid = GridModule(max_import=100,
                  max_export=100,
                  time_series=grid_ts)

modules = [
    small_battery,
    large_battery,
    ('pv', pv),
    load,
    grid]

microgrid = Microgrid(modules)

#env = gym.make("CartPole-v1", render_mode="rgb_array")
env = DiscreteMicrogridEnv.from_scenario(microgrid_number=4)
#env = DiscreteMicrogridEnv.from_microgrid(microgrid)
#check_env(env)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1)
vec_env = model.get_env()
obs = vec_env.reset()

actions_lst = []
reward_lst = []
load_lst = []
pv_lst = []
grid_lst = []
battery_lst = []
battery_abs_lst = []
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    #print(env.convert_action(0))
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
    
    print(rewards,infos)

plt.subplot(411)
plt.plot(np.linspace(0,100,1000),reward_lst)
plt.subplot(412)
plt.plot(np.linspace(0,100,1000),actions_lst)
plt.subplot(413)
plt.plot(np.linspace(0,100,1000),load_lst)
plt.plot(np.linspace(0,100,1000),pv_lst)
plt.plot(np.linspace(0,100,1000),battery_lst)
plt.plot(np.linspace(0,100,1000),grid_lst)
plt.plot(np.linspace(0,100,1000),battery_abs_lst)
plt.legend(["load","pv","battery","grid","battery_absorbed"])
plt.subplot(414)
plt.plot(np.linspace(0,100,1000),battery_lst)
plt.plot(np.linspace(0,100,1000),battery_abs_lst)
plt.legend(["battery","battery_absorbed"])
plt.show()
    #vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()