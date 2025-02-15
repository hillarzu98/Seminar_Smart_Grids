from pathlib import Path
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from pymgrid.envs import DiscreteMicrogridEnv
import gym
import pandas as pd
import matplotlib.pyplot as plt

# Definiere die benutzerdefinierte Umgebung
class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.env = DiscreteMicrogridEnv.from_scenario(microgrid_number=1)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.episode_length = 24  # 24-hour episode

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Normalize reward based on current microgrid
        normalized_reward = reward / 6E7  # You might want to adjust this per microgrid
        
        return obs, normalized_reward, done, info

def env_creator(env_config):
    return MyEnv(env_config)

# Umgebung registrieren
register_env("my_env", env_creator)

# Umgebung zum Extrahieren der Spaces initialisieren
env = MyEnv({})

# Konfiguration für DQN mit DQNConfig()
config = (
    DQNConfig()
    .environment(
        env="my_env", 
        observation_space=env.observation_space, 
        action_space=env.action_space
    )
    .framework("torch")
    .resources(num_gpus=0)
    .rollouts(num_rollout_workers=4, num_envs_per_worker=1, rollout_fragment_length=24)
    .training(
        gamma=0.99,
        lr=1e-3,
        train_batch_size=2400,
        replay_buffer_config={"capacity": 100000},  # Größerer Buffer für stabileres Training
    )
    .exploration(
        exploration_config={
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.1,
            "epsilon_timesteps": 50000,  # Längerer Epsilon-Decay
        }
    )
)

analysis = tune.run(
    "DQN", 
    config=config.to_dict(),  
    stop={"timesteps_total": 2_000_000},  # Ändere dies je nach gewünschter Stop-Kondition
    checkpoint_at_end=True,
    metric="episode_reward_mean",
    mode="max",
    local_dir="./results"
)

# Besten Checkpoint auswählen (Fehler abfangen, falls keiner vorhanden ist)
best_checkpoint = None
try:
    best_checkpoint = analysis.get_best_checkpoint(trial=analysis.trials[0], metric="episode_reward_mean")
    print(f"Best checkpoint: {best_checkpoint}")
except Exception as e:
    print(f"Fehler beim Abrufen des besten Checkpoints: {e}")