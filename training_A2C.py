import gym
import matplotlib.pyplot as plt
from ray import tune
from ray.rllib.algorithms.a2c import A2CConfig
from ray.tune.registry import register_env
from pymgrid.envs import DiscreteMicrogridEnv
import pandas as pd

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
        normalized_reward = reward / 4E7  # You might want to adjust this per microgrid
        
        return obs, normalized_reward, done, info

def env_creator(env_config):
    return MyEnv(env_config)  # Gib die benutzerdefinierte Umgebung zurück

# Registriere die Umgebung
register_env("my_env", env_creator)

# A2C-Konfiguration
config = A2CConfig().environment("my_env").framework("torch").resources(num_gpus=0)
trainer = config.build()

# Trainiere den A2C-Agenten
analysis = tune.run(
    "A2C",  # Verwende den A2C-Algorithmus
    config=config.to_dict(),  # Konfiguration übergeben
    stop={"timesteps_total": 2_000_000},  # Trainiere für probeweise 1000 Timesteps (2 Millionen Timesteps)
    checkpoint_at_end=True,  # Speichere das finale Modell
    local_dir="./results",  # Verzeichnis für Ergebnisse
    metric="episode_reward_mean",  # Metrik für die Auswahl des besten Modells
    mode="max"  # Maximieren des Rewards
)

# Bestes Checkpoint laden
best_checkpoint = analysis.get_best_checkpoint(trial=analysis.trials[0], metric="episode_reward_mean")
print(f"Best checkpoint: {best_checkpoint}")