from pathlib import Path
from ray.rllib.algorithms.algorithm import Algorithm
import gym
import pandas as pd
from ray.rllib.algorithms.dqn import DQNConfig
from pymgrid.envs import DiscreteMicrogridEnv
from ray.tune.registry import register_env


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
        return obs, reward, done, info


def env_creator(env_config):
    return MyEnv(env_config)  # Rückgabe einer Instanz der benutzerdefinierten Umgebung


# Liste zur Speicherung der Ergebnisse
df_list = []

# Umgebung registrieren
register_env("my_env", env_creator)

# Konfiguration des DQN-Algorithmus
config = (
    DQNConfig()
    .environment(env="my_env")
    .framework("torch")  # PyTorch-Backend
    .resources(num_gpus=0)  # Keine GPU verwenden
    .rollouts(num_rollout_workers=1)  # Ein Rollout-Worker
    .training(
        gamma=0.99,  # Diskontierungsfaktor
        lr=1e-3,  # Lernrate
        train_batch_size=32,  # Batch-Größe
        replay_buffer_config={"capacity": 10000},  # Replay-Buffer
    )
)

# Exploration-Einstellungen hinzufügen
config.exploration_config = {
    "type": "EpsilonGreedy",  # Explorationsstrategie
    "initial_epsilon": 1.0,  # Startwert für Epsilon
    "final_epsilon": 0.1,  # Minimaler Epsilon-Wert
    "epsilon_timesteps": 10000,  # Dauer des Decay
}

# Training mit DQN starten
algo = config.build()

# Training über mehrere Iterationen
for i in range(10):  # Anzahl der Iterationen
    result = algo.train()
    print(f"Iteration {i}: episode_reward_mean = {result['episode_reward_mean']}")

# Evaluation
for i in range(1, 2):  # Nur Microgrid 1 (anpassbar)
    env = DiscreteMicrogridEnv.from_scenario(microgrid_number=i)

    # Agent von einem gespeicherten Checkpoint laden, falls nötig
    # algo = Algorithm.from_checkpoint("<Pfad_zum_Checkpoint>")

    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = algo.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    print(f"Total reward for microgrid {i}: {episode_reward}")

    df = env.log
    print(df["balance"][0]["reward"].std())
    try:
        if df['grid'][0]["grid_status_current"].eq(1).all():
            df["weak_grid"] = 0
        else:
            df["weak_grid"] = 1
    except:
        df["weak_grid"] = pd.NA
    df_list.append(df)

# Ergebnisse analysieren
df = pd.concat(df_list)
all_df = df["balance"][0]["reward"]
print("Mean Cost:" + str(all_df.mean()))
print("Total Cost:" + str(all_df.sum()))
