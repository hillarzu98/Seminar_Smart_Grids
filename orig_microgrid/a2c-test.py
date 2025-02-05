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
        return obs, reward, done, info

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

# Evaluation mit dem besten Checkpoint
df_list = []
algo = trainer  # Verwende den Trainer, der im Training gebaut wurde

# Lade das beste Modell
algo.restore(best_checkpoint)

# Evaluation durchführen
for i in range(1, 2, 1):  # Nur Microgrid 1 (anpassbar)
    env = DiscreteMicrogridEnv.from_scenario(microgrid_number=i)
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = algo.compute_single_action(obs)  # Berechne die beste Aktion
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

# Plotten der Resultate (optional, je nach Wunsch)
plt.figure(figsize=(12, 6))

# Nach dem Training prüfen und sicherstellen, dass 'timesteps_total' gültig ist
if 'timesteps_total' in analysis.results_df.columns:
    # Entferne Zeilen mit NaN-Werten in 'timesteps_total'
    clean_df = analysis.results_df.dropna(subset=['timesteps_total'])
    if not clean_df.empty:
        print(clean_df['timesteps_total'])
else:
    print("timesteps_total ist noch nicht verfügbar.")

# Achte darauf, dass 'timesteps_total' vorhanden ist, bevor du es plottest
if 'timesteps_total' in analysis.results_df.columns:
    plt.plot(analysis.results_df['timesteps_total'], analysis.results_df['episode_reward_mean'], label="Mean Reward", color="blue")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward Mean")
    plt.title("Training Progress with A2C")
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("timesteps_total ist nicht in den Ergebnissen vorhanden.")
