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
        return obs, reward, done, info

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
        train_batch_size=2400,  # Ähnlich wie bei PPO
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
    "DQN",  # Name des Algorithmus als String
    config=config.to_dict(),  # `config.to_dict()` verwenden
    stop={"timesteps_total": 2_000_000},  # Ändere dies je nach gewünschter Stop-Kondition
    checkpoint_at_end=True,
    metric="episode_reward_mean",
    mode="max",
    local_dir="./results"
)

# Bestes Checkpoint auswählen (Fehler abfangen, falls keiner vorhanden ist)
best_checkpoint = None
try:
    best_checkpoint = analysis.get_best_checkpoint(trial=analysis.trials[0], metric="episode_reward_mean")
    print(f"Best checkpoint: {best_checkpoint}")
except Exception as e:
    print(f"Fehler beim Abrufen des besten Checkpoints: {e}")

# Evaluation mit dem besten Checkpoint (nur wenn ein Checkpoint vorhanden ist)
if best_checkpoint:
    df_list = []
    algo = Algorithm.from_checkpoint(best_checkpoint)

    for i in range(1, 2, 1):  # Nur Microgrid 1 (anpassbar)
        env = DiscreteMicrogridEnv.from_scenario(microgrid_number=i)
        obs = env.reset()
        done = False
        episode_reward = 0

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

    # Excel-Export der Ergebnisse
    output_path = Path("./results/evaluation_results.xlsx")
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Speichern der Daten im Excel
        df.to_excel(writer, sheet_name='Training Results', index=False)
        
        # Zusammenfassung der Kosten hinzufügen
        summary_df = pd.DataFrame({
            'Mean Cost': [all_df.mean()],
            'Total Cost': [all_df.sum()]
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Diagramm zum Excel hinzufügen
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['timesteps_total'], df['episode_reward_mean'], label="Episode Reward Mean")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Reward Mean")
        ax.set_title("Training Progress")
        ax.grid(True)
        ax.legend()

        # Speichern des Diagramms als Bild
        img_path = Path("./results/training_progress.png")
        fig.savefig(img_path)

        # Füge das Diagramm in das Excel-Blatt ein
        worksheet = writer.sheets['Training Results']
        worksheet.insert_image('F2', str(img_path), {'x_scale': 0.5, 'y_scale': 0.5})

    print(f"Results have been saved to: {output_path}")

# Nach dem Training Plotten der Reward-Entwicklung und mehrere Plots nebeneinander
plt.figure(figsize=(12, 6))

# Erstelle Subplots nebeneinander
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot für den einzelnen Run
axes[0].plot(df["timesteps_total"], df["episode_reward_mean"], label="Mean Reward", color="blue")
axes[0].set_xlabel("Timesteps")
axes[0].set_ylabel("Episode Reward Mean")
axes[0].set_title("Training Progress")
axes[0].grid(True)
axes[0].legend()

# Mehrere Runs vergleichen
for trial in analysis.trials:
    trial_df = trial.last_result
    axes[1].plot(trial_df["timesteps_total"], trial_df["episode_reward_mean"], label=trial.trial_id)

# Nach dem Training prüfen und sicherstellen, dass 'timesteps_total' gültig ist
if 'timesteps_total' in analysis.results_df.columns:
    # Entferne Zeilen mit NaN-Werten in 'timesteps_total'
    clean_df = analysis.results_df.dropna(subset=['timesteps_total'])
    if not clean_df.empty:
        print(clean_df['timesteps_total'])
else:
    print("timesteps_total ist noch nicht verfügbar.")

axes[1].set_xlabel("Timesteps")
axes[1].set_ylabel("Episode Reward Mean")
axes[1].set_title("Vergleich verschiedener Runs")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
