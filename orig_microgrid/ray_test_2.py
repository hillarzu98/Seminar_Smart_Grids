import gym
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env
import numpy as np
from pymgrid.envs import DiscreteMicrogridEnv

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
        normalized_reward = reward / 4E7  # Normalize reward
        return obs, normalized_reward, done, info

def env_creator(env_config):
    return MyEnv(env_config)  # return an env instance

register_env("my_env", env_creator)
# Register the custom model with RLlib
#ModelCatalog.register_custom_model("microgrid", MyEnv)

# Define the PPO configuration
config = {
    "env": "my_env",  # Replace with your custom environment
    "framework": "torch",  # Use PyTorch
    "lr": 3e-4,  # Learning rate
    "gamma": 0.99,  # Discount factor
    "lambda": 0.95,  # GAE parameter
    "clip_param": 0.2,  # Clipping parameter
    "entropy_coeff": 0.01,  # Entropy coefficient
    "train_batch_size": 2400,  # Total batch size per training iteration
    "sgd_minibatch_size": 256,  # Mini-batch size
    "num_sgd_iter": 10,  # Number of epochs per update
    "num_workers": 4,  # Number of parallel workers
    "num_envs_per_worker": 1,  # Environments per worker
    "rollout_fragment_length": 24,  # Steps per worker per rollout
    "observation_filter": "MeanStdFilter",  # Normalize observations
    "vf_loss_coeff": 0.5,  # Value function loss coefficient
    "grad_clip": 0.5,  # Gradient clipping
    "seed": 42,  # Random seed for reproducibility
}

# Train the PPO agent
analysis = tune.run(
    PPOTrainer,
    config=config,
    stop={"timesteps_total": 2_000_000},  # Train for 1 million timesteps
    checkpoint_at_end=True,  # Save the final model checkpoint
    local_dir="./results",  # Directory to save results
    metric="episode_reward_mean",
    mode="max"  # Specify we want to maximize the reward
)

# Load the best checkpoint
best_checkpoint = analysis.get_best_checkpoint(trial=analysis.trials[0], metric="episode_reward_mean")
print(f"Best checkpoint: {best_checkpoint}")