import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext
import gymnasium as gym
import numpy as np

class CustomCartPoleEnv(gym.Env):
    """
    A custom wrapper around CartPole environment to demonstrate RLLib usage
    """
    def __init__(self, config: EnvContext):
        # Use the standard CartPole environment
        self.env = gym.make('CartPole-v1')
        
        # Define action and observation spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Optional: Use the worker index and vector index for any environment-specific logic
        self.worker_index = config.worker_index
        self.vector_index = config.vector_index
        
        # Reset state
        self.state = None

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and return initial observation
        """
        # Reset the environment
        self.state, info = self.env.reset(seed=seed)
        return self.state, info

    def step(self, action):
        """
        Take a step in the environment
        """
        # Apply the action and get next state, reward, done flag, and additional info
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # Update current state
        self.state = next_state
        
        # Combine terminated and truncated for RLLib compatibility
        done = terminated or truncated
        
        return next_state, reward, done, False, info

def train_cartpole():
    """
    Train a PPO agent on the CartPole environment
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    try:
        # Configure the training
        config = (
            PPOConfig()
            .environment(CustomCartPoleEnv)
            .rollouts(num_rollout_workers=2)
            .training(
                train_batch_size=4000,
                lr=0.0001,
                gamma=0.99,
                lambda_=0.95,
                use_gae=True,
                clip_param=0.2,
                num_sgd_iter=10,
                vf_loss_coeff=0.5,
                entropy_coeff=0.001
            )
            .debugging(log_level="INFO")
            .resources(num_gpus=0)
        )

        # Run the training
        tune.run(
            "PPO",
            config=config.to_dict(),
            stop={
                "episode_reward_mean": 195,  # Stop when average reward reaches 195
                "training_iteration": 100    # Or stop after 100 iterations
            },
            verbose=1
        )

    finally:
        # Shutdown Ray
        ray.shutdown()

# Run the training
if __name__ == "__main__":
    train_cartpole()