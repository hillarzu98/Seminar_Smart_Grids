import pandas as pd
from pymgrid.envs import DiscreteMicrogridEnv
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext
import gymnasium as gym
from gymnasium.wrappers import EnvCompatibility
import numpy as np
from ray.tune import register_env
import logging
import os
from datetime import datetime

# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"env_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CustomEnvironment(gym.Env):
    """
    A custom OpenAI Gym environment for reinforcement learning.
    This is a simple example of a custom environment.
    """
    def __init__(self, env, config=None):
        super().__init__()
        
        # Wrap the old Gym environment with Gymnasium's compatibility wrapper
        self.env = EnvCompatibility(env)
        self.env2 = env
        # Now we can directly use the wrapped environment's spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state = None
        self.steps = 0
        self.episode_count = 0
        
        logger.info(f"Environment initialized with:")
        logger.info(f"Observation Space: {self.observation_space}")
        logger.info(f"Action Space: {self.action_space}")
        
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to initial state.
        Returns the initial observation and info dict.
        """
        self.episode_count += 1
        observation, info = self.env.reset(seed=seed, options=options)
        self.state = observation
        self.steps = 0
        
        logger.info(f"Episode {self.episode_count} - Reset:")
        logger.info(f"Initial Observation: {observation}")
        logger.info(f"Reset Info: {info}")
        
        return observation, info
        
    def step(self, action):
        """
        Execute one time step within the environment.
        
        Parameters:
        -----------
        action : int
            Action to be taken
        
        Returns:
        --------
        observation : numpy array
            Agent's observation of the environment
        reward : float
            Amount of reward returned after previous action
        terminated : bool
            Whether the episode has ended naturally
        truncated : bool
            Whether the episode was artificially terminated
        info : dict
            Additional diagnostic information
        """
        self.steps += 1
        logger.info(f"Episode {self.episode_count} - Step {self.steps}:")
        logger.info(f"Action taken: {action}")
        
        # Get the unwrapped environment to access convert_action
        try:
            if hasattr(self.env2, 'convert_action'):
                converted_action = self.env2.convert_action(action)
                logger.info(f"Converted action: {converted_action}")
            else:
                logger.warning("Environment does not have convert_action method")
                converted_action = action
        except Exception as e:
            logger.warning(f"Action conversion failed: {str(e)}")
            converted_action = action
        
        # Execute step
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        logger.info(f"Step Result:")
        logger.info(f"Observation: {observation}")
        logger.info(f"Reward: {reward}")
        logger.info(f"Terminated: {terminated}")
        logger.info(f"Truncated: {truncated}")
        logger.info(f"Info: {info}")
        
        if terminated or truncated:
            logger.info(f"Episode {self.episode_count} finished after {self.steps} steps with total reward: {reward}")
        
        return observation, reward, terminated, truncated, info

env = DiscreteMicrogridEnv.from_scenario(microgrid_number=0)

#env = env.to_normalized

# # Registriere die Microgrid-Umgebung bei RLlib
custom_env = CustomEnvironment(env)

print("Environment created:", custom_env)

# # Testing the environment
state = custom_env.reset()
print("Initial state:", state)
for _ in range(5):  # Run a few steps
    action = custom_env.action_space.sample()  # Random action
    state, reward, terminated, truncated, info = custom_env.step(action)
    print(f"State: {state}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    if terminated or truncated:
        break

# Initialisiere Ray
ray.init(ignore_reinit_error=True)

def env_creator(env_config):
    env = DiscreteMicrogridEnv.from_scenario(microgrid_number=0)
    return CustomEnvironment(env)

register_env("microgrid", env_creator)

# Configure the training
config = (
    PPOConfig()
    .environment("microgrid")
    .rollouts(num_rollout_workers=2)
    .training(
        train_batch_size=1000,
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
        "training_iteration": 20    # Or stop after 100 iterations
    },
    verbose=1
)