import gym
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env
import numpy as np
from pymgrid.envs import DiscreteMicrogridEnv

from gym.spaces import Dict

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN


tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

# Import PyTorch
torch, nn = try_import_torch()

class MultiMicrogridEnv(gym.Env):
    def __init__(self, env_config):
        self.microgrid_numbers = env_config.get("microgrid_numbers", [1, 2])  # List of available microgrids
        
        # Create environments for each microgrid
        self.envs = {
            num: DiscreteMicrogridEnv.from_scenario(microgrid_number=num)
            for num in self.microgrid_numbers
        }
        
        # Get the maximum action space size across all microgrids
        max_action_size = max(env.action_space.n for env in self.envs.values())
        
        # Create a unified action space that can accommodate all microgrids
        self.action_space = gym.spaces.Discrete(max_action_size)
        
        # Get the observation space from one of the microgrids
        sample_obs = self.envs[self.microgrid_numbers[0]].observation_space
        obs_dim = sample_obs.shape[0]  # Shape of the observation (e.g., 12)
        
        # Define the observation space as a Dict with "obs" and "action_mask"
        self.observation_space = gym.spaces.Dict({
            "obs": sample_obs,  # Original observation space
            "action_mask": gym.spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)  # Action mask
        })
 
        self.episode_length = 24
        self.current_microgrid = None
        self.current_env = None
        
    def get_action_mask(self):
        # Return a mask indicating valid actions for the current microgrid
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        valid_actions = self.current_env.get_action_mask()  # Assuming this method exists
        mask[:len(valid_actions)] = valid_actions
        return mask
        
    def reset(self):
        # Randomly select a microgrid for this episode
        self.current_microgrid = np.random.choice(self.microgrid_numbers)
        self.current_env = self.envs[self.current_microgrid]
        
        # Get observation and action mask
        obs = self.current_env.reset()
        action_mask = self.get_action_mask()
        
        return {
            "obs": obs,
            "action_mask": action_mask
        }
        
    def step(self, action):
        # Check if action is valid using the mask
        action_mask = self.get_action_mask()
        #print(action, action_mask)
        if not action_mask[action]:
            return self.reset(), -1.0, True, {"invalid_action": True}
            
        obs, reward, done, info = self.current_env.step(action)
        
        # Normalize reward based on current microgrid
        normalized_reward = reward / 4E7  # You might want to adjust this per microgrid
        
        info['microgrid_number'] = self.current_microgrid
        return {
            "obs": obs,
            "action_mask": self.get_action_mask()
        }, normalized_reward, done, info

class ActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "obs" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["obs"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Ensure the action mask is valid (no NaNs or extreme values)
        if torch.isnan(action_mask).any() or torch.isinf(action_mask).any():
            raise ValueError("Action mask contains NaN or Inf values.")

        # Add a small epsilon to avoid log(0)
        action_mask = torch.clamp(action_mask, min=1e-10, max=1.0)

        #print("Action Mask:", action_mask)
        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["obs"]})
        #print("Logits:", logits)
        if torch.isnan(logits).any():
            raise ValueError("Logits contain NaN values.")

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        #print("Masked Logits:", masked_logits)
        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()

multi = MultiMicrogridEnv({})
obs = multi.reset()

obs, reward, done, info = multi.step(1)

# Register the custom model
ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)

# Register the environment
def env_creator(env_config):
    return MultiMicrogridEnv(env_config)

register_env("multi_microgrid", env_creator)

# Configuration for PPO
config = {
    "disable_env_checking": True,
    "env": "multi_microgrid",
    "env_config": {
        "microgrid_numbers": [1, 2]  # Add all microgrid numbers you want to support
    },
    "framework": "torch",
    "lr": 1e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_param": 0.2,
    "entropy_coeff": 0.01,
    "train_batch_size": 2400,
    "sgd_minibatch_size": 256,
    "num_sgd_iter": 10,
    "num_workers": 4,
    "num_envs_per_worker": 1,
    "rollout_fragment_length": 24,
    "observation_filter": "MeanStdFilter",
    "vf_loss_coeff": 0.5,
    "grad_clip": 0.5,
    "seed": 42,
    "model": {
        "custom_model": "action_mask_model",  # Use the custom action mask model
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "use_lstm": False
    },
    "exploration_config": {
        "type": "SoftQ",
        "temperature": 1.0,
    },
}

# Train the PPO agent
analysis = tune.run(
    PPOTrainer,
    config=config,
    stop={"timesteps_total": 1_000},  # Train for 1 million timesteps
    checkpoint_at_end=True,
    local_dir="./results_multi",
    metric="episode_reward_mean",
    mode="max"
)

# Load the best checkpoint
best_checkpoint = analysis.get_best_checkpoint(
    trial=analysis.trials[0], 
    metric="episode_reward_mean"
)
print(f"Best checkpoint: {best_checkpoint}")