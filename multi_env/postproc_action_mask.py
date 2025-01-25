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
from ray.rllib.algorithms.algorithm import Algorithm

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

multi = MultiMicrogridEnv({
        "microgrid_numbers": [5]  # Add all microgrid numbers you want to support
    })
obs = multi.reset()

#obs, reward, done, info = multi.step(1)

# Register the custom model
ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)

# Register the environment
def env_creator(env_config):
    return MultiMicrogridEnv(env_config)

register_env("multi_microgrid", env_creator)

algo = Algorithm.from_checkpoint("C:/Users/arin1/Seminar_Smart_Grids/results_multi/PPO_2025-01-25_12-38-27/PPO_multi_microgrid_e49b1_00000_0_2025-01-25_12-38-27/checkpoint_000001")

#env = DiscreteMicrogridEnv.from_scenario(microgrid_number=1)

episode_reward = 0
done = False
obs = multi.reset()
while not done:
    action = algo.compute_single_action(obs)
    obs, reward, done, info = multi.step(action)

    # df = env.log
    # print(df["balance"][0]["reward"].mean())
    # print(df["balance"][0]["reward"].sum())
    # print(df["balance"][0]["reward"].std())
    # try:
    #     # if file.name in "mpc_24.csv":
    #     #     print(file.name)
    #     if df['grid'][0]["grid_status_current"].eq(1).all():
    #         #print("NO Weak Grid")
    #         df["weak_grid"] = 0
    #     else:
    #         #print("Weak Grid")
    #         df["weak_grid"] = 1
    # except:
    #     #print("NO Grid")
    #     df["weak_grid"] = pd.NA
    # df_list.append(df)

#df = pd.concat(df_list)

df = multi.envs[5].log
#All Grids x25
all_df = df["balance"][0]["reward"]
print("Mean Cost:" + str(all_df.mean()))
print("Total Cost:" + str(all_df.sum()))

# #----- Only needed when it is possible to run multiple grids with a agent
# # Grid Only x7
# grid_only = df[pd.notna(df['grid'][0]["reward"]) & pd.isna(df['genset'][0]["reward"])]
# print("Grid Only Mean:" + str(grid_only["balance"][0]["reward"].mean()))
# print("Grid Only Std:" + str(grid_only["balance"][0]["reward"].std()))
# print("Grid Only Sum:" + str(grid_only["balance"][0]["reward"].sum()/7))
# len(df[pd.notna(df['grid'][0]["reward"]) & pd.isna(df['genset'][0]["reward"])])/8758
# # Genset Only x10
# genset_only = df[pd.isna(df['grid'][0]["reward"]) & pd.notna(df['genset'][0]["reward"])]
# print("Genset Only Mean:" + str(genset_only["balance"][0]["reward"].mean()))
# print("Genset Only Std:" + str(genset_only["balance"][0]["reward"].std()))
# print("Genset Only Sum:" + str(genset_only["balance"][0]["reward"].sum()/10))
# len(df[pd.isna(df['grid'][0]["reward"]) & pd.notna(df['genset'][0]["reward"])])/8758
# # Grid + Genset x4
# grid_genset = df[pd.notna(df['grid'][0]["reward"]) & pd.notna(df['genset'][0]["reward"]) & df["weak_grid"].eq(0)]
# print("Grid + Genset Mean:" + str(grid_genset["balance"][0]["reward"].mean()))
# print("Grid + Genset Std:" + str(grid_genset["balance"][0]["reward"].std()))
# print("Grid + Genset Sum:" + str(grid_genset["balance"][0]["reward"].sum()/4))
# len(df[pd.notna(df['grid'][0]["reward"]) & pd.notna(df['genset'][0]["reward"]) & df["weak_grid"].eq(0)] )/8758
# # Genset + Weak Grid x4
# grid_genset_weak = df[pd.notna(df['grid'][0]["reward"]) & pd.notna(df['genset'][0]["reward"]) & df["weak_grid"].eq(1)]
# print("Genset + Weak Grid Mean:" + str(grid_genset_weak["balance"][0]["reward"].mean()))
# print("Genset + Weak Grid Std:" + str(grid_genset_weak["balance"][0]["reward"].std()))
# print("Genset + Weak Grid Sum:" + str(grid_genset_weak["balance"][0]["reward"].sum()/4))
# len(df[pd.notna(df['grid'][0]["reward"]) & pd.notna(df['genset'][0]["reward"]) & df["weak_grid"].eq(1)] )/8758
