import numpy as np
from pymgrid.envs import DiscreteMicrogridEnv

for i in range(0, 2):
    env = DiscreteMicrogridEnv.from_scenario(microgrid_number=i)
    
    print(len(env.get_priority_lists(remove_redundant_gensets=False)))
    print(env.get_action_mask())
    #print(len(env._get_obs()))
    env.step(1)