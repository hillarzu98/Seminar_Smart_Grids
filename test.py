import numpy as np
import pandas as pd

np.random.seed(0)

from pymgrid import Microgrid
from pymgrid.modules import (
    BatteryModule,
    LoadModule,
    RenewableModule,
    GridModule)
small_battery = BatteryModule(min_capacity=10,
                              max_capacity=100,
                              max_charge=50,
                              max_discharge=50,
                              efficiency=0.9,
                              init_soc=0.2)

large_battery = BatteryModule(min_capacity=10,
                              max_capacity=1000,
                              max_charge=10,
                              max_discharge=10,
                              efficiency=0.7,
                              init_soc=0.2)
load_ts = 100+100*np.random.rand(24*90) # random load data in the range [100, 200].
pv_ts = 200*np.random.rand(24*90) # random pv data in the range [0, 200].

load = LoadModule(time_series=load_ts)

pv = RenewableModule(time_series=pv_ts)

grid_ts = [0.2, 0.1, 0.5] * np.ones((24*90, 3))

grid = GridModule(max_import=100,
                  max_export=100,
                  time_series=grid_ts)

modules = [
    small_battery,
    large_battery,
    ('pv', pv),
    load,
    grid]

microgrid = Microgrid(modules)

print(microgrid)
print(microgrid.modules.pv)
print(microgrid.modules.grid is microgrid.modules['grid'])

load = -1.0 * microgrid.modules.load.item().current_load
pv = microgrid.modules.pv.item().current_renewable

net_load = load + pv # negative if load demand exceeds pv

if net_load > 0:
    net_load = 0.0

battery_0_discharge = min(-1*net_load, microgrid.modules.battery[0].max_production)
net_load += battery_0_discharge

battery_1_discharge = min(-1*net_load, microgrid.modules.battery[1].max_production)
net_load += battery_1_discharge

grid_import = min(-1*net_load, microgrid.modules.grid.item().max_production)

control = {"battery" : [battery_0_discharge, battery_1_discharge],
           "grid": [grid_import]}

#obs, reward, done, info = microgrid.step(control, normalized=False)

#print(obs, reward, done, info)

for _ in range(10):
    obs, reward, done, info = microgrid.step(control, normalized=False)
    print(obs, reward, done, info)

microgrid.log[[('load', 0, 'load_met'),
               ('load', 0, 'load_current'),
               ('pv', 0, 'renewable_used'),
               ('battery', 0, 'charge_amount'),
               ('battery', 0, 'discharge_amount'),
               ('battery', 1, 'charge_amount'),
               ('battery', 1, 'discharge_amount'),
               ('grid', 0, 'grid_export'),
               ('grid', 0, 'grid_import'),
               ('balancing', 0, 'loss_load')]].droplevel(axis=1, level=1).plot()