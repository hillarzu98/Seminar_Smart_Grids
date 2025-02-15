import pandas as pd
from pymgrid import Microgrid
from pymgrid.algos import ModelPredictiveControl
from pymgrid.algos import RuleBasedControl
from pymgrid.modules.base import BaseTimeSeriesMicrogridModule
from pymgrid.forecast import OracleForecaster, GaussianNoiseForecaster

mpc_list = []
rbc_list = []

for i in range(0,25,1):
    microgrid = Microgrid.from_scenario(microgrid_number=i)
    mpc = ModelPredictiveControl(microgrid)

    mpc.reset()
    mpc_results = mpc.run()
    mpc_results.to_csv(f"mpc_{i}.csv")
    mpc_list.append(mpc_results)

    rbc = RuleBasedControl(microgrid)

    rbc.reset()
    rbc_results = rbc.run()
    rbc_results.to_csv(f"rbc_{i}.csv")
    rbc_list.append(rbc_results)

with pd.ExcelWriter('microgrid_results.xlsx',engine='xlsxwriter') as writer:
    # Write MPC results
    for idx, df in enumerate(mpc_list):
        df.to_excel(writer, sheet_name=f'MPC_Grid_{idx+1}')
    
    # Write RBC results
    for idx, df in enumerate(rbc_list):
        df.to_excel(writer, sheet_name=f'RBC_Grid_{idx+1}')

print("b")