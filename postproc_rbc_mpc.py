from pathlib import Path
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd

df_list = []
for file in Path("./orig_microgrid/implemented_algorithms/").glob("mpc*.csv"):
    df = pd.read_csv(file, header=[0,2])
    #print(file.name)
    print(df["balance"]["reward"].std())
    #print(df["balance"]["reward"].sum())
    try:
        # Create figure with specific size and additional height for legend
        plt.figure(figsize=(15, 10))  # Increased height to accommodate legend
        
        # Slice the data between timesteps 3000 and 4000
        slice_range = slice(3600, 3768)
        
        # Create line plots for each column with sliced data
        plt.plot(df['load']['load_current'][slice_range]*-1, label='Load (inverted)', linewidth=2)
        plt.plot(df['pv']['renewable_current'][slice_range], label='PV', linewidth=2)
        plt.plot(df['battery']['charge_amount'][slice_range], label='Battery Charge', linewidth=2)
        plt.plot(df['battery']['discharge_amount'][slice_range], label='Battery Discharge', linewidth=2)
        plt.plot(df['grid']['grid_import'][slice_range], label='Grid Import', linewidth=2)
        plt.plot(df['grid']['grid_export'][slice_range], label='Grid Export', linewidth=2)
        plt.plot(df['genset']['genset_production'][slice_range], label='Genset', linewidth=2)

        # Customize the plot
        plt.title('Microgrid Power Flow', pad=80, fontsize=14)  # Add padding above title
        plt.xlabel('Time Step [h]', fontsize=12)
        plt.ylabel('Power [kW]', fontsize=12)
        
        # Move legend to top with adjusted spacing
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), 
                  loc='lower left',
                  ncol=4, 
                  mode="expand", 
                  borderaxespad=0.2,
                  fontsize=12)  # Increased padding
        
        # Adjust layout with more top margin
        plt.tight_layout(rect=[0, 0, 1, 0.9000])  # Reserve top 10% for legend
        
        # Show the plot
        plt.show()
        # if file.name in "mpc_24.csv":
        #     print(file.name)
        if df['grid']["grid_status_current"].eq(1).all():
            #print("NO Weak Grid")
            df["weak_grid"] = 0
        else:
            #print("Weak Grid")
            df["weak_grid"] = 1
    except:
        #print("NO Grid")
        df["weak_grid"] = pd.NA
    df_list.append(df)
    #print(file.name)

df = pd.concat(df_list)

#All Grids x25
all_df = df["balance"]["reward"]
print("Mean Cost MPC:" + str(all_df.mean()))
print("Total Cost MPC:" + str(all_df.sum()))

# Grid Only x7
grid_only = df[pd.notna(df['grid']["reward"]) & pd.isna(df['genset']["reward"])]
print("Grid Only Mean:" + str(grid_only["balance"]["reward"].mean()))
print("Grid Only Std:" + str(grid_only["balance"]["reward"].std()))
print("Grid Only Sum:" + str(grid_only["balance"]["reward"].sum()/7))
len(df[pd.notna(df['grid']["reward"]) & pd.isna(df['genset']["reward"])])/8758
# Genset Only x10
genset_only = df[pd.isna(df['grid']["reward"]) & pd.notna(df['genset']["reward"])]
print("Genset Only Mean:" + str(genset_only["balance"]["reward"].mean()))
print("Genset Only Std:" + str(genset_only["balance"]["reward"].std()))
print("Genset Only Sum:" + str(genset_only["balance"]["reward"].sum()/10))
len(df[pd.isna(df['grid']["reward"]) & pd.notna(df['genset']["reward"])])/8758
# Grid + Genset x4
grid_genset = df[pd.notna(df['grid']["reward"]) & pd.notna(df['genset']["reward"]) & df["weak_grid"].eq(0)]
print("Grid + Genset Mean:" + str(grid_genset["balance"]["reward"].mean()))
print("Grid + Genset Std:" + str(grid_genset["balance"]["reward"].std()))
print("Grid + Genset Sum:" + str(grid_genset["balance"]["reward"].sum()/4))
len(df[pd.notna(df['grid']["reward"]) & pd.notna(df['genset']["reward"]) & df["weak_grid"].eq(0)] )/8758
# Genset + Weak Grid x4
grid_genset_weak = df[pd.notna(df['grid']["reward"]) & pd.notna(df['genset']["reward"]) & df["weak_grid"].eq(1)]
print("Genset + Weak Grid Mean:" + str(grid_genset_weak["balance"]["reward"].mean()))
print("Genset + Weak Grid Std:" + str(grid_genset_weak["balance"]["reward"].std()))
print("Genset + Weak Grid Sum:" + str(grid_genset_weak["balance"]["reward"].sum()/4))
len(df[pd.notna(df['grid']["reward"]) & pd.notna(df['genset']["reward"]) & df["weak_grid"].eq(1)] )/8758

print("--------------- RBC -----------------")
df_list = []
for file in Path("./orig_microgrid/").glob("rbc*.csv"):
    df = pd.read_csv(file, header=[0,2])
    #print(file.name)
    #print(df["balance"]["reward"].mean())
    print(df["balance"]["reward"].std())
    try:
        # if file.name in "mpc_24.csv":
        #     print(file.name)
        if df['grid']["grid_status_current"].eq(1).all():
            #print("NO Weak Grid")
            df["weak_grid"] = 0
        else:
            #print("Weak Grid")
            df["weak_grid"] = 1
    except:
        #print("NO Grid")
        df["weak_grid"] = pd.NA
    df_list.append(df)
    #print(file.name)

df = pd.concat(df_list)

#All Grids
all_df = df["balance"]["reward"]
print("Mean Cost MPC:" + str(all_df.mean()))
print("Total Cost MPC:" + str(all_df.sum()))

# Grid Only
grid_only = df[pd.notna(df['grid']["reward"]) & pd.isna(df['genset']["reward"])]
print("Grid Only Mean:" + str(grid_only["balance"]["reward"].mean()))
print("Grid Only Std:" + str(grid_only["balance"]["reward"].std()))
print("Grid Only Sum:" + str(grid_only["balance"]["reward"].sum()/7))
len(df[pd.notna(df['grid']["reward"]) & pd.isna(df['genset']["reward"])])/8758
# Genset Only
genset_only = df[pd.isna(df['grid']["reward"]) & pd.notna(df['genset']["reward"])]
print("Genset Only Mean:" + str(genset_only["balance"]["reward"].mean()))
print("Genset Only Std:" + str(genset_only["balance"]["reward"].std()))
print("Genset Only Sum:" + str(genset_only["balance"]["reward"].sum()/10))
len(df[pd.isna(df['grid']["reward"]) & pd.notna(df['genset']["reward"])])/8758
# Grid + Genset
grid_genset = df[pd.notna(df['grid']["reward"]) & pd.notna(df['genset']["reward"]) & df["weak_grid"].eq(0)]
print("Grid + Genset Mean:" + str(grid_genset["balance"]["reward"].mean()))
print("Grid + Genset Std:" + str(grid_genset["balance"]["reward"].std()))
print("Grid + Genset Sum:" + str(grid_genset["balance"]["reward"].sum()/4))
len(df[pd.notna(df['grid']["reward"]) & pd.notna(df['genset']["reward"]) & df["weak_grid"].eq(0)] )/8758
# Genset + Weak Grid
grid_genset_weak = df[pd.notna(df['grid']["reward"]) & pd.notna(df['genset']["reward"]) & df["weak_grid"].eq(1)]
print("Genset + Weak Grid Mean:" + str(grid_genset_weak["balance"]["reward"].mean()))
print("Genset + Weak Grid Std:" + str(grid_genset_weak["balance"]["reward"].std()))
print("Genset + Weak Grid Sum:" + str(grid_genset_weak["balance"]["reward"].sum()/4))
len(df[pd.notna(df['grid']["reward"]) & pd.notna(df['genset']["reward"]) & df["weak_grid"].eq(1)] )/8758