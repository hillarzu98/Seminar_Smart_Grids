import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Seed für Reproduzierbarkeit
np.random.seed(0)

# pymgrid Module importieren
from pymgrid import Microgrid
from pymgrid.modules import BatteryModule, LoadModule, RenewableModule, GridModule

# Initialisiere Module
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

# Dynamische Lastdaten basierend auf Tageszeit
def generate_dynamic_load(hour):
    """Erzeugt dynamischen Verbrauch basierend auf der Tageszeit."""
    base_load = 100
    if 6 <= hour < 18:  # Tagsüber höhere Last
        return base_load + 50 * np.random.rand()  # Variiert um 50 kW
    else:  # Nachts niedrigere Last
        return base_load + 20 * np.random.rand()  # Variiert um 20 kW

# Dynamische PV-Daten basierend auf Tageszeit
def generate_dynamic_pv(hour):
    """Erzeugt dynamische PV-Produktion basierend auf der Tageszeit."""
    if 6 <= hour < 18:  # Tagsüber PV-Erzeugung
        return 200 * np.sin((hour - 6) * np.pi / 12) + 20 * np.random.rand()
    else:  # Nachts keine PV-Erzeugung
        return 0

# Netz-Zeitreihe initialisieren
grid_ts = [0.2, 0.1, 0.5] * np.ones((24 * 90, 3))

# Netz-Modul erstellen
grid = GridModule(max_import=100, max_export=100, time_series=grid_ts)

# Module in das Microgrid hinzufügen
modules = [
    small_battery,
    large_battery,
    grid
]

microgrid = Microgrid(modules)

# Simulationszeitraum (90 Tage in Stunden)
hours = 24 * 90

# Arrays für Last und PV-Werte
load_data = []
pv_data = []

# Simulation
for hour in range(hours):
    # Dynamische Last und PV-Daten berechnen
    current_load = generate_dynamic_load(hour % 24)  # Modulo 24 für die Stunde des Tages
    current_pv = generate_dynamic_pv(hour % 24)
    
    # Daten für Visualisierung speichern
    load_data.append(current_load)
    pv_data.append(current_pv)

# Konvertiere Listen zu numpy-Arrays für Diagramm
load_data = np.array(load_data)
pv_data = np.array(pv_data)

# Zeitachse in Tagen (für einfachere Visualisierung)
days = np.arange(hours // 24)

# Durchschnittswerte pro Tag berechnen
load_daily_avg = load_data.reshape(-1, 24).mean(axis=1)
pv_daily_avg = pv_data.reshape(-1, 24).mean(axis=1)

# Plot erstellen
plt.figure(figsize=(12, 6))

# Dynamische Last plotten
plt.plot(days, load_daily_avg, label='Daily Average Load (Consumption)', color='blue', marker='o', linestyle='-', alpha=0.7)
# Dynamische PV-Produktion plotten
plt.plot(days, pv_daily_avg, label='Daily Average PV Production', color='orange', marker='o', linestyle='-', alpha=0.7)

# Diagramm-Beschriftungen
plt.title("Daily Average Load and PV Production over 90 Days with Dynamic Data")
plt.xlabel("Day")
plt.ylabel("Power (kW)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Diagramm anzeigen
plt.show()