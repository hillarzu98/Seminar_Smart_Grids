import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Seed für Reproduzierbarkeit
np.random.seed(0)

# pymgrid Module importieren
from pymgrid import Microgrid
from pymgrid.modules import BatteryModule, LoadModule, RenewableModule, GridModule

# Initialisiere die Batterie
battery_capacity = 500  # maximale Kapazität in kWh
battery_soc = 0.5 * battery_capacity  # initialer Ladezustand
battery_charge_efficiency = 0.9
battery_discharge_efficiency = 0.9

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

# Simulationseinstellungen
hours = 24 * 90

# Ergebnis-Arrays für Load Met, PV Consumed, Lost Load
load_met_data = []
pv_consumed_data = []
lost_load_data = []

# Simulation
for hour in range(hours):
    # Dynamische Last und PV-Erzeugung berechnen
    current_load = generate_dynamic_load(hour % 24)  # Modulo 24 für die Stunde des Tages
    current_pv = generate_dynamic_pv(hour % 24)
    
    # Berechnung des PV-Verbrauchs und der Batterienutzung
    pv_consumed = min(current_load, current_pv)  # PV wird zuerst verwendet, um Last zu decken
    remaining_load = current_load - pv_consumed  # übriggebliebene Last nach PV
    load_met = pv_consumed  # Initialisiere Load Met mit PV-Anteil
    
    # Batterienutzung für Restlast
    if remaining_load > 0:
        battery_discharge = min(battery_soc * battery_discharge_efficiency, remaining_load)
        load_met += battery_discharge
        battery_soc -= battery_discharge / battery_discharge_efficiency  # Batterie entladen
        remaining_load -= battery_discharge

    # Falls PV-Überschuss vorhanden, Batterie laden
    pv_surplus = current_pv - pv_consumed
    if pv_surplus > 0:
        battery_charge = min((battery_capacity - battery_soc) / battery_charge_efficiency, pv_surplus)
        pv_consumed += battery_charge  # zusätzlicher PV-Verbrauch durch Ladung
        battery_soc += battery_charge * battery_charge_efficiency  # Batterie laden
    
    # Berechnung von Lost Load (ungedeckte Last)
    lost_load = remaining_load if remaining_load > 0 else 0

    # Daten für Visualisierung speichern
    load_met_data.append(load_met)
    pv_consumed_data.append(pv_consumed)
    lost_load_data.append(lost_load)

# Konvertiere Listen zu numpy-Arrays für Diagramm
load_met_data = np.array(load_met_data)
pv_consumed_data = np.array(pv_consumed_data)
lost_load_data = np.array(lost_load_data)

# Zeitachse in Tagen (für einfachere Visualisierung)
days = np.arange(hours // 24)

# Durchschnittswerte pro Tag berechnen
load_met_daily_avg = load_met_data.reshape(-1, 24).mean(axis=1)
pv_consumed_daily_avg = pv_consumed_data.reshape(-1, 24).mean(axis=1)
lost_load_daily_avg = lost_load_data.reshape(-1, 24).mean(axis=1)

# Plot erstellen
plt.figure(figsize=(12, 6))

# Datenreihen plotten
plt.plot(days, load_met_daily_avg, label='Daily Average Load Met', color='green', marker='o', linestyle='-', alpha=0.7)
plt.plot(days, pv_consumed_daily_avg, label='Daily Average PV Consumed', color='orange', marker='o', linestyle='-', alpha=0.7)
plt.plot(days, lost_load_daily_avg, label='Daily Average Lost Load', color='red', marker='o', linestyle='-', alpha=0.7)

# Diagramm-Beschriftungen
plt.title("Daily Averages: Load Met, PV Consumed, and Lost Load over 90 Days")
plt.xlabel("Day")
plt.ylabel("Power (kW)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Diagramm anzeigen
plt.show()