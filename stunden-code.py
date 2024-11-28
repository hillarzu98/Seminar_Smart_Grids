import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

# Seed für Reproduzierbarkeit
np.random.seed(0)

# Batteriekapazität und Eigenschaften
battery_capacity = 500  # maximale Kapazität in kWh
battery_soc = 0.5 * battery_capacity  # initialer Ladezustand
battery_charge_efficiency = 0.9
battery_discharge_efficiency = 0.9

# Historische Daten für Visualisierung
load_history = []
pv_history = []
load_met_history = []
pv_consumed_history = []
lost_load_history = []
battery_soc_history = []
battery_discharge_history = []
battery_charge_history = []

# Funktion zum Laden der CSV-Dateien
def load_generation_data_from_csv(file_path):
    """Lädt die Erzeugungsdaten aus einer CSV-Datei."""
    df = pd.read_csv(file_path, delimiter=";")
    df['Startzeitpunkt'] = pd.to_datetime(df['Startzeitpunkt'], format='%d.%m.%Y %H:%M')
    df['Endzeitpunkt'] = pd.to_datetime(df['Endzeitpunkt'], format='%d.%m.%Y %H:%M')
    return df

def load_consumption_data_from_csv(file_path):
    """Lädt die Verbrauchsdaten aus einer CSV-Datei."""
    df = pd.read_csv(file_path, delimiter=";")
    df['Startzeitpunkt'] = pd.to_datetime(df['Startzeitpunkt'], format='%d.%m.%Y %H:%M')
    df['Endzeitpunkt'] = pd.to_datetime(df['Endzeitpunkt'], format='%d.%m.%Y %H:%M')
    return df



def get_generation_data_from_csv(file_path):
    df = load_generation_data_from_csv(file_path)
    
    # Konvertiere Erzeugungswerte in kW und runde auf 2 Dezimalstellen
    # Für jede Spalte ab der 3. Spalte (index 2) sicherstellen, dass sie numerisch ist
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).round(2)  # NaN-Werte werden mit 0 ersetzt und gerundet

    df['Total Generation (kW)'] = df.iloc[:, 2:].sum(axis=1).round(2)
    
    # Extrahiere Stunden für die Erzeugung
    df['Startstunde'] = df['Startzeitpunkt'].dt.hour
    df['Endstunde'] = df['Endzeitpunkt'].dt.hour
    
    return df

def get_consumption_data_from_csv(file_path):
    df = load_consumption_data_from_csv(file_path)
    
    # Extrahiere Stunden für den Verbrauch
    df['Startstunde'] = df['Startzeitpunkt'].dt.hour
    df['Endstunde'] = df['Endzeitpunkt'].dt.hour
    
    return df

# Optimierungsfunktion
def optimize_battery_use(load, generation, soc):
    """Optimiert die Nutzung der Batterie zur Minimierung von Lost Load."""
    def objective(x):
        battery_discharge, battery_charge = x
        lost_load = load - generation - battery_discharge + battery_charge
        return max(0, lost_load)

    def battery_constraints(x):
        battery_discharge, battery_charge = x
        return [
            soc - battery_discharge / battery_discharge_efficiency,  # Batterie darf nicht überentladen
            (battery_capacity - soc) - battery_charge * battery_charge_efficiency  # Batterie darf nicht überladen
        ]

    bounds = [(0, min(load, soc * battery_discharge_efficiency)),  # Entladung begrenzen
              (0, min(generation, (battery_capacity - soc) / battery_charge_efficiency))]  # Ladung begrenzen

    result = minimize(
        objective,
        [0, 0],  # Startwerte
        bounds=bounds,
        constraints={"type": "ineq", "fun": battery_constraints},
        method="SLSQP"
    )
    return result.x

# CSV-Dateien einlesen
generation_file_path = 'path_to_generation_csv.csv'  # Erzeugung CSV
consumption_file_path = 'path_to_consumption_csv.csv'  # Verbrauch CSV

generation_data = get_generation_data_from_csv(r"C:\test\Stromerzeugung.csv")
consumption_data = get_consumption_data_from_csv(r"C:\test\Stromverbrauch.csv")

# Simulation für 24 Stunden
for hour in range(24):
    # Finde relevante Erzeugungs- und Verbrauchsdaten für die Stunde
    generation_for_hour = generation_data[generation_data['Startstunde'] <= hour]
    generation_for_hour = generation_for_hour[generation_for_hour['Endstunde'] >= hour]
    
    # Sicherstellen, dass die Generierung in numerische Werte konvertiert wird
    current_generation = float(generation_for_hour['Total Generation (kW)'].sum().round(2))
    
    consumption_for_hour = consumption_data[consumption_data['Startstunde'] <= hour]
    consumption_for_hour = consumption_for_hour[consumption_for_hour['Endstunde'] >= hour]
    
    # Beispielhafte Lastdaten: Berechnet zufällig Lasten für die Stunde
    current_load = consumption_for_hour['Endzeitpunkt'].apply(lambda x: 100 + np.random.rand() * 50).mean().round(2)

    # Optimierung der Batterie
    battery_discharge, battery_charge = optimize_battery_use(current_load, current_generation, battery_soc)
    
    # Batterie-SoC aktualisieren
    battery_soc -= battery_discharge / battery_discharge_efficiency
    battery_soc += battery_charge * battery_charge_efficiency
    
    # Berechnung von PV-Verbrauch, Lastdeckung und verlorener Last
    pv_consumed = min(current_load, current_generation) + battery_charge
    load_met = pv_consumed + battery_discharge
    lost_load = max(0, current_load - load_met)

    # Daten speichern
    load_history.append(current_load)
    pv_history.append(current_generation)
    load_met_history.append(load_met)
    pv_consumed_history.append(pv_consumed)
    lost_load_history.append(lost_load)
    battery_soc_history.append(battery_soc)
    battery_discharge_history.append(battery_discharge)
    battery_charge_history.append(battery_charge)


# Ergebnisse visualisieren
hours_list = list(range(24))

plt.figure(figsize=(14, 10))

# Subplot 1: Last, Erzeugung und verlorene Last
plt.subplot(2, 1, 1)
plt.plot(hours_list, load_history, label='Load (kW)', color='blue', marker='o', linestyle='-', alpha=0.7)
plt.plot(hours_list, pv_history, label='Total Generation (kW)', color='orange', marker='o', linestyle='-', alpha=0.7)
plt.plot(hours_list, lost_load_history, label='Lost Load (kW)', color='red', marker='o', linestyle='-', alpha=0.7)
plt.title("Load, Total Generation, and Lost Load Over 24 Hours")
plt.xlabel("Hour of Day")
plt.ylabel("Power (kW)")
plt.xticks(hours_list)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Batteriezustand, Ladung und Entladung
plt.subplot(2, 1, 2)
plt.plot(hours_list, battery_soc_history, label='Battery SoC (kWh)', color='green', marker='o', linestyle='-', alpha=0.7)
plt.bar(hours_list, battery_discharge_history, label='Battery Discharge (kW)', color='purple', alpha=0.5, width=0.5)
plt.bar(hours_list, battery_charge_history, label='Battery Charge (kW)', color='yellow', alpha=0.5, width=0.5)
plt.title("Battery State of Charge and Actions Over 24 Hours")
plt.xlabel("Hour of Day")
plt.ylabel("Energy/Power (kW or kWh)")
plt.xticks(hours_list)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

'''

# Ergebnisse visualisieren (nur ein Schaubild)
plt.figure(figsize=(10, 6))

# Plot für Verbrauch (Load) und Erzeugung (Generation)
plt.plot(hours_list, load_history, label='Load (kW)', color='blue', marker='o', linestyle='-', alpha=0.7)
plt.plot(hours_list, pv_history, label='Total Generation (kW)', color='orange', marker='o', linestyle='-', alpha=0.7)

# Diagramm anpassen
plt.title("Load and Total Generation Over 24 Hours")
plt.xlabel("Hour of Day")
plt.ylabel("Power (kW)")
plt.xticks(hours_list)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Diagramm anzeigen
plt.tight_layout()
plt.show()
'''