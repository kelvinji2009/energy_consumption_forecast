
import pandas as pd
import numpy as np
import os

# --- Configuration ---
N_DAYS = 365
HOURS_IN_DAY = 24
N_POINTS = N_DAYS * HOURS_IN_DAY
START_DATE = "2024-01-01"

# --- Time Index ---
time_index = pd.to_datetime(pd.date_range(start=START_DATE, periods=N_POINTS, freq="H"))

# --- Simulate Production (产量) ---
# Weekly pattern: lower on weekends (day 5 and 6)
weekly_cycle = np.sin(np.linspace(0, N_DAYS / 7 * 2 * np.pi, N_POINTS)) * -1 + 1
day_of_week = time_index.dayofweek
weekend_mask = (day_of_week >= 5)
# Simulate lower production on weekends
production_weekend_factor = np.ones(N_POINTS)
production_weekend_factor[weekend_mask] = np.random.uniform(0.1, 0.3, size=np.sum(weekend_mask))


# Daily pattern: lower during night hours (e.g., 0-6)
daily_cycle = np.sin(np.linspace(0, N_DAYS * 2 * np.pi, N_POINTS))
hour_of_day = time_index.hour
night_mask = (hour_of_day <= 6) | (hour_of_day >= 22)
# Simulate lower production at night
production_night_factor = np.ones(N_POINTS)
production_night_factor[night_mask] = np.random.uniform(0.0, 0.2, size=np.sum(night_mask))


# Combine cycles and add noise
production = (100 + daily_cycle * 50 + weekly_cycle * 30) * production_weekend_factor * production_night_factor
production[production < 0] = 0
production += np.random.normal(0, 5, N_POINTS)
production = np.maximum(0, production).astype(int)

# --- Simulate Temperature (温度) ---
# Seasonal pattern (yearly cycle)
seasonal_cycle_temp = -np.cos(np.linspace(0, 2 * np.pi, N_POINTS)) * 10 + 15 # Varies between 5 and 25
# Daily pattern
daily_cycle_temp = np.sin(np.linspace(0, N_DAYS * 2 * np.pi, N_POINTS)) * 5 # +/- 5 degrees daily
temperature = seasonal_cycle_temp + daily_cycle_temp + np.random.normal(0, 1, N_POINTS)

# --- Simulate Humidity (湿度) ---
# Inversely related to temperature with some lag and noise
humidity = 70 - (temperature - temperature.mean()) * 1.5 + np.random.normal(0, 5, N_POINTS)
humidity = np.clip(humidity, 20, 95) # Clip to realistic bounds

# --- Simulate Energy Consumption (能耗) ---
# Base load + production component + temperature component + anomalies
base_load = 50
production_component = production * 1.2
# HVAC component: more energy when temp is very high or very low
temp_deviation = np.abs(temperature - 18) # Assume 18C is optimal
hvac_component = temp_deviation * 5
# Combine
energy = base_load + production_component + hvac_component + np.random.normal(0, 10, N_POINTS)

# Inject some anomalies
# Anomaly 1: High energy despite low production (e.g., equipment left on)
anomaly_idx_1 = np.random.randint(0, N_POINTS - 24)
production[anomaly_idx_1 : anomaly_idx_1 + 5] = 0
energy[anomaly_idx_1 : anomaly_idx_1 + 5] *= np.random.uniform(2.5, 3.5)


# Anomaly 2: Unusually high energy spike
anomaly_idx_2 = np.random.randint(0, N_POINTS - 24)
energy[anomaly_idx_2 : anomaly_idx_2 + 3] *= np.random.uniform(2, 3)


energy = np.maximum(0, energy)

# --- Create DataFrame and Save ---
df = pd.DataFrame({
    "timestamp": time_index,
    "energy_kwh": energy,
    "production_units": production,
    "temperature_celsius": temperature,
    "humidity_percent": humidity
})

# Create directory and save file
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "simulated_plant_data.csv")
df.to_csv(output_path, index=False)

print(f"Successfully generated and saved data to {output_path}")
