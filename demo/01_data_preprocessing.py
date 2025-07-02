

import pandas as pd
import numpy as np
import os

# --- Configuration ---
INPUT_PATH = os.path.join('..', 'data', 'simulated_plant_data.csv')
OUTPUT_DIR = '.'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'processed_data.csv')

# --- Load Data ---
print(f"Loading data from {INPUT_PATH}...")
df = pd.read_csv(INPUT_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

print("Original data loaded:")
print(df.head())

# --- Feature Engineering: Unit Consumption (单耗) ---
print("\nCalculating unit consumption (energy_kwh / production_units)...")

# To avoid division by zero, we replace 0 production with NaN temporarily
production_no_zeros = df['production_units'].replace(0, np.nan)

df['unit_consumption'] = df['energy_kwh'] / production_no_zeros

# Handle infinite values and NaNs that resulted from the division.
# A common strategy is to fill them. We can use forward fill and then backfill.
df['unit_consumption'] = df['unit_consumption'].replace([np.inf, -np.inf], np.nan)
df['unit_consumption'].fillna(method='ffill', inplace=True)
df['unit_consumption'].fillna(method='bfill', inplace=True)

print("Data with unit consumption calculated:")
print(df.head())

# --- Save Processed Data ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(OUTPUT_PATH)

print(f"\nProcessed data successfully saved to {OUTPUT_PATH}")

