import pandas as pd
import numpy as np
import os
import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.timeseries_generation import datetime_attribute_timeseries

# --- Configuration ---
PROCESSED_DATA_PATH = 'processed_data.csv'
MODEL_OUTPUT_DIR = 'models'
MODEL_NAME = 'tft_energy_model_no_past_cov' # Changed model name

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)

# --- 1. Load and Prepare Data ---
print("Loading processed data...")
df = pd.read_csv(PROCESSED_DATA_PATH, index_col='timestamp', parse_dates=True)
df = df.astype(np.float32)

# Create the main target TimeSeries for energy consumption
series_energy = TimeSeries.from_series(df['energy_kwh'], freq='H').astype(np.float32)

# Create future covariates TimeSeries (time-based features)
future_covariates = datetime_attribute_timeseries(
    series_energy,
    attribute="hour",
    one_hot=True
).stack(
    datetime_attribute_timeseries(series_energy, attribute="day_of_week", one_hot=True)
).astype(np.float32)

# --- 2. Preprocessing ---
print("Preprocessing data: scaling and splitting...")

# Split data into training and validation sets (leaving last 2 weeks for validation)
train_cutoff = series_energy.time_index[- (14 * 24)]
train_energy, val_energy = series_energy.split_before(train_cutoff)
train_future_cov, val_future_cov = future_covariates.split_before(train_cutoff)

# Scale all the series
scaler_energy = Scaler()
scaler_future_cov = Scaler()

train_energy_scaled = scaler_energy.fit_transform(train_energy)
val_energy_scaled = scaler_energy.transform(val_energy)

train_future_cov_scaled = scaler_future_cov.fit_transform(train_future_cov)
val_future_cov_scaled = scaler_future_cov.transform(val_future_cov)

# --- 3. Model Training ---
print("Initializing and training the TFT model for total energy consumption (no past covariates)...")

# Define the model with hyperparameters
# We'll use a lookback window of 1 week (168 hours) to predict the next day (24 hours)
input_chunk_length = 24 * 7
output_chunk_length = 24

model_energy = TFTModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    hidden_size=64,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=16,
    n_epochs=20,  # Using a smaller number of epochs for this demo
    add_relative_index=False,
    add_encoders=None,
    likelihood=None,  # Use a point prediction loss
    random_state=42,
    model_name=MODEL_NAME,
    work_dir=MODEL_OUTPUT_DIR,
    save_checkpoints=True,
    force_reset=True,
)

model_energy.fit(
    series=train_energy_scaled,
    # Removed past_covariates from training
    future_covariates=train_future_cov_scaled,
    val_series=val_energy_scaled,
    # Removed val_past_covariates from training
    val_future_covariates=val_future_cov_scaled,
    verbose=True
)

# --- 4. Evaluation ---
print("\nEvaluating model on the validation set...")

# Perform historical forecasting (backtesting)
# This simulates how the model would have performed in the past
historical_forecasts_scaled = model_energy.historical_forecasts(
    series=val_energy_scaled,
    # Removed past_covariates from historical_forecasts
    future_covariates=val_future_cov_scaled,
    start=0.1, # Start forecasting after the first 10% of the validation series
    forecast_horizon=1, # Predict 1 hour ahead at each step
    stride=1,
    retrain=False,
    verbose=True
)

# Inverse transform the scaled predictions to get actual values
historical_forecasts = scaler_energy.inverse_transform(historical_forecasts_scaled)

# Calculate and print MAPE
mape_score = mape(val_energy, historical_forecasts)
print(f"\nMAPE on Validation Set: {mape_score:.2f}%")

# --- 5. Save Model ---
print(f"Saving the trained model...")
# The model checkpoints are already saved during training in the MODEL_OUTPUT_DIR
# We can load the best model with:
# model = TFTModel.load_from_checkpoint(model_name=MODEL_NAME, work_dir=MODEL_OUTPUT_DIR, best=True)
print(f"Model saved in directory: {os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME)}")
print("\nScript finished successfully.")
