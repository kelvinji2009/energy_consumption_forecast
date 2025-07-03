import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.ad.detectors import QuantileDetector

# --- Configuration ---
PROCESSED_DATA_PATH = 'processed_data.csv'
MODEL_DIR = 'models'
MODEL_NAME = 'tft_energy_model_no_past_cov' # Changed to load the new model
PLOT_OUTPUT_DIR = 'plots'
PLOT_FILENAME = 'anomaly_detection_and_forecast_2025_H1.png'

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)

# --- 1. Load Data and Model ---
print("Loading processed data and trained model...")
df = pd.read_csv(PROCESSED_DATA_PATH, index_col='timestamp', parse_dates=True)
df = df.astype(np.float32)

# Load the best model from the training checkpoints
model_energy = TFTModel.load_from_checkpoint(model_name=MODEL_NAME, work_dir=MODEL_DIR, best=True)

# --- 2. Recreate TimeSeries and Preprocessing Objects ---
print("Recreating TimeSeries and preprocessing steps...")

# Create the main target TimeSeries for energy consumption
series_energy = TimeSeries.from_series(df['energy_kwh'], freq='H').astype(np.float32)

# Create past covariates TimeSeries (still needed for anomaly detection historical forecasts)
past_covariates = TimeSeries.from_dataframe(df, value_cols=['production_units', 'temperature_celsius', 'humidity_percent'], freq='H').astype(np.float32)

# Create future covariates TimeSeries
future_covariates = datetime_attribute_timeseries(
    series_energy,
    attribute="hour",
    one_hot=True
).stack(
    datetime_attribute_timeseries(series_energy, attribute="day_of_week", one_hot=True)
).astype(np.float32)

# Split data exactly as in the training script
train_cutoff = series_energy.time_index[- (14 * 24)]
train_energy, val_energy = series_energy.split_before(train_cutoff)
train_past_cov, val_past_cov = past_covariates.split_before(train_cutoff)
train_future_cov, val_future_cov = future_covariates.split_before(train_cutoff)

# Re-fit the scalers on the training data to ensure consistency
scaler_energy = Scaler()
scaler_past_cov = Scaler()
scaler_future_cov = Scaler()

train_energy_scaled = scaler_energy.fit_transform(train_energy)
val_energy_scaled = scaler_energy.transform(val_energy)
train_past_cov_scaled = scaler_past_cov.fit_transform(train_past_cov)
val_past_cov_scaled = scaler_past_cov.transform(val_past_cov)
train_future_cov_scaled = scaler_future_cov.fit_transform(train_future_cov)
val_future_cov_scaled = scaler_future_cov.transform(val_future_cov)

# Scale the full series for prediction context
series_energy_scaled = scaler_energy.transform(series_energy)
# past_covariates_scaled = scaler_past_cov.transform(past_covariates) # Not needed for predict with new model

# Define input_chunk_length and output_chunk_length (same as in 02_train_and_evaluate.py)
input_chunk_length = 24 * 7
output_chunk_length = 24

# --- 3. Anomaly Detection (Manual Residual Calculation) ---
print("Setting up anomaly detection with manual residual calculation...")

# Generate historical forecasts for training data
# This will produce scaled forecasts
forecasts_train_scaled = model_energy.historical_forecasts(
    series=train_energy_scaled,
    # past_covariates=train_past_cov_scaled, # Removed for consistency with new model
    future_covariates=train_future_cov_scaled,
    start=input_chunk_length, # Start after the first input_chunk_length to get valid forecasts
    forecast_horizon=1,
    stride=1,
    retrain=False,
    verbose=False # Suppress verbose output during scoring
)

# Calculate absolute residuals for training data
# Ensure alignment for subtraction
diff_train = train_energy_scaled[forecasts_train_scaled.time_index] - forecasts_train_scaled
scores_train_series = pd.Series(np.abs(diff_train.all_values()).reshape(-1), index=diff_train.time_index)
scores_train = TimeSeries.from_series(scores_train_series)

# Initialize QuantileDetector
detector = QuantileDetector(high_quantile=0.98)

print("Fitting the detector on training data residuals...")
detector.fit(scores_train)

print("Detecting anomalies on the validation data...")
# Generate historical forecasts for validation data
forecasts_val_scaled = model_energy.historical_forecasts(
    series=val_energy_scaled,
    # past_covariates=val_past_cov_scaled, # Removed for consistency with new model
    future_covariates=val_future_cov_scaled,
    start=input_chunk_length, # Start after the first input_chunk_length
    forecast_horizon=1,
    stride=1,
    retrain=False,
    verbose=False # Suppress verbose output during scoring
)

# Calculate absolute residuals for validation data
# Ensure alignment for subtraction
diff_val = val_energy_scaled[forecasts_val_scaled.time_index] - forecasts_val_scaled
scores_val_series = pd.Series(np.abs(diff_val.all_values()).reshape(-1), index=diff_val.time_index)
scores_val = TimeSeries.from_series(scores_val_series)

anomalies = detector.detect(scores_val)

# --- 4. Future Forecasting ---
print("Generating future forecasts...")

# Define the forecast horizon (Jan-Jun 2025)
forecast_horizon = 4344 # 6 months * approx 30.4375 days/month * 24 hours/day

# Determine the start time for future covariates to include the input_chunk_length history
future_cov_start_time = series_energy.end_time() - series_energy.freq * (input_chunk_length - 1)

# Create future time index for forecasting, including the historical part needed for input_chunk_length
full_future_time_index = pd.date_range(
    start=future_cov_start_time,
    periods=input_chunk_length + forecast_horizon,
    freq=series_energy.freq
)

# Create a dummy TimeSeries for future covariates generation
dummy_future_series = TimeSeries.from_times_and_values(full_future_time_index, np.zeros(len(full_future_time_index)))

# Generate future covariates (hour, day_of_week) for the full extended period
future_covariates_forecast = datetime_attribute_timeseries(
    dummy_future_series,
    attribute="hour",
    one_hot=True
).stack(
    datetime_attribute_timeseries(dummy_future_series, attribute="day_of_week", one_hot=True)
).astype(np.float32)

# Scale future covariates using the same scaler as training
future_covariates_forecast_scaled = scaler_future_cov.transform(future_covariates_forecast)

# Make the forecast
# We need to provide the entire series_energy_scaled as input to the predict method
# and the future_covariates_forecast_scaled
forecast_scaled = model_energy.predict(
    n=forecast_horizon,
    series=series_energy_scaled, # Provide the entire scaled series for context
    # Removed past_covariates from predict as they are not available for future prediction
    future_covariates=future_covariates_forecast_scaled # Provide future covariates for the forecast horizon
)

# Inverse transform the forecast to get actual energy values
forecast = scaler_energy.inverse_transform(forecast_scaled)

# --- 5. Visualization ---
print("Visualizing results and saving plot...")

# Create the output directory if it doesn't exist
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

plt.figure(figsize=(18, 8)) # Increased figure size for better visibility

# Plot the actual energy consumption (entire series)
series_energy.plot(label='Actual Energy Consumption', color='blue')

# Plot the detected anomalies (only on validation part)
val_energy_sliced = val_energy.slice_intersect(anomalies)
(val_energy_sliced * anomalies).plot(label='Detected Anomaly', lw=4, c='red')

# Plot the future forecast
forecast.plot(label='Energy Forecast (2025 H1)', color='green', linestyle='--')

plt.title('Energy Consumption: Anomaly Detection and Future Forecast')
plt.xlabel('Timestamp')
plt.ylabel('Energy (kWh)')
plt.legend()

# Save the plot
plot_path = os.path.join(PLOT_OUTPUT_DIR, PLOT_FILENAME)
plt.savefig(plot_path)

print(f"\nAnomaly detection plot saved to: {plot_path}")
print("Script finished successfully.")
