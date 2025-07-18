import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, Any, Callable

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import LightGBMModel, TiDEModel, RNNModel, TFTModel
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.metrics import mape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.ad import QuantileDetector

# For reproducibility
np.random.seed(42)

def train_lgbm_model(
    data: pd.DataFrame,
    target_column: str,
    input_chunk_length: int,
    output_chunk_length: int,
    n_epochs: int = 20 # Added n_epochs, though not used by LightGBM
) -> Tuple[ForecastingModel, Scaler, Scaler, Dict[str, Any]]:
    """Trains a LightGBM forecasting model."""
    # (Implementation is the same as before, but now it's a helper function)
    print("--- Starting LGBM Model Training ---")
    series_target = TimeSeries.from_series(data[target_column], freq='H').astype(np.float32)
    future_covariates = datetime_attribute_timeseries(series_target, attribute="hour", one_hot=True).stack(
        datetime_attribute_timeseries(series_target, attribute="day_of_week", one_hot=True)
    ).astype(np.float32)

    train_target, val_target = series_target.split_before(0.8)
    train_cov, val_cov = future_covariates.split_before(0.8)

    scaler_target = Scaler()
    scaler_cov = Scaler()
    train_target_scaled = scaler_target.fit_transform(train_target)
    val_target_scaled = scaler_target.transform(val_target)
    train_cov_scaled = scaler_cov.fit_transform(train_cov)
    val_cov_scaled = scaler_cov.transform(val_cov)

    model = LightGBMModel(
        lags=input_chunk_length,
        lags_future_covariates=[0, output_chunk_length - 1],
        output_chunk_length=output_chunk_length,
        random_state=42,
        force_reset=True,
    )
    model.fit(series=train_target_scaled, future_covariates=train_cov_scaled)
    
    historical_forecasts_scaled = model.historical_forecasts(val_target_scaled, future_covariates=val_cov_scaled, start=0.1, forecast_horizon=1, stride=1, retrain=False, verbose=True)
    historical_forecasts = scaler_target.inverse_transform(historical_forecasts_scaled)
    mape_score = mape(val_target, historical_forecasts)
    metrics = {'mape': float(mape_score.item())}
    print(f"--- LGBM Validation MAPE: {mape_score:.2f}% ---")
    return model, scaler_target, scaler_cov, metrics

def train_tide_model(
    data: pd.DataFrame,
    target_column: str,
    input_chunk_length: int,
    output_chunk_length: int,
    n_epochs: int = 20 # Added n_epochs
) -> Tuple[ForecastingModel, Scaler, Scaler, Dict[str, Any]]:
    """Trains a TiDE forecasting model."""
    print("--- Starting TiDE Model Training ---")
    series_target = TimeSeries.from_series(data[target_column], freq='H').astype(np.float32)
    future_covariates = datetime_attribute_timeseries(series_target, attribute="hour", one_hot=True).stack(
        datetime_attribute_timeseries(series_target, attribute="day_of_week", one_hot=True)
    ).astype(np.float32)

    train_target, val_target = series_target.split_before(0.8)
    train_cov, val_cov = future_covariates.split_before(0.8)

    scaler_target = Scaler()
    scaler_cov = Scaler()
    train_target_scaled = scaler_target.fit_transform(train_target)
    val_target_scaled = scaler_target.transform(val_target)
    train_cov_scaled = scaler_cov.fit_transform(train_cov)
    val_cov_scaled = scaler_cov.transform(val_cov)

    model = TiDEModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=64,
        n_epochs=n_epochs, # Used n_epochs
        random_state=42,
        force_reset=True,
    )
    model.fit(series=train_target_scaled, future_covariates=train_cov_scaled, val_series=val_target_scaled, val_future_covariates=val_cov_scaled, verbose=True)

    historical_forecasts_scaled = model.historical_forecasts(val_target_scaled, future_covariates=val_cov_scaled, start=0.1, forecast_horizon=1, stride=1, retrain=False, verbose=True)
    historical_forecasts = scaler_target.inverse_transform(historical_forecasts_scaled)
    mape_score = mape(val_target, historical_forecasts)
    metrics = {'mape': float(mape_score.item())}
    print(f"--- TiDE Validation MAPE: {mape_score:.2f}% ---")
    return model, scaler_target, scaler_cov, metrics

def train_lstm_model(
    data: pd.DataFrame,
    target_column: str,
    input_chunk_length: int,
    output_chunk_length: int,
    n_epochs: int = 20 # Added n_epochs
) -> Tuple[ForecastingModel, Scaler, Scaler, Dict[str, Any]]:
    """Trains an LSTM forecasting model."""
    print("--- Starting LSTM Model Training ---")
    series_target = TimeSeries.from_series(data[target_column], freq='H').astype(np.float32)
    future_covariates = datetime_attribute_timeseries(series_target, attribute="hour", one_hot=True).stack(
        datetime_attribute_timeseries(series_target, attribute="day_of_week", one_hot=True)
    ).astype(np.float32)

    train_target, val_target = series_target.split_before(0.8)
    train_cov, val_cov = future_covariates.split_before(0.8)

    scaler_target = Scaler()
    scaler_cov = Scaler()
    train_target_scaled = scaler_target.fit_transform(train_target)
    val_target_scaled = scaler_target.transform(val_target)
    train_cov_scaled = scaler_cov.fit_transform(train_cov)
    val_cov_scaled = scaler_cov.transform(val_cov)

    model = RNNModel(
        model='LSTM',
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        training_length=input_chunk_length,
        n_epochs=n_epochs, # Used n_epochs
        random_state=42,
        force_reset=True,
    )
    model.fit(series=train_target_scaled, future_covariates=train_cov_scaled, val_series=val_target_scaled, val_future_covariates=val_cov_scaled, verbose=True)

    historical_forecasts_scaled = model.historical_forecasts(val_target_scaled, future_covariates=val_cov_scaled, start=0.1, forecast_horizon=1, stride=1, retrain=False, verbose=True)
    historical_forecasts = scaler_target.inverse_transform(historical_forecasts_scaled)
    mape_score = mape(val_target, historical_forecasts)
    metrics = {'mape': float(mape_score.item())}
    print(f"--- LSTM Validation MAPE: {mape_score:.2f}% ---")
    return model, scaler_target, scaler_cov, metrics

def train_tft_model(
    data: pd.DataFrame,
    target_column: str,
    input_chunk_length: int,
    output_chunk_length: int,
    n_epochs: int = 20 # Added n_epochs
) -> Tuple[ForecastingModel, Scaler, Scaler, Scaler, Dict[str, Any]]: # Added Scaler for past_covariates
    """Trains a TFT forecasting model."""
    print("--- Starting TFT Model Training ---")
    series_target = TimeSeries.from_series(data[target_column], freq='H').astype(np.float32)
    
    # Past covariates for TFT
    past_covariates = TimeSeries.from_dataframe(data, value_cols=['production_units', 'temperature_celsius', 'humidity_percent'], freq='H').astype(np.float32)

    future_covariates = datetime_attribute_timeseries(series_target, attribute="hour", one_hot=True).stack(
        datetime_attribute_timeseries(series_target, attribute="day_of_week", one_hot=True)
    ).astype(np.float32)

    train_target, val_target = series_target.split_before(0.8)
    train_past_cov, val_past_cov = past_covariates.split_before(0.8) # Split past covariates
    train_cov, val_cov = future_covariates.split_before(0.8)

    scaler_target = Scaler()
    scaler_past_cov = Scaler() # Scaler for past covariates
    scaler_cov = Scaler()

    train_target_scaled = scaler_target.fit_transform(train_target)
    val_target_scaled = scaler_target.transform(val_target)
    train_past_cov_scaled = scaler_past_cov.fit_transform(train_past_cov) # Scale past covariates
    val_past_cov_scaled = scaler_past_cov.transform(val_past_cov) # Scale past covariates
    train_cov_scaled = scaler_cov.fit_transform(train_cov)
    val_cov_scaled = scaler_cov.transform(val_cov)

    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=16,
        n_epochs=n_epochs, # Used n_epochs
        random_state=42,
        force_reset=True,
    )
    model.fit(
        series=train_target_scaled,
        past_covariates=train_past_cov_scaled, # Pass scaled past covariates
        future_covariates=train_cov_scaled,
        val_series=val_target_scaled,
        val_past_covariates=val_past_cov_scaled, # Pass scaled past covariates
        val_future_covariates=val_cov_scaled,
        verbose=True
    )

    historical_forecasts_scaled = model.historical_forecasts(
        series=val_target_scaled,
        past_covariates=val_past_cov_scaled, # Pass scaled past covariates
        future_covariates=val_cov_scaled,
        start=0.1,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=True
    )
    historical_forecasts = scaler_target.inverse_transform(historical_forecasts_scaled)
    mape_score = mape(val_target, historical_forecasts)
    metrics = {'mape': float(mape_score.item())}
    print(f"--- TFT Validation MAPE: {mape_score:.2f}% ---")
    return model, scaler_target, scaler_past_cov, scaler_cov, metrics # Return past_cov_scaler

def train_tft_no_past_cov_model(
    data: pd.DataFrame,
    target_column: str,
    input_chunk_length: int,
    output_chunk_length: int,
    n_epochs: int = 20 # Added n_epochs
) -> Tuple[ForecastingModel, Scaler, Scaler, Dict[str, Any]]: # No Scaler for past_covariates
    """Trains a TFT forecasting model without past covariates."""
    print("--- Starting TFT (No Past Covariates) Model Training ---")
    series_target = TimeSeries.from_series(data[target_column], freq='H').astype(np.float32)
    
    future_covariates = datetime_attribute_timeseries(series_target, attribute="hour", one_hot=True).stack(
        datetime_attribute_timeseries(series_target, attribute="day_of_week", one_hot=True)
    ).astype(np.float32)

    train_target, val_target = series_target.split_before(0.8)
    train_cov, val_cov = future_covariates.split_before(0.8)

    scaler_target = Scaler()
    scaler_cov = Scaler()

    train_target_scaled = scaler_target.fit_transform(train_target)
    val_target_scaled = scaler_target.transform(val_target)
    train_cov_scaled = scaler_cov.fit_transform(train_cov)
    val_cov_scaled = scaler_cov.transform(val_cov)

    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=16,
        n_epochs=n_epochs, # Used n_epochs
        random_state=42,
        force_reset=True,
    )
    model.fit(
        series=train_target_scaled,
        future_covariates=train_cov_scaled,
        val_series=val_target_scaled,
        val_future_covariates=val_cov_scaled,
        verbose=True
    )

    historical_forecasts_scaled = model.historical_forecasts(
        series=val_target_scaled,
        future_covariates=val_cov_scaled,
        start=0.1,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=True
    )
    historical_forecasts = scaler_target.inverse_transform(historical_forecasts_scaled)
    mape_score = mape(val_target, historical_forecasts)
    metrics = {'mape': float(mape_score.item())}
    print(f"--- TFT (No Past Covariates) Validation MAPE: {mape_score:.2f}% ---")
    return model, scaler_target, scaler_cov, metrics

# --- Model Training Dispatcher ---
MODEL_TRAINERS: Dict[str, Callable] = {
    "LightGBM": train_lgbm_model,
    "TiDE": train_tide_model,
    "LSTM": train_lstm_model,
    "TFT": train_tft_model,
    "TFT (No Past Covariates)": train_tft_no_past_cov_model,
}

def train_model(
    model_type: str,
    data: pd.DataFrame,
    target_column: str = 'energy_kwh',
    input_chunk_length: int = 24 * 7,
    output_chunk_length: int = 24,
    n_epochs: int = 20 # Added n_epochs
) -> Tuple[ForecastingModel, Scaler, Scaler, Scaler, Dict[str, Any]]: # Updated return type for TFT
    """
    Dispatches to the correct model training function based on model_type.
    """
    trainer = MODEL_TRAINERS.get(model_type)
    if not trainer:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types are: {list(MODEL_TRAINERS.keys())}")
    
    # Call the trainer with appropriate arguments
    if model_type == "TFT":
        return trainer(
            data=data,
            target_column=target_column,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs
        )
    elif model_type == "TFT (No Past Covariates)":
        return trainer(
            data=data,
            target_column=target_column,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs
        )
    else:
        return trainer(
            data=data,
            target_column=target_column,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs
        )

# --- Anomaly Detection Service ---

from darts.ad import QuantileDetector

def fit_anomaly_detector(
    model: ForecastingModel,
    series: TimeSeries,
    past_covariates: TimeSeries = None,
    future_covariates: TimeSeries = None,
    scaler: Scaler = None,
    high_quantile: float = 0.98
) -> QuantileDetector:
    """
    Fits a QuantileDetector on the residuals of a model's historical forecasts.
    """
    print("--- Fitting Anomaly Detector ---")
    
    # Generate historical forecasts
    historical_forecasts_scaled = model.historical_forecasts(
        series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        start=0.1,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=True
    )
    
    # Inverse transform forecasts to get actual values
    if scaler:
        historical_forecasts = scaler.inverse_transform(historical_forecasts_scaled)
    else:
        historical_forecasts = historical_forecasts_scaled

    # Align the original series with the forecasts
    original_series_aligned = series.slice_intersect(historical_forecasts)
    historical_forecasts_aligned = historical_forecasts.slice_intersect(series)

    # Calculate absolute residuals
    residuals = (original_series_aligned - historical_forecasts_aligned).map(np.abs)
    
    # Fit the detector
    detector = QuantileDetector(high_quantile=high_quantile)
    detector.fit(residuals)
    
    
    return detector

def detect_anomalies(
    model: ForecastingModel,
    detector: QuantileDetector,
    series: TimeSeries,
    past_covariates: TimeSeries = None,
    future_covariates: TimeSeries = None,
    scaler: Scaler = None
) -> pd.DataFrame:
    """
    Detects anomalies in a new series using a pre-fitted model and detector.
    """
    print("--- Detecting Anomalies ---")
    
    # Generate historical forecasts for the new data
    historical_forecasts_scaled = model.historical_forecasts(
        series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        start=0.1,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=True
    )
    
    # Inverse transform forecasts
    if scaler:
        historical_forecasts = scaler.inverse_transform(historical_forecasts_scaled)
    else:
        historical_forecasts = historical_forecasts_scaled

    # Align series and forecasts
    original_series_aligned = series.slice_intersect(historical_forecasts)
    historical_forecasts_aligned = historical_forecasts.slice_intersect(series)
    
    # Calculate absolute residuals
    residuals = (original_series_aligned - historical_forecasts_aligned).map(np.abs)
    
    # Detect anomalies
    anomaly_scores = detector.detect(residuals)
    
    # Filter for actual anomalies (where score is 1)
    # Convert original_series_aligned to pandas Series for flexible indexing
    original_series_pd = original_series_aligned.pd_series()
    
    # Align anomaly_scores index with original_series_aligned before filtering
    aligned_anomaly_scores = anomaly_scores.pd_series().reindex(original_series_pd.index, fill_value=0)
    
    # Apply boolean mask using pandas indexing
    anomalies_pd = original_series_pd[aligned_anomaly_scores == 1]
    
    # Convert back to Darts TimeSeries, explicitly setting frequency
    anomalies = TimeSeries.from_series(anomalies_pd, freq='H', fill_missing_dates=True)
    
    print(f"--- Found {len(anomalies_pd)} anomalies. ---")
    
    # Return as a DataFrame directly, formatted for frontend
    anomalies_df = anomalies_pd.to_frame(name='value') # Rename the column to 'value'
    anomalies_df = anomalies_df.reset_index() # Convert index (timestamp) to a column
    anomalies_df = anomalies_df.rename(columns={'time': 'timestamp'}) # Rename the timestamp column
    return anomalies_df