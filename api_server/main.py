import pandas as pd
import numpy as np
import os
import joblib
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from sqlalchemy import select
from database.database import engine, models_table, api_keys_table

from api_server.admin_api import router as admin_router

# 导入Darts相关的库
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import LightGBMModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.ad.detectors import QuantileDetector

print("--- Script main.py starting to execute ---")

# --- Pydantic Models for Data Validation ---

class TimeSeriesDataPoint(BaseModel):
    timestamp: datetime
    value: float
    temp: Optional[float] = None
    production: Optional[float] = None

class PredictionRequest(BaseModel):
    historical_data: List[TimeSeriesDataPoint]
    forecast_horizon: int = Field(..., gt=0, description="Forecast horizon must be greater than 0")

class ForecastDataPoint(BaseModel):
    timestamp: datetime
    predicted_value: float

class PredictionResponse(BaseModel):
    asset_id: str
    forecast_data: List[ForecastDataPoint]

class AnomalyDetectionRequest(BaseModel):
    data_stream: List[TimeSeriesDataPoint]

class AnomalyDataPoint(BaseModel):
    timestamp: datetime
    value: float
    reason: str

class AnomalyDetectionResponse(BaseModel):
    asset_id: str
    anomalies: List[AnomalyDataPoint]

# --- FastAPI Lifespan for Startup and Shutdown ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Lifespan] Startup event triggered.")
    app.state.model_cache = {}
    try:
        print("[Lifespan] Attempting to connect to database and load active models...")
        with engine.connect() as connection:
            stmt = select(models_table).where(models_table.c.is_active == True)
            result = connection.execute(stmt).fetchall()
            if not result:
                print("[Lifespan] No active models found in the database.")
            for row in result:
                asset_id = row.asset_id
                model_relative_path = row.path
                model_type = row.model_type
                model_path_abs = os.path.join(PROJECT_ROOT, model_relative_path)
                model_obj, detector_obj, scaler_obj = None, None, None
                try:
                    print(f"[Lifespan] Loading {model_type} model for asset '{asset_id}' from: {model_path_abs}")
                    model_obj = joblib.load(model_path_abs)
                    print(f"[Lifespan] Model for asset '{asset_id}' loaded successfully.")
                    model_dir, model_filename = os.path.dirname(model_path_abs), os.path.basename(model_path_abs)
                    detector_path_abs = os.path.join(model_dir, model_filename.replace("model_", "detector_"))
                    if os.path.exists(detector_path_abs):
                        print(f"[Lifespan] Loading detector for asset '{asset_id}' from: {detector_path_abs}")
                        detector_obj = joblib.load(detector_path_abs)
                        print(f"[Lifespan] Detector for asset '{asset_id}' loaded successfully.")
                    else:
                        print(f"[Lifespan] No detector found for asset '{asset_id}' at {detector_path_abs}.")
                    scaler_path_abs = os.path.join(model_dir, model_filename.replace("model_", "scaler_"))
                    if os.path.exists(scaler_path_abs):
                        print(f"[Lifespan] Loading scaler for asset '{asset_id}' from: {scaler_path_abs}")
                        scaler_obj = joblib.load(scaler_path_abs)
                        print(f"[Lifespan] Scaler for asset '{asset_id}' loaded successfully.")
                    else:
                        print(f"[Lifespan] No scaler found for asset '{asset_id}' at {scaler_path_abs}.")
                    app.state.model_cache[asset_id] = {'model': model_obj, 'detector': detector_obj, 'scaler': scaler_obj}
                except Exception as e:
                    print(f"[Lifespan] ERROR: Failed to load model/detector/scaler for asset '{asset_id}'. Error: {e}")
    except Exception as e:
        print(f"[Lifespan] CRITICAL: Database connection or initial model loading failed. Error: {e}")
    yield
    print("[Lifespan] Shutdown event triggered.")
    app.state.model_cache.clear()
    print("[Lifespan] Model cache cleared.")

# --- FastAPI Application ---

print("Initializing FastAPI app...")
app = FastAPI(title="能耗预测与异常检测API", description="一个用于工业能耗预测和异常检测的API服务。", version="5.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(admin_router)
print("FastAPI app initialized.")

# --- Security Dependency ---
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = credentials.credentials
    with engine.connect() as connection:
        stmt = select(api_keys_table).where(api_keys_table.c.key_hash == api_key, api_keys_table.c.is_active == True)
        if not connection.execute(stmt).fetchone():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or inactive API Key", headers={"WWW-Authenticate": "Bearer"})
    return api_key

# --- Helper Functions ---

def _create_timeseries_from_request(data: List[TimeSeriesDataPoint]) -> TimeSeries:
    df = pd.DataFrame([d.model_dump() for d in data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return TimeSeries.from_dataframe(df, "timestamp", "value", freq='h')

def _build_future_covariates(series: TimeSeries, horizon: int, model) -> TimeSeries:
    required_cov_len = horizon + model.output_chunk_length
    future_index = pd.date_range(start=series.end_time() + series.freq, periods=required_cov_len, freq=series.freq)
    dummy_series = TimeSeries.from_times_and_values(future_index, np.zeros(required_cov_len))
    return datetime_attribute_timeseries(dummy_series, attribute="hour", one_hot=True).stack(
        datetime_attribute_timeseries(dummy_series, attribute="day_of_week", one_hot=True)
    ).astype(np.float32)

# --- API Endpoints ---

@app.get("/ping", summary="Check service status")
def ping():
    return {"status": "ok", "message": "Service is running."}

@app.post("/assets/{asset_id}/predict", response_model=PredictionResponse, summary="Execute energy consumption forecast", dependencies=[Depends(verify_api_key)])
def predict(asset_id: str, request: PredictionRequest, http_request: Request):
    print(f"\n--- Received prediction request for asset: {asset_id} ---")
    model_cache_entry = http_request.app.state.model_cache.get(asset_id)
    if not model_cache_entry or not model_cache_entry.get('model'):
        raise HTTPException(status_code=503, detail=f"No active model found for asset '{asset_id}'.")
    model, scaler = model_cache_entry['model'], model_cache_entry.get('scaler')
    print(f"Model for asset '{asset_id}' retrieved from cache.")
    try:
        series = _create_timeseries_from_request(request.historical_data)
        if scaler:
            series = scaler.transform(series)
            print("[Helper] Input series scaled for prediction.")
        future_covs = _build_future_covariates(series, request.forecast_horizon, model)
        print(f"Executing prediction for horizon: {request.forecast_horizon}")
        forecast = model.predict(n=request.forecast_horizon, series=series, future_covariates=future_covs)
        print("Prediction successful.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
        
    forecast_df = pd.DataFrame(data=forecast.values(), index=forecast.time_index, columns=['predicted_value'])

    if scaler:
        inverse_transformed_ts = scaler.inverse_transform(forecast)
        forecast_df['predicted_value'] = inverse_transformed_ts.values()
        print("[Helper] Forecast results inverse-scaled.")
        
    forecast_data = [ForecastDataPoint(timestamp=ts, predicted_value=row['predicted_value']) for ts, row in forecast_df.iterrows()]
    return PredictionResponse(asset_id=asset_id, forecast_data=forecast_data)

@app.post("/assets/{asset_id}/detect_anomalies", response_model=AnomalyDetectionResponse, summary="Execute anomaly detection", dependencies=[Depends(verify_api_key)])
def detect_anomalies(asset_id: str, request: AnomalyDetectionRequest, http_request: Request):
    print(f"\n--- Received anomaly detection request for asset: {asset_id} ---")
    model_cache_entry = http_request.app.state.model_cache.get(asset_id)
    if not model_cache_entry or not model_cache_entry.get('model'):
        raise HTTPException(status_code=503, detail=f"No active model found for asset '{asset_id}'.")
    if not model_cache_entry.get('detector'):
        raise HTTPException(status_code=503, detail=f"No active anomaly detector found for asset '{asset_id}'.")
        
    model, detector, scaler = model_cache_entry['model'], model_cache_entry['detector'], model_cache_entry.get('scaler')
    print(f"Model and detector for asset '{asset_id}' retrieved from cache.")
    
    try:
        series_to_detect = _create_timeseries_from_request(request.data_stream)
        input_chunk_length = len(model.lags['target'])
        
        if len(series_to_detect) <= input_chunk_length:
             raise ValueError(f"Data stream length ({len(series_to_detect)}) must be greater than model's required input length ({input_chunk_length}).")

        series_to_detect_scaled = scaler.transform(series_to_detect) if scaler else series_to_detect
        print("[Helper] Input series scaled for anomaly detection." if scaler else "[Helper] Input series not scaled.")
        
        full_covariates = datetime_attribute_timeseries(
            series_to_detect_scaled, "hour", True).stack(datetime_attribute_timeseries(series_to_detect_scaled, "day_of_week", True)
        ).astype(np.float32)

        historical_forecasts_scaled = model.historical_forecasts(
            series=series_to_detect_scaled, future_covariates=full_covariates, start=input_chunk_length,
            forecast_horizon=1, stride=1, retrain=False, verbose=False)
        
        actual_values_aligned = series_to_detect_scaled.slice_intersect(historical_forecasts_scaled)
        residuals_scaled = actual_values_aligned - historical_forecasts_scaled
        residuals_ts = TimeSeries.from_series(pd.Series(np.abs(residuals_scaled.values().flatten()), index=residuals_scaled.time_index), freq='h')
        
        anomalies_ts = detector.detect(residuals_ts)
        print("Anomaly detection successful.")
        
        anomalies_list = []

        # --- THE ULTIMATE FIX: Manually construct Pandas Series from .values() and .time_index ---
        # This is guaranteed to work on any version of Darts.
        anomalies_series = pd.Series(anomalies_ts.values().flatten(), index=anomalies_ts.time_index)
        anomaly_timestamps = anomalies_series[anomalies_series == 1].index

        if not anomaly_timestamps.empty:
            historical_forecasts_unscaled = scaler.inverse_transform(historical_forecasts_scaled) if scaler else historical_forecasts_scaled
            
            # Manually construct all other series for maximum robustness
            original_values_series = pd.Series(series_to_detect.values().flatten(), index=series_to_detect.time_index)
            predicted_values_series = pd.Series(historical_forecasts_unscaled.values().flatten(), index=historical_forecasts_unscaled.time_index)

            for ts in anomaly_timestamps:
                original_value = original_values_series.get(ts)
                predicted_value = predicted_values_series.get(ts)

                if original_value is not None and predicted_value is not None:
                    reason = (f"Value ({original_value:.2f}) is significantly higher than predicted ({predicted_value:.2f})."
                              if original_value > predicted_value else
                              f"Value ({original_value:.2f}) is significantly lower than predicted ({predicted_value:.2f}).")
                    anomalies_list.append(AnomalyDataPoint(timestamp=ts, value=original_value, reason=reason))
        
        print(f"Found {len(anomalies_list)} anomalies.")
        
    except Exception as e:
        import traceback
        print(f"[Error] Anomaly detection failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {e}")
        
    return AnomalyDetectionResponse(asset_id=asset_id, anomalies=anomalies_list)

print("--- Script main.py finished execution ---")