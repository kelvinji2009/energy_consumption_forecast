import pandas as pd
import numpy as np
import os
import joblib
from fastapi import FastAPI, HTTPException, Request, Depends, status, UploadFile, File, Form

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import io

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from sqlalchemy import select
from database.database import engine, models_table, api_keys_table
from api_server.admin_api import router as admin_router

# Import Darts, this works for both 'darts' and 'u8darts'
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import LightGBMModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.ad.detectors import QuantileDetector

print("--- Script main.py starting to execute (Ultimate Compatibility Version) ---")

# --- Pydantic Models ---
# ... (Keep all your Pydantic models as they were) ...
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


# --- Lifespan, App, Security setup ---
# ... (Keep these sections as they were, they are correct) ...
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
                asset_id, model_relative_path, model_type = row.asset_id, row.path, row.model_type
                model_path_abs = os.path.join(PROJECT_ROOT, model_relative_path)
                try:
                    print(f"[Lifespan] Loading {model_type} model for asset '{asset_id}' from: {model_path_abs}")
                    model_obj = joblib.load(model_path_abs)
                    print(f"[Lifespan] Model for asset '{asset_id}' loaded successfully.")
                    
                    model_dir, model_filename = os.path.dirname(model_path_abs), os.path.basename(model_path_abs)
                    
                    # [FIX] Robustly find scaler and detector paths
                    detector_filename = model_filename.replace("model", "detector", 1)
                    scaler_filename = model_filename.replace("model", "scaler", 1)
                    
                    detector_path_abs = os.path.join(model_dir, detector_filename)
                    scaler_path_abs = os.path.join(model_dir, scaler_filename)
                    scaler_cov_path_abs = os.path.join(model_dir, model_filename.replace("model", "scaler_cov", 1))

                    detector_obj = joblib.load(detector_path_abs) if os.path.exists(detector_path_abs) else None
                    if detector_obj: print(f"[Lifespan] Detector for asset '{asset_id}' loaded successfully from {detector_path_abs}.")
                    
                    scaler_obj = joblib.load(scaler_path_abs) if os.path.exists(scaler_path_abs) else None
                    if scaler_obj: print(f"[Lifespan] Scaler for asset '{asset_id}' loaded successfully from {scaler_path_abs}.")

                    scaler_cov_obj = joblib.load(scaler_cov_path_abs) if os.path.exists(scaler_cov_path_abs) else None
                    if scaler_cov_obj: print(f"[Lifespan] Covariate Scaler for asset '{asset_id}' loaded successfully from {scaler_cov_path_abs}.")

                    app.state.model_cache[asset_id] = {'model': model_obj, 'detector': detector_obj, 'scaler': scaler_obj, 'scaler_cov': scaler_cov_obj}
                except Exception as e:
                    print(f"[Lifespan] ERROR loading artifacts for asset '{asset_id}': {e}")
    except Exception as e:
        print(f"[Lifespan] CRITICAL: Database connection or initial model loading failed. Error: {e}")
    yield
    print("[Lifespan] Shutdown event triggered.")
    app.state.model_cache.clear()

app = FastAPI(title="能耗预测与异常检测API", version="7.0.0-final", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"], # Add the dev server origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(admin_router)
security = HTTPBearer()
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = credentials.credentials
    with engine.connect() as connection:
        stmt = select(api_keys_table).where(api_keys_table.c.key_hash == api_key, api_keys_table.c.is_active == True)
        if not connection.execute(stmt).fetchone():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or inactive API Key")
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
    return datetime_attribute_timeseries(dummy_series, "hour", True).stack(
        datetime_attribute_timeseries(dummy_series, "day_of_week", True)
    ).astype(np.float32)

@app.get("/ping", summary="Check service status")
def ping(): return {"status": "ok"}

# --- API Endpoints with Bulletproof Code ---

@app.post("/assets/{asset_id}/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def predict(asset_id: str, request: PredictionRequest, http_request: Request):
    print(f"\n--- Received prediction request for asset: {asset_id} ---")
    model_cache_entry = http_request.app.state.model_cache.get(asset_id)
    if not (model_cache_entry and model_cache_entry.get('model')):
        raise HTTPException(status_code=503, detail=f"No active model for asset '{asset_id}'.")
    
    model = model_cache_entry['model']
    scaler = model_cache_entry.get('scaler')
    scaler_cov = model_cache_entry.get('scaler_cov') # Get the covariate scaler

    try:
        series = _create_timeseries_from_request(request.historical_data)
        series_scaled = scaler.transform(series) if scaler else series
        
        # [FIX] Use the covariate scaler to transform the future covariates
        future_covs_raw = _build_future_covariates(series_scaled, request.forecast_horizon, model)
        future_covs = scaler_cov.transform(future_covs_raw) if scaler_cov else future_covs_raw

        forecast_scaled = model.predict(n=request.forecast_horizon, series=series_scaled, future_covariates=future_covs)
        
        forecast = scaler.inverse_transform(forecast_scaled) if scaler else forecast_scaled
        
        print("Prediction successful.")
        
        # Bulletproof DataFrame creation
        forecast_df = pd.DataFrame(data=forecast.values(), index=forecast.time_index, columns=['predicted_value'])
        
        forecast_data = [ForecastDataPoint(timestamp=ts, predicted_value=row['predicted_value']) for ts, row in forecast_df.iterrows()]
        return PredictionResponse(asset_id=asset_id, forecast_data=forecast_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/assets/{asset_id}/detect_anomalies", response_model=AnomalyDetectionResponse, dependencies=[Depends(verify_api_key)])
def detect_anomalies(asset_id: str, request: AnomalyDetectionRequest, http_request: Request):
    print(f"\n--- Received anomaly detection request for asset: {asset_id} ---")
    model_cache_entry = http_request.app.state.model_cache.get(asset_id)
    if not (model_cache_entry and model_cache_entry.get('model') and model_cache_entry.get('detector')):
        raise HTTPException(status_code=503, detail=f"Model or detector not available for asset '{asset_id}'.")
        
    model, detector, scaler = model_cache_entry['model'], model_cache_entry['detector'], model_cache_entry.get('scaler')
    
    try:
        series_to_detect = _create_timeseries_from_request(request.data_stream)
        input_chunk_length = len(model.lags['target'])
        
        if len(series_to_detect) <= input_chunk_length:
             raise ValueError(f"Data stream length must be > model's required input length ({input_chunk_length}).")

        series_scaled = scaler.transform(series_to_detect) if scaler else series_to_detect
        
        full_covariates = datetime_attribute_timeseries(series_scaled, "hour", True).stack(datetime_attribute_timeseries(series_scaled, "day_of_week", True)).astype(np.float32)
        
        historical_forecasts_scaled = model.historical_forecasts(series=series_scaled, future_covariates=full_covariates, start=input_chunk_length, forecast_horizon=1, stride=1, retrain=False, verbose=False)
        
        actual_values_aligned = series_scaled.slice_intersect(historical_forecasts_scaled)
        residuals_scaled = actual_values_aligned - historical_forecasts_scaled
        residuals_ts = TimeSeries.from_series(pd.Series(np.abs(residuals_scaled.values().flatten()), index=residuals_scaled.time_index), freq='h')
        
        anomalies_ts = detector.detect(residuals_ts)
        print("Anomaly detection successful.")
        
        # Bulletproof Series creation and filtering
        anomalies_series = pd.Series(anomalies_ts.values().flatten(), index=anomalies_ts.time_index)
        anomaly_timestamps = anomalies_series[anomalies_series == 1].index
        
        anomalies_list = []
        if not anomaly_timestamps.empty:
            historical_forecasts_unscaled = scaler.inverse_transform(historical_forecasts_scaled) if scaler else historical_forecasts_scaled
            
            original_values_series = pd.Series(series_to_detect.values().flatten(), index=series_to_detect.time_index)
            predicted_values_series = pd.Series(historical_forecasts_unscaled.values().flatten(), index=historical_forecasts_unscaled.time_index)

            for ts in anomaly_timestamps:
                original_value = original_values_series.get(ts)
                predicted_value = predicted_values_series.get(ts)
                if original_value is not None and predicted_value is not None:
                    reason = (f"Value ({original_value:.2f}) is significantly higher than predicted ({predicted_value:.2f})." if original_value > predicted_value else f"Value ({original_value:.2f}) is significantly lower than predicted ({predicted_value:.2f}).")
                    anomalies_list.append(AnomalyDataPoint(timestamp=ts, value=original_value, reason=reason))
        
        print(f"Found {len(anomalies_list)} anomalies.")
        return AnomalyDetectionResponse(asset_id=asset_id, anomalies=anomalies_list)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {e}")

@app.post("/assets/{asset_id}/predict_from_csv", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict_from_csv(asset_id: str, http_request: Request, file: UploadFile = File(...), forecast_horizon: int = Form(168)):
    print(f"\n--- Received prediction request for asset: {asset_id} from CSV with horizon: {forecast_horizon} ---")
    
    # 1. Get Model from Cache
    model_cache_entry = http_request.app.state.model_cache.get(asset_id)
    if not (model_cache_entry and model_cache_entry.get('model')):
        raise HTTPException(status_code=503, detail=f"No active model for asset '{asset_id}'.")
    
    model = model_cache_entry['model']
    scaler = model_cache_entry.get('scaler')
    scaler_cov = model_cache_entry.get('scaler_cov')

    # 2. Read and Parse CSV
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        df.rename(columns={'energy_kwh': 'value'}, inplace=True)
        
        # Basic validation
        if 'timestamp' not in df.columns or 'value' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must have 'timestamp' and 'value' columns.")

        # --- Horizon Validation ---
        historical_hours = len(df)
        max_horizon = historical_hours // 4
        if forecast_horizon > max_horizon:
            raise HTTPException(
                status_code=400, 
                detail=f"Forecast horizon ({forecast_horizon} hours) is too large for the provided historical data ({historical_hours} hours). Maximum allowed horizon is {max_horizon} hours."
            )
            
        # Convert to TimeSeriesDataPoint list
        historical_data = [
            TimeSeriesDataPoint(
                timestamp=row['timestamp'], 
                value=row['value'],
                temp=row.get('temp'),
                production=row.get('production')
            ) for index, row in df.iterrows()
        ]

    except Exception as e:
        # Catch the specific HTTPException and re-raise it
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {e}")

    # 3. Reuse existing prediction logic
    try:
        series = _create_timeseries_from_request(historical_data)
        series_scaled = scaler.transform(series) if scaler else series
        
        future_covs_raw = _build_future_covariates(series_scaled, forecast_horizon, model)
        future_covs = scaler_cov.transform(future_covs_raw) if scaler_cov else future_covs_raw

        forecast_scaled = model.predict(n=forecast_horizon, series=series_scaled, future_covariates=future_covs)
        
        forecast = scaler.inverse_transform(forecast_scaled) if scaler else forecast_scaled
        
        print("Prediction from CSV successful.")
        
        forecast_df = pd.DataFrame(data=forecast.values(), index=forecast.time_index, columns=['predicted_value'])
        
        forecast_data = [ForecastDataPoint(timestamp=ts, predicted_value=row['predicted_value']) for ts, row in forecast_df.iterrows()]
        
        return PredictionResponse(asset_id=asset_id, forecast_data=forecast_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

print("--- Script main.py finished execution ---")