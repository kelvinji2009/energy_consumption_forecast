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
from sqlalchemy.orm import Session # Import Session
from database.database import engine, Model, ApiKey, Asset # Import Model, ApiKey, Asset
from api_server.admin_api import router as admin_router

# Import Darts, this works for both 'darts' and 'u8darts'
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import LightGBMModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.ad.detectors import QuantileDetector

print("--- Script main.py starting to execute (Ultimate Compatibility Version) ---")

# --- Pydantic Models ---
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

class ModelInfo(BaseModel):
    id: int
    model_type: str
    model_version: str
    description: Optional[str] = None


# --- Lifespan, App, Security setup ---
# ... (Keep these sections as they were, they are correct) ...
async def lifespan(app: FastAPI):
    print("[Lifespan] Startup event triggered.")
    app.state.model_cache = {}
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

# Dependency to get DB session
def get_db():
    with Session(engine) as session:
        yield session

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    api_key = credentials.credentials
    stmt = select(ApiKey).where(ApiKey.key_hash == api_key, ApiKey.is_active == True)
    if not db.execute(stmt).scalar_one_or_none():
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

@app.get("/admin/assets/{asset_id}/models", response_model=List[ModelInfo], dependencies=[Depends(verify_api_key)])
async def get_asset_models(asset_id: str, db: Session = Depends(get_db)):
    """获取指定资产的所有可用模型信息。"""
    stmt = select(Model).where(Model.asset_id == asset_id).order_by(Model.model_type, Model.model_version.desc())
    models = db.execute(stmt).scalars().all()
    return [
        ModelInfo(
            id=m.id,
            model_type=m.model_type,
            model_version=m.model_version,
            description=m.description
        ) for m in models
    ]

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
async def predict_from_csv(asset_id: str, http_request: Request, file: UploadFile = File(...), forecast_horizon: int = Form(168), model_id: int = Form(...), db: Session = Depends(get_db)):
    print(f"\n--- Received prediction request for asset: {asset_id} from CSV with horizon: {forecast_horizon} using model ID: {model_id} ---")
    
    # 1. Load Model from Cache or Database
    model_cache = http_request.app.state.model_cache
    model_key = f"model_{model_id}"

    if model_key not in model_cache:
        # Fetch model paths from DB
        model_record = db.query(Model).filter(Model.id == model_id, Model.asset_id == asset_id).first()
        if not model_record:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found for asset {asset_id}.")

        try:
            model_obj = joblib.load(os.path.join(PROJECT_ROOT, model_record.model_path))
            scaler_obj = joblib.load(os.path.join(PROJECT_ROOT, model_record.scaler_path)) if model_record.scaler_path else None
            scaler_cov_obj = joblib.load(os.path.join(PROJECT_ROOT, model_record.scaler_cov_path)) if model_record.scaler_cov_path else None
            
            model_cache[model_key] = {
                'model': model_obj,
                'scaler': scaler_obj,
                'scaler_cov': scaler_cov_obj
            }
            print(f"[Cache] Model ID {model_id} loaded into cache.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model artifacts for ID {model_id}: {e}")
    
    model_artifacts = model_cache[model_key]
    model = model_artifacts['model']
    scaler = model_artifacts.get('scaler')
    scaler_cov = model_artifacts.get('scaler_cov')

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