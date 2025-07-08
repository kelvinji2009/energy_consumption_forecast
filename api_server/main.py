import pandas as pd
import numpy as np
import os
import joblib
from fastapi import FastAPI, HTTPException, Request, Depends, status, UploadFile, File, Form, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import io
import boto3 # Import boto3
from typing import Any

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
from darts.models import LightGBMModel, TiDEModel, RNNModel, TFTModel # Import all models
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from core.training_service import detect_anomalies as run_anomaly_detection

print("--- Script main.py starting to execute (Ultimate Compatibility Version) ---")

# --- S3/MinIO Configuration ---
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = "models"

def get_s3_client():
    """Creates a boto3 S3 client."""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY
    )

# --- Pydantic Models ---
class TimeSeriesDataPoint(BaseModel):
    timestamp: datetime
    value: float
    temp: Optional[float] = None
    production: Optional[float] = None
    humidity: Optional[float] = None # Added humidity

class PredictionRequest(BaseModel):
    historical_data: List[TimeSeriesDataPoint]
    forecast_horizon: int = Field(..., gt=0, description="Forecast horizon must be greater than 0")

class ForecastDataPoint(BaseModel):
    timestamp: datetime
    predicted_value: float

class PredictionResponse(BaseModel):
    asset_id: str
    forecast_data: List[ForecastDataPoint]
    historical_data: Optional[List[TimeSeriesDataPoint]] = None

class AnomalyDetectionRequest(BaseModel):
    data_stream: List[TimeSeriesDataPoint]

class AnomalyDataPoint(BaseModel):
    timestamp: datetime
    value: float

class AnomalyDetectionResponse(BaseModel):
    asset_id: str
    anomalies: List[AnomalyDataPoint]
    historical_data: Optional[List[TimeSeriesDataPoint]] = None


class ModelInfoResponse(BaseModel):
    id: int
    model_type: str
    model_version: str
    status: str
    description: Optional[str] = None
    metrics: Optional[dict] = None
    detector_path: Optional[str] = None # Use path for boolean check on frontend
    created_at: datetime

    class Config:
        from_attributes = True


# --- Lifespan, App, Security setup ---
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

from api_server.dependencies import verify_api_key, get_db

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

# --- New Helper to load artifacts from S3 ---
def _load_artifact_from_s3(s3_path: str, s3_client) -> Any:
    if not s3_path:
        return None
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_path)
        return joblib.load(io.BytesIO(response['Body'].read()))
    except Exception as e:
        print(f"Error loading artifact from S3 '{s3_path}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model artifact from S3: {s3_path}")

@app.get("/ping", summary="Check service status")
def ping(): return {"status": "ok"}

@app.get("/admin/assets/{asset_id}/models", response_model=List[ModelInfoResponse], dependencies=[Depends(verify_api_key)])
async def get_asset_models(asset_id: str, db: Session = Depends(get_db)):
    """获取指定资产的所有可用模型信息。"""
    stmt = select(Model).where(Model.asset_id == asset_id).order_by(Model.created_at.desc())
    models = db.execute(stmt).scalars().all()
    return models

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

# @app.post("/assets/{asset_id}/detect_anomalies", response_model=AnomalyDetectionResponse, dependencies=[Depends(verify_api_key)])
# def detect_anomalies(asset_id: str, request: AnomalyDetectionRequest, http_request: Request):
#     print(f"\n--- Received anomaly detection request for asset: {asset_id} ---")
#     model_cache_entry = http_request.app.state.model_cache.get(asset_id)
#     if not (model_cache_entry and model_cache_entry.get('model') and model_cache_entry.get('detector')):
#         raise HTTPException(status_code=503, detail=f"Model or detector not available for asset '{asset_id}'.")
#         
#     model, detector, scaler = model_cache_entry['model'], model_cache_entry['detector'], model_cache_entry.get('scaler')
#     
#     try:
#         series_to_detect = _create_timeseries_from_request(request.data_stream)
#         input_chunk_length = len(model.lags['target'])
#         
#         if len(series_to_detect) <= input_chunk_length:
#              raise ValueError(f"Data stream length must be > model's required input length ({input_chunk_length}).")
#
#         series_scaled = scaler.transform(series_to_detect) if scaler else series_to_detect
#         
#         full_covariates = datetime_attribute_timeseries(series_scaled, "hour", True).stack(datetime_attribute_timeseries(series_scaled, "day_of_week", True)).astype(np.float32)
#         
#         historical_forecasts_scaled = model.historical_forecasts(series=series_scaled, future_covariates=full_covariates, start=input_chunk_length, forecast_horizon=1, stride=1, retrain=False, verbose=False)
#         
#         actual_values_aligned = series_scaled.slice_intersect(historical_forecasts_scaled)
#         residuals_scaled = actual_values_aligned - historical_forecasts_scaled
#         residuals_ts = TimeSeries.from_series(pd.Series(np.abs(residuals_scaled.values().flatten()), index=residuals_scaled.time_index), freq='h')
#         
#         anomalies_ts = detector.detect(residuals_ts)
#         print("Anomaly detection successful.")
#         
#         # Bulletproof Series creation and filtering
#         anomalies_series = pd.Series(anomalies_ts.values().flatten(), index=anomalies_ts.time_index)
#         anomaly_timestamps = anomalies_series[anomalies_series == 1].index
#         
#         anomalies_list = []
#         if not anomaly_timestamps.empty:
#             historical_forecasts_unscaled = scaler.inverse_transform(historical_forecasts_scaled) if scaler else historical_forecasts_scaled
#             
#             original_values_series = pd.Series(series_to_detect.values().flatten(), index=series_to_detect.time_index)
#             predicted_values_series = pd.Series(historical_forecasts_unscaled.values().flatten(), index=historical_forecasts_unscaled.time_index)
#
#             for ts in anomaly_timestamps:
#                 original_value = original_values_series.get(ts)
#                 predicted_value = predicted_values_series.get(ts)
#                 if original_value is not None and predicted_value is not None:
#                     reason = (f"Value ({original_value:.2f}) is significantly higher than predicted ({predicted_value:.2f})." if original_value > predicted_value else f"Value ({original_value:.2f}) is significantly lower than predicted ({predicted_value:.2f}).")
#                     anomalies_list.append(AnomalyDataPoint(timestamp=ts, value=original_value, reason=reason))
#         
#         print(f"Found {len(anomalies_list)} anomalies.")
#         return AnomalyDetectionResponse(asset_id=asset_id, anomalies=anomalies_list)
#         
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {e}")

@app.post("/assets/{asset_id}/predict_from_csv", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict_from_csv(asset_id: str, http_request: Request, file: UploadFile = File(...), forecast_horizon: int = Form(168), model_id: int = Form(...), db: Session = Depends(get_db)):
    print(f"""
    --- Received prediction request for asset: {asset_id} from CSV with horizon: {forecast_horizon} using model ID: {model_id} ---""")
    
    # 0. Fetch model metadata from DB first
    model_record = db.query(Model).filter(Model.id == model_id, Model.asset_id == asset_id).first()
    if not model_record:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found for asset {asset_id}.")

    # 1. Load Model from Cache or Database
    model_cache = http_request.app.state.model_cache
    model_key = f"model_{model_id}"

    if model_key not in model_cache:
        # Fetch model paths from DB
        # model_record = db.query(Model).filter(Model.id == model_id, Model.asset_id == asset_id).first()
        # if not model_record:
        #     raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found for asset {asset_id}.")

        try:
            s3_client = get_s3_client()
            model_obj = _load_artifact_from_s3(model_record.model_path, s3_client)
            
            # --- BUG FIX: Check if model loaded successfully ---
            if model_obj is None:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Model Consistency Error: Model file not found in S3 for model ID {model_id} at path '{model_record.model_path}'. Please retrain the model."
                )

            scaler_obj = _load_artifact_from_s3(model_record.scaler_path, s3_client)
            scaler_cov_obj = _load_artifact_from_s3(model_record.scaler_cov_path, s3_client)
            scaler_past_cov_obj = _load_artifact_from_s3(model_record.scaler_past_cov_path, s3_client) # Load past_cov_scaler
            detector_obj = _load_artifact_from_s3(model_record.detector_path, s3_client) # Load detector
            
            model_cache[model_key] = {
                'model': model_obj,
                'scaler': scaler_obj,
                'scaler_cov': scaler_cov_obj,
                'scaler_past_cov': scaler_past_cov_obj, # Store past_cov_scaler
                'detector': detector_obj # Store detector
            }
            print(f"[Cache] Model ID {model_id} loaded into cache.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model artifacts for ID {model_id}: {e}")
    
    model_artifacts = model_cache[model_key]
    model = model_artifacts['model']
    scaler = model_artifacts.get('scaler')
    scaler_cov = model_artifacts.get('scaler_cov')
    scaler_past_cov = model_artifacts.get('scaler_past_cov') # Get past_cov_scaler

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
                production=row.get('production'),
                humidity=row.get('humidity') # Added humidity
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

        # --- Handle past covariates for TFT models ---
        past_covs = None
        if model_record.model_type == "TFT": # Check model type from DB record
            # Extract past covariates from historical_data
            past_cov_df = pd.DataFrame([d.model_dump() for d in historical_data])
            past_cov_df['timestamp'] = pd.to_datetime(past_cov_df['timestamp'])
            past_covs = TimeSeries.from_dataframe(past_cov_df, "timestamp", ['production', 'temp', 'humidity'], freq='h').astype(np.float32)
            past_covs = scaler_past_cov.transform(past_covs) if scaler_past_cov else past_covs

        forecast_scaled = model.predict(n=forecast_horizon, series=series_scaled, future_covariates=future_covs, past_covariates=past_covs)
        
        forecast = scaler.inverse_transform(forecast_scaled) if scaler else forecast_scaled
        
        print("Prediction from CSV successful.")
        
        forecast_df = pd.DataFrame(data=forecast.values(), index=forecast.time_index, columns=['predicted_value'])
        
        forecast_data = [ForecastDataPoint(timestamp=ts, predicted_value=row['predicted_value']) for ts, row in forecast_df.iterrows()]
        
        return PredictionResponse(asset_id=asset_id, forecast_data=forecast_data, historical_data=historical_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/assets/{asset_id}/detect_anomalies_from_csv", response_model=AnomalyDetectionResponse, dependencies=[Depends(verify_api_key)])
async def detect_anomalies_from_csv(asset_id: str, http_request: Request, file: UploadFile = File(...), model_id: int = Form(...), db: Session = Depends(get_db)):
    print(f'\n--- Received anomaly detection request for asset: {asset_id} from CSV using model ID: {model_id} ---')
    
    model_record = db.query(Model).filter(Model.id == model_id, Model.asset_id == asset_id).first()
    if not model_record:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found for asset {asset_id}.")
    if not model_record.detector_path:
        raise HTTPException(status_code=400, detail=f"Model with ID {model_id} does not have an anomaly detector.")

    model_cache = http_request.app.state.model_cache
    model_key = f"model_{model_id}"

    if model_key not in model_cache:
        try:
            s3_client = get_s3_client()
            model_obj = _load_artifact_from_s3(model_record.model_path, s3_client)

            # --- BUG FIX: Check if model loaded successfully ---
            if model_obj is None:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Model Consistency Error: Model file not found in S3 for model ID {model_id} at path '{model_record.model_path}'. Please retrain the model."
                )
            
            scaler_obj = _load_artifact_from_s3(model_record.scaler_path, s3_client)
            scaler_cov_obj = _load_artifact_from_s3(model_record.scaler_cov_path, s3_client)
            scaler_past_cov_obj = _load_artifact_from_s3(model_record.scaler_past_cov_path, s3_client)
            detector_obj = _load_artifact_from_s3(model_record.detector_path, s3_client)
            
            model_cache[model_key] = {
                'model': model_obj,
                'scaler': scaler_obj,
                'scaler_cov': scaler_cov_obj,
                'scaler_past_cov': scaler_past_cov_obj,
                'detector': detector_obj
            }
            print(f"[Cache] Model ID {model_id} and detector loaded into cache.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model artifacts for ID {model_id}: {e}")
    
    model_artifacts = model_cache[model_key]
    model = model_artifacts['model']
    detector = model_artifacts.get('detector')
    scaler = model_artifacts.get('scaler')
    scaler_cov = model_artifacts.get('scaler_cov')
    scaler_past_cov = model_artifacts.get('scaler_past_cov')

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.rename(columns={'energy_kwh': 'value'}, inplace=True)
        if 'timestamp' not in df.columns or 'value' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must have 'timestamp' and 'value' columns.")
        
        # Convert timestamp column to datetime objects before creating TimeSeriesDataPoint
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        historical_data = [TimeSeriesDataPoint(**row) for index, row in df.iterrows()]

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {e}")

    try:
        series = _create_timeseries_from_request(historical_data)
        series_scaled = scaler.transform(series) if scaler else series

        past_covs = None
        if model_record.model_type == "TFT":
            past_cov_df = pd.DataFrame([d.model_dump() for d in historical_data])
            past_cov_df['timestamp'] = pd.to_datetime(past_cov_df['timestamp'])
            past_covs = TimeSeries.from_dataframe(past_cov_df, "timestamp", ['production_units', 'temperature_celsius', 'humidity_percent'], freq='h').astype(np.float32)
            past_covs = scaler_past_cov.transform(past_covs) if scaler_past_cov else past_covs

        future_covs = None
        if model_record.model_type in ["TFT", "TiDE", "LSTM", "LightGBM"]:
            future_covs_raw = datetime_attribute_timeseries(series_scaled, attribute="hour", one_hot=True).stack(
                datetime_attribute_timeseries(series_scaled, attribute="day_of_week", one_hot=True)
            ).astype(np.float32)
            future_covs = scaler_cov.transform(future_covs_raw) if scaler_cov else future_covs_raw

        anomalies_df = run_anomaly_detection(
            model=model,
            detector=detector,
            series=series_scaled,
            past_covariates=past_covs,
            future_covariates=future_covs,
            scaler=scaler
        )
        
        print("--- Anomalies DataFrame (before sending to frontend) ---")
        print(anomalies_df.head())
        print(anomalies_df['timestamp'].dtype)

        anomalies_data = [AnomalyDataPoint(timestamp=row['timestamp'], value=row['value']) for index, row in anomalies_df.iterrows()]
        
        return AnomalyDetectionResponse(asset_id=asset_id, anomalies=anomalies_data, historical_data=historical_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {e}")

@app.post("/assets/{asset_id}/predict_from_s3", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict_from_s3(asset_id: str, http_request: Request, s3_data_path: str = Query(...), forecast_horizon: int = Query(168), model_id: int = Query(...), db: Session = Depends(get_db)):
    print(f"\n--- Received prediction request for asset: {asset_id} from S3 path: {s3_data_path} with horizon: {forecast_horizon} using model ID: {model_id} ---")

    # 0. Fetch model metadata from DB first
    model_record = db.query(Model).filter(Model.id == model_id, Model.asset_id == asset_id).first()
    if not model_record:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found for asset {asset_id}.")

    # 1. Load Model from Cache or Database
    model_cache = http_request.app.state.model_cache
    model_key = f"model_{model_id}"

    if model_key not in model_cache:
        try:
            s3_client = get_s3_client()
            model_obj = _load_artifact_from_s3(model_record.model_path, s3_client)

            # --- BUG FIX: Check if model loaded successfully ---
            if model_obj is None:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Model Consistency Error: Model file not found in S3 for model ID {model_id} at path '{model_record.model_path}'. Please retrain the model."
                )

            scaler_obj = _load_artifact_from_s3(model_record.scaler_path, s3_client)
            scaler_cov_obj = _load_artifact_from_s3(model_record.scaler_cov_path, s3_client)
            scaler_past_cov_obj = _load_artifact_from_s3(model_record.scaler_past_cov_path, s3_client)
            detector_obj = _load_artifact_from_s3(model_record.detector_path, s3_client) # Load detector
            
            model_cache[model_key] = {
                'model': model_obj,
                'scaler': scaler_obj,
                'scaler_cov': scaler_cov_obj,
                'scaler_past_cov': scaler_past_cov_obj,
                'detector': detector_obj # Store detector
            }
            print(f"[Cache] Model ID {model_id} loaded into cache.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model artifacts for ID {model_id}: {e}")
    
    model_artifacts = model_cache[model_key]
    model = model_artifacts['model']
    scaler = model_artifacts.get('scaler')
    scaler_cov = model_artifacts.get('scaler_cov')
    scaler_past_cov = model_artifacts.get('scaler_past_cov')

    # 2. Read and Parse CSV from S3
    try:
        s3_client = get_s3_client()
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_data_path)
        contents = response['Body'].read()
        df = pd.read_csv(io.BytesIO(contents))
        
        df.rename(columns={'energy_kwh': 'value'}, inplace=True)
        
        if 'timestamp' not in df.columns or 'value' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must have 'timestamp' and 'value' columns.")

        historical_hours = len(df)
        max_horizon = historical_hours // 4
        if forecast_horizon > max_horizon:
            raise HTTPException(
                status_code=400, 
                detail=f"Forecast horizon ({forecast_horizon} hours) is too large for the provided historical data ({historical_hours} hours). Maximum allowed horizon is {max_horizon} hours."
            )
            
        historical_data = [
            TimeSeriesDataPoint(
                timestamp=row['timestamp'], 
                value=row['value'],
                temp=row.get('temp'),
                production=row.get('production'),
                humidity=row.get('humidity')
            ) for index, row in df.iterrows()
        ]

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Error processing CSV file from S3: {e}")

    # 3. Reuse existing prediction logic
    try:
        series = _create_timeseries_from_request(historical_data)
        series_scaled = scaler.transform(series) if scaler else series
        
        future_covs_raw = _build_future_covariates(series_scaled, forecast_horizon, model)
        future_covs = scaler_cov.transform(future_covs_raw) if scaler_cov else future_covs_raw

        past_covs = None
        if model_record.model_type == "TFT": # Check model type from DB record
            past_cov_df = pd.DataFrame([d.model_dump() for d in historical_data])
            past_cov_df['timestamp'] = pd.to_datetime(past_cov_df['timestamp'])
            past_covs = TimeSeries.from_dataframe(past_cov_df, "timestamp", ['production', 'temp', 'humidity'], freq='h').astype(np.float32)
            past_covs = scaler_past_cov.transform(past_covs) if scaler_past_cov else past_covs

        forecast_scaled = model.predict(n=forecast_horizon, series=series_scaled, future_covariates=future_covs, past_covariates=past_covs)
        
        forecast = scaler.inverse_transform(forecast_scaled) if scaler else forecast_scaled
        
        print("Prediction from S3 successful.")
        
        forecast_df = pd.DataFrame(data=forecast.values(), index=forecast.time_index, columns=['predicted_value'])
        
        forecast_data = [ForecastDataPoint(timestamp=ts, predicted_value=row['predicted_value']) for ts, row in forecast_df.iterrows()]
        
        return PredictionResponse(asset_id=asset_id, forecast_data=forecast_data, historical_data=historical_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/assets/{asset_id}/detect_anomalies_from_s3", response_model=AnomalyDetectionResponse, dependencies=[Depends(verify_api_key)])
async def detect_anomalies_from_s3(asset_id: str, http_request: Request, s3_data_path: str = Query(...), model_id: int = Query(...), db: Session = Depends(get_db)):
    print(f'\n--- Received anomaly detection request for asset: {asset_id} from S3 path: {s3_data_path} using model ID: {model_id} ---')

    model_record = db.query(Model).filter(Model.id == model_id, Model.asset_id == asset_id).first()
    if not model_record:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found for asset {asset_id}.")
    if not model_record.detector_path:
        raise HTTPException(status_code=400, detail=f"Model with ID {model_id} does not have an anomaly detector.")

    model_cache = http_request.app.state.model_cache
    model_key = f"model_{model_id}"

    if model_key not in model_cache:
        try:
            s3_client = get_s3_client()
            model_obj = _load_artifact_from_s3(model_record.model_path, s3_client)

            # --- BUG FIX: Check if model loaded successfully ---
            if model_obj is None:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Model Consistency Error: Model file not found in S3 for model ID {model_id} at path '{model_record.model_path}'. Please retrain the model."
                )

            scaler_obj = _load_artifact_from_s3(model_record.scaler_path, s3_client)
            scaler_cov_obj = _load_artifact_from_s3(model_record.scaler_cov_path, s3_client)
            scaler_past_cov_obj = _load_artifact_from_s3(model_record.scaler_past_cov_path, s3_client)
            detector_obj = _load_artifact_from_s3(model_record.detector_path, s3_client)
            
            model_cache[model_key] = {
                'model': model_obj,
                'scaler': scaler_obj,
                'scaler_cov': scaler_cov_obj,
                'scaler_past_cov': scaler_past_cov_obj,
                'detector': detector_obj
            }
            print(f"[Cache] Model ID {model_id} and detector loaded into cache.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model artifacts for ID {model_id}: {e}")
    
    model_artifacts = model_cache[model_key]
    model = model_artifacts['model']
    detector = model_artifacts.get('detector')
    scaler = model_artifacts.get('scaler')
    scaler_cov = model_artifacts.get('scaler_cov')
    scaler_past_cov = model_artifacts.get('scaler_past_cov')

    try:
        s3_client = get_s3_client()
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_data_path)
        contents = response['Body'].read()
        df = pd.read_csv(io.BytesIO(contents))
        df.rename(columns={'energy_kwh': 'value'}, inplace=True)
        if 'timestamp' not in df.columns or 'value' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must have 'timestamp' and 'value' columns.")
        
        # Convert timestamp column to datetime objects before creating TimeSeriesDataPoint
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        historical_data = [TimeSeriesDataPoint(**row) for index, row in df.iterrows()]

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV file from S3: {e}")

    try:
        series = _create_timeseries_from_request(historical_data)
        series_scaled = scaler.transform(series) if scaler else series

        past_covs = None
        if model_record.model_type == "TFT":
            past_cov_df = pd.DataFrame([d.model_dump() for d in historical_data])
            past_cov_df['timestamp'] = pd.to_datetime(past_cov_df['timestamp'])
            past_covs = TimeSeries.from_dataframe(past_cov_df, "timestamp", ['production_units', 'temperature_celsius', 'humidity_percent'], freq='h').astype(np.float32)
            past_covs = scaler_past_cov.transform(past_covs) if scaler_past_cov else past_covs

        future_covs = None
        if model_record.model_type in ["TFT", "TiDE", "LSTM", "LightGBM"]:
            future_covs_raw = datetime_attribute_timeseries(series_scaled, attribute="hour", one_hot=True).stack(
                datetime_attribute_timeseries(series_scaled, attribute="day_of_week", one_hot=True)
            ).astype(np.float32)
            future_covs = scaler_cov.transform(future_covs_raw) if scaler_cov else future_covs_raw

        anomalies_df = run_anomaly_detection(
            model=model,
            detector=detector,
            series=series_scaled,
            past_covariates=past_covs,
            future_covariates=future_covs,
            scaler=scaler
        )
        
        anomalies_data = [AnomalyDataPoint(timestamp=row['timestamp'], value=row['value']) for index, row in anomalies_df.iterrows()]
        
        return AnomalyDetectionResponse(asset_id=asset_id, anomalies=anomalies_data, historical_data=historical_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {e}")

print("--- Script main.py finished execution ---")