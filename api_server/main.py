import pandas as pd
import numpy as np
import os
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# 延迟导入Darts
# from darts import TimeSeries
# from darts.utils.timeseries_generation import datetime_attribute_timeseries

print("--- Script main.py starting to execute ---")

# --- Configuration ---
MODEL_DIR = os.path.join("..", "demo", "models")
DEFAULT_MODEL_NAME = "lgbm_energy_model"
MODEL_PATH = os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME, "model.joblib")

print(f"Model path configured to: {os.path.abspath(MODEL_PATH)}")

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

# --- FastAPI Lifespan for Startup and Shutdown ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Lifespan] Startup event triggered.")
    app.state.model_cache = {}
    try:
        print(f"[Lifespan] Attempting to load model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        app.state.model_cache["default"] = model
        print("[Lifespan] Model loaded successfully and cached.")
    except Exception as e:
        print(f"[Lifespan] CRITICAL: Failed to load model during startup. Error: {e}")
    yield
    print("[Lifespan] Shutdown event triggered.")
    app.state.model_cache.clear()
    print("[Lifespan] Model cache cleared.")

# --- FastAPI Application ---

print("Initializing FastAPI app...")
app = FastAPI(
    title="能耗预测与异常检测API",
    description="一个用于工业能耗预测和异常检测的API服务。",
    version="1.9.0", # 版本号+1
    lifespan=lifespan
)
print("FastAPI app initialized.")

# --- Helper Functions ---

def _create_timeseries_from_request(data: List[TimeSeriesDataPoint]) -> 'TimeSeries':
    from darts import TimeSeries
    print("[Helper] Creating TimeSeries from request data.")
    timestamps = [item.timestamp for item in data]
    values = [item.value for item in data]
    df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "value": values})
    return TimeSeries.from_dataframe(df, "timestamp", "value", freq='h')

def _build_future_covariates(series: 'TimeSeries', horizon: int, model) -> 'TimeSeries':
    from darts import TimeSeries
    from darts.utils.timeseries_generation import datetime_attribute_timeseries
    print(f"[Helper] Building future covariates for horizon: {horizon}")

    output_chunk_length = model.output_chunk_length
    required_cov_len = horizon + output_chunk_length
    
    last_timestamp = series.end_time()
    future_index = pd.date_range(start=last_timestamp + series.freq, periods=required_cov_len, freq=series.freq)
    
    dummy_series = TimeSeries.from_times_and_values(future_index, np.zeros(required_cov_len))
    
    future_covariates = datetime_attribute_timeseries(
        dummy_series,
        attribute="hour",
        one_hot=True
    ).stack(
        datetime_attribute_timeseries(dummy_series, attribute="day_of_week", one_hot=True)
    ).astype(np.float32)
    
    print(f"[Helper] Future covariates built successfully with length: {len(future_covariates)}.")
    return future_covariates

# --- API Endpoints ---

@app.get("/ping", summary="Check service status")
def ping():
    return {"status": "ok", "message": "Service is running."}

@app.post("/assets/{asset_id}/predict", 
          response_model=PredictionResponse,
          summary="Execute energy consumption forecast")
def predict(asset_id: str, request: PredictionRequest, http_request: Request):
    print(f"\n--- Received prediction request for asset: {asset_id} ---")
    
    model_cache = http_request.app.state.model_cache
    if "default" not in model_cache:
        print("[Error] Model not found in cache.")
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    model = model_cache["default"]
    print("Model retrieved from cache.")

    try:
        series = _create_timeseries_from_request(request.historical_data)
    except Exception as e:
        print(f"[Error] Failed to parse time series data: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse time series data: {e}")

    try:
        future_covs = _build_future_covariates(series, request.forecast_horizon, model)
        
        print(f"Executing prediction for horizon: {request.forecast_horizon}")
        forecast = model.predict(
            n=request.forecast_horizon, 
            series=series,
            future_covariates=future_covs
        )
        print("Prediction successful.")
    except Exception as e:
        print(f"[Error] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # --- 最终修复方案 ---
    # 解释：由于您使用的 u8darts 0.36.0 版本没有 pd_dataframe() 方法，
    # 我们将使用 TimeSeries 对象最核心的 .values() 和 .time_index 属性来手动构建 DataFrame。
    # 这种方法具有最好的版本兼容性。
    try:
        print("[INFO] Building DataFrame manually using .values() and .time_index")
        
        # 核心修复代码：
        forecast_df = pd.DataFrame(
            data=forecast.values(),          # 获取数值数据
            index=forecast.time_index,       # 获取时间戳索引
            columns=['predicted_value']      # 指定列名，可以自定义
        )
        
        print("[SUCCESS] Manual DataFrame construction successful.")

    except Exception as e:
        print(f"[CRITICAL] Manual DataFrame construction failed: {e}")
        # 打印诊断信息以防万一
        print(f"[DIAGNOSTICS] The type of 'forecast' is: {type(forecast)}")
        print(f"[DIAGNOSTICS] Available attributes/methods on 'forecast': {dir(forecast)}")
        raise HTTPException(status_code=500, detail="Failed to convert forecast object to DataFrame. Please check server logs.")
    # --- 修复结束 ---

    # 将DataFrame转换为Pydantic模型的响应格式
    # 注意：这里需要根据新的列名 'predicted_value' 来获取值
    forecast_data = [
        ForecastDataPoint(timestamp=ts, predicted_value=row['predicted_value'])
        for ts, row in forecast_df.iterrows()
    ]
    print("Formatting response.")

    return PredictionResponse(asset_id=asset_id, forecast_data=forecast_data)

print("--- Script main.py finished execution ---")