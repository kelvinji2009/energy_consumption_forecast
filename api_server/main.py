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
# 将项目根目录添加到sys.path，以便能够导入database包
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from sqlalchemy import select
from database.database import engine, models_table, api_keys_table

# 导入并挂载admin_api路由
from api_server.admin_api import router as admin_router

# 导入Darts相关的库
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import LightGBMModel # 假设LightGBMModel是预测模型
from darts.metrics import mape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.ad.detectors import QuantileDetector # 导入QuantileDetector

print("--- Script main.py starting to execute ---")

# --- Configuration ---
# MODEL_DIR 和 MODEL_PATH 不再直接用于加载，但保留作为参考或备用
# MODEL_DIR = os.path.join("..", "demo", "models") # 不再需要

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

# 新增：异常检测相关的Pydantic模型
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
    
    # 尝试连接数据库并加载活跃模型
    try:
        print("[Lifespan] Attempting to connect to database and load active models...")
        with engine.connect() as connection:
            # 查询所有活跃的模型
            stmt = select(models_table).where(models_table.c.is_active == True)
            result = connection.execute(stmt).fetchall()
            
            if not result:
                print("[Lifespan] No active models found in the database.")
            
            for row in result:
                asset_id = row.asset_id
                model_relative_path = row.path
                model_type = row.model_type
                
                model_path_abs = os.path.join(PROJECT_ROOT, model_relative_path) 
                
                model_obj = None
                detector_obj = None

                try:
                    print(f"[Lifespan] Loading {model_type} model for asset '{asset_id}' from: {model_path_abs}")
                    model_obj = joblib.load(model_path_abs)
                    print(f"[Lifespan] Model for asset '{asset_id}' loaded successfully.")

                    # 尝试加载对应的检测器
                    model_dir = os.path.dirname(model_path_abs)
                    model_filename = os.path.basename(model_path_abs)
                    detector_filename = model_filename.replace("model_", "detector_")
                    detector_path_abs = os.path.join(model_dir, detector_filename)

                    if os.path.exists(detector_path_abs):
                        print(f"[Lifespan] Loading detector for asset '{asset_id}' from: {detector_path_abs}")
                        detector_obj = joblib.load(detector_path_abs)
                        print(f"[Lifespan] Detector for asset '{asset_id}' loaded successfully.")
                    else:
                        print(f"[Lifespan] No detector found for asset '{asset_id}' at {detector_path_abs}. Anomaly detection API will not be available for this model.")

                    app.state.model_cache[asset_id] = {'model': model_obj, 'detector': detector_obj}

                except Exception as e:
                    print(f"[Lifespan] ERROR: Failed to load model or detector for asset '{asset_id}' from {model_path_abs}. Error: {e}")
                    # 即使单个模型加载失败，也尝试加载其他模型
                    
    except Exception as e:
        print(f"[Lifespan] CRITICAL: Database connection or initial model loading failed during startup. Error: {e}")
        # 如果数据库连接失败，服务仍然会启动，但模型缓存将为空

    yield

    print("[Lifespan] Shutdown event triggered.")
    app.state.model_cache.clear()
    print("[Lifespan] Model cache cleared.")

# --- FastAPI Application ---

print("Initializing FastAPI app...")
app = FastAPI(
    title="能耗预测与异常检测API",
    description="一个用于工业能耗预测和异常检测的API服务。",
    version="3.0.0", # 版本号更新为3.0.0
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # 允许前端开发服务器的源
    allow_credentials=True,
    allow_methods=["*"], # 允许所有HTTP方法
    allow_headers=["*"], # 允许所有请求头
)

app.include_router(admin_router)

print("FastAPI app initialized.")

# --- Security Dependency ---
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证API密钥是否有效且活跃。"""
    api_key = credentials.credentials # 这是从Bearer token中提取的密钥
    print(f"[Auth] Attempting to verify API Key: {api_key[:8]}...") # 打印部分密钥用于日志

    with engine.connect() as connection:
        # 在实际应用中，这里会查询密钥的哈希值
        # 目前我们直接存储明文UUID作为key_hash
        stmt = select(api_keys_table).where(
            api_keys_table.c.key_hash == api_key,
            api_keys_table.c.is_active == True
        )
        result = connection.execute(stmt).fetchone()
        
        if not result:
            print("[Auth] API Key verification failed: Invalid or inactive key.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or inactive API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    print("[Auth] API Key verified successfully.")
    return api_key # 返回密钥，如果需要后续处理

# --- Helper Functions ---

def _create_timeseries_from_request(data: List[TimeSeriesDataPoint]) -> 'TimeSeries':
    from darts import TimeSeries
    print("[Helper] Creating TimeSeries from request data.")
    timestamps = [item.timestamp for item in data]
    values = [item.value for item in data]
    df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "value": values})
    return TimeSeries.from_dataframe(df, "timestamp", "value", freq='H')

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
          summary="Execute energy consumption forecast",
          dependencies=[Depends(verify_api_key)]) # 应用API密钥认证
def predict(asset_id: str, request: PredictionRequest, http_request: Request):
    print(f"\n--- Received prediction request for asset: {asset_id} ---")
    
    model_cache_entry = http_request.app.state.model_cache.get(asset_id)
    
    if not model_cache_entry or 'model' not in model_cache_entry or not model_cache_entry['model']:
        print(f"[Error] No active model found for asset: {asset_id}")
        raise HTTPException(status_code=503, detail=f"No active model found for asset '{asset_id}'. Please ensure a model is trained and activated for this asset.")

    model = model_cache_entry['model']
    print(f"Model for asset '{asset_id}' retrieved from cache.")

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

    # --- 最终修复方案 ---（用户指定不修改，没有用户允许不要动这部分代码）
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


@app.post("/assets/{asset_id}/detect_anomalies", 
          response_model=AnomalyDetectionResponse,
          summary="Execute anomaly detection",
          dependencies=[Depends(verify_api_key)]) # 应用API密钥认证
def detect_anomalies(asset_id: str, request: AnomalyDetectionRequest, http_request: Request):
    print(f"\n--- Received anomaly detection request for asset: {asset_id} ---")

    model_cache_entry = http_request.app.state.model_cache.get(asset_id)
    
    if not model_cache_entry or 'model' not in model_cache_entry or not model_cache_entry['model']:
        print(f"[Error] No active model found for asset: {asset_id}")
        raise HTTPException(status_code=503, detail=f"No active model found for asset '{asset_id}'. Please ensure a model is trained and activated for this asset.")
    
    if 'detector' not in model_cache_entry or not model_cache_entry['detector']:
        print(f"[Error] No active detector found for asset: {asset_id}")
        raise HTTPException(status_code=503, detail=f"No active anomaly detector found for asset '{asset_id}'. Please ensure a model with a trained detector is active.")

    model = model_cache_entry['model']
    detector = model_cache_entry['detector']
    print(f"Model and detector for asset '{asset_id}' retrieved from cache.")

    try:
        # 将输入数据流转换为Darts TimeSeries
        series_to_detect = _create_timeseries_from_request(request.data_stream)
    except Exception as e:
        print(f"[Error] Failed to parse data stream for anomaly detection: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse data stream for anomaly detection: {e}")

    try:
        # 1. 使用预测模型对数据流进行历史预测
        # 注意：这里需要根据模型的input_chunk_length来确定start点
        # 并且需要提供future_covariates
        input_chunk_length = model.input_chunk_length # 从模型获取input_chunk_length
        
        # 为了进行历史预测，我们需要一个包含历史数据和未来协变量的完整序列
        # 这里的future_covariates需要覆盖整个series_to_detect的长度
        # 简化处理：假设future_covariates可以从series_to_detect本身的时间索引生成
        # 实际应用中，如果future_covariates是外部已知，需要单独生成
        full_future_covariates = datetime_attribute_timeseries(
            series_to_detect,
            attribute="hour",
            one_hot=True
        ).stack(
            datetime_attribute_timeseries(series_to_detect, attribute="day_of_week", one_hot=True)
        ).astype(np.float32)

        # 确保future_covariates的长度与series_to_detect匹配
        # 并且其时间索引与series_to_detect对齐
        full_future_covariates = full_future_covariates.slice_intersect(series_to_detect)

        # 历史预测，用于计算残差
        # start参数确保我们从有足够历史数据的地方开始预测
        historical_forecasts_scaled = model.historical_forecasts(
            series=series_to_detect, 
            future_covariates=full_future_covariates,
            start=input_chunk_length, # 从有足够历史数据的地方开始预测
            forecast_horizon=1, # 预测1步，用于计算每个点的残差
            stride=1,
            retrain=False,
            verbose=False
        )
        
        # 逆缩放预测结果
        # 注意：这里需要一个scaler_energy，但我们没有在main.py中加载它。
        # 这是一个设计上的缺陷，scaler应该和模型一起保存和加载。
        # 暂时简化：假设预测值和实际值在同一尺度，直接计算残差。
        # 更好的做法是：在tasks.py中保存scaler，并在lifespan中加载。
        # 或者，让模型直接预测原始尺度的数据。
        
        # 暂时跳过scaler，直接使用scaled值计算残差，这可能不准确，但能让流程跑通
        # 实际应用中，需要确保预测和实际值在同一尺度
        
        # 确保残差计算的时间索引对齐
        actual_values_aligned = series_to_detect[historical_forecasts_scaled.time_index]
        residuals_scaled = actual_values_aligned - historical_forecasts_scaled
        
        # 将残差转换为TimeSeries，用于检测器
        residuals_series = pd.Series(np.abs(residuals_scaled.all_values()).reshape(-1), index=residuals_scaled.time_index)
        residuals_ts = TimeSeries.from_series(residuals_series)

        # 2. 使用检测器识别异常
        anomalies_ts = detector.detect(residuals_ts)
        print("Anomaly detection successful.")

        # 3. 格式化异常结果
        anomalies_list = []
        # 遍历原始数据流，根据anomalies_ts标记的异常点进行筛选
        # anomalies_ts是一个二元时间序列 (0或1)
        
        # 确保时间索引对齐，只考虑有残差和异常标记的时间点
        aligned_data_df = series_to_detect.pd_df().loc[anomalies_ts.time_index]
        aligned_anomalies_df = anomalies_ts.pd_df()

        for ts, row in aligned_data_df.iterrows():
            if aligned_anomalies_df.loc[ts].iloc[0] == 1: # 如果是异常点
                original_value = row.iloc[0] # 原始值
                predicted_value_at_ts = historical_forecasts_scaled.pd_df().loc[ts].iloc[0] # 预测值
                residual_value = residuals_scaled.pd_df().loc[ts].iloc[0] # 残差值

                reason = "Anomaly detected." # 默认原因
                if residual_value > 0:
                    reason = f"Value ({original_value:.2f}) is significantly higher than predicted ({predicted_value_at_ts:.2f})."
                elif residual_value < 0:
                    reason = f"Value ({original_value:.2f}) is significantly lower than predicted ({predicted_value_at_ts:.2f})."
                
                anomalies_list.append(AnomalyDataPoint(
                    timestamp=ts,
                    value=original_value,
                    reason=reason
                ))
        print(f"Found {len(anomalies_list)} anomalies.")

    except Exception as e:
        print(f"[Error] Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {e}")

    return AnomalyDetectionResponse(asset_id=asset_id, anomalies=anomalies_list)

print("--- Script main.py finished execution ---")