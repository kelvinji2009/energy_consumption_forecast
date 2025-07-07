import os
import sys
import requests
import pandas as pd
import numpy as np
import joblib
import uuid
from datetime import datetime

# 将项目根目录添加到sys.path，以便能够导入database和demo脚本
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from celery_worker.celery_app import celery_app
from database.database import engine, Model
from sqlalchemy import insert, select, update

# 导入Darts相关的库
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import LightGBMModel
from darts.metrics import mape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.ad.detectors import QuantileDetector

# --- Configuration for Task ---
MODELS_SAVE_BASE_DIR = os.path.join(PROJECT_ROOT, "demo", "models")

@celery_app.task(bind=True, name="train_model_for_asset")
def train_model_for_asset(self, asset_id: str, data_url: str):
    """Celery任务：为指定资产训练新的能耗预测模型。"""
    task_id = self.request.id
    print(f"[Celery Task {task_id}] Starting training for asset: {asset_id} from URL: {data_url}")

    try:
        # --- 1. 下载数据 ---
        print(f"[Celery Task {task_id}] Downloading data from {data_url}...")
        response = requests.get(data_url, stream=True)
        response.raise_for_status() # 检查HTTP请求是否成功
        
        # 将下载的数据保存到临时文件
        temp_csv_path = os.path.join(PROJECT_ROOT, "data", f"temp_data_{asset_id}_{task_id}.csv")
        with open(temp_csv_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[Celery Task {task_id}] Data downloaded to {temp_csv_path}")

        # --- 2. 数据预处理 (模拟 01_data_preprocessing.py) ---
        print(f"[Celery Task {task_id}] Starting data preprocessing...")
        df = pd.read_csv(temp_csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # 计算单耗 (unit_consumption) - 简化处理，仅为演示
        production_no_zeros = df['production_units'].replace(0, np.nan)
        df['unit_consumption'] = df['energy_kwh'] / production_no_zeros
        df['unit_consumption'].fillna(method='ffill', inplace=True)
        df['unit_consumption'].fillna(method='bfill', inplace=True)
        df = df.astype(np.float32)
        print(f"[Celery Task {task_id}] Data preprocessing complete.")

        # --- 3. 准备Darts TimeSeries (模拟 02_train_and_evaluate_lgbm.py) ---
        print(f"[Celery Task {task_id}] Preparing TimeSeries for Darts...")
        series_energy = TimeSeries.from_series(df['energy_kwh'], freq='H').astype(np.float32)
        future_covariates = datetime_attribute_timeseries(
            series_energy,
            attribute="hour",
            one_hot=True
        ).stack(
            datetime_attribute_timeseries(series_energy, attribute="day_of_week", one_hot=True)
        ).astype(np.float32)

        # 分割数据
        train_cutoff = series_energy.time_index[- (14 * 24)] # 留出最后两周做验证
        train_energy, val_energy = series_energy.split_before(train_cutoff)
        train_future_cov, val_future_cov = future_covariates.split_before(train_cutoff)

        # 缩放数据
        scaler_energy = Scaler()
        scaler_future_cov = Scaler()
        train_energy_scaled = scaler_energy.fit_transform(train_energy)
        val_energy_scaled = scaler_energy.transform(val_energy)
        train_future_cov_scaled = scaler_future_cov.fit_transform(train_future_cov)
        val_future_cov_scaled = scaler_future_cov.transform(val_future_cov)
        print(f"[Celery Task {task_id}] TimeSeries preparation complete.")

        # --- 4. 模型训练 (LightGBM) ---
        print(f"[Celery Task {task_id}] Initializing and training LightGBM model...")
        input_chunk_length = 24 * 7
        output_chunk_length = 24

        model_energy = LightGBMModel(
            lags=input_chunk_length,
            lags_future_covariates=[0, output_chunk_length-1],
            output_chunk_length=output_chunk_length,
            random_state=42,
        )

        model_energy.fit(
            series=train_energy_scaled,
            future_covariates=train_future_cov_scaled,
            val_series=val_energy_scaled,
            val_future_covariates=val_future_cov_scaled,
        )
        print(f"[Celery Task {task_id}] Model training complete.")

        # --- 5. 评估模型 (可选，但推荐) ---
        print(f"[Celery Task {task_id}] Evaluating model...")
        historical_forecasts_scaled = model_energy.historical_forecasts(
            series=val_energy_scaled,
            future_covariates=val_future_cov_scaled,
            start=0.1,
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=False
        )
        historical_forecasts = scaler_energy.inverse_transform(historical_forecasts_scaled)
        mape_score = mape(val_energy, historical_forecasts)
        print(f"[Celery Task {task_id}] MAPE on Validation Set: {mape_score:.2f}%")
        metrics = {"mape": round(mape_score, 2)}

        # --- 6. 拟合并保存异常检测器 ---
        print(f"[Celery Task {task_id}] Fitting and saving anomaly detector...")
        # 使用训练数据生成历史预测来拟合检测器
        train_historical_forecasts_scaled = model_energy.historical_forecasts(
            series=train_energy_scaled,
            future_covariates=train_future_cov_scaled,
            start=input_chunk_length, # 从有足够滞后特征的地方开始
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=False
        )
        # 计算残差
        train_residuals_scaled = train_energy_scaled[train_historical_forecasts_scaled.time_index] - train_historical_forecasts_scaled
        train_residuals_series = pd.Series(np.abs(train_residuals_scaled.all_values()).reshape(-1), index=train_residuals_scaled.time_index)
        train_residuals_ts = TimeSeries.from_series(train_residuals_series)

        # 拟合QuantileDetector
        detector = QuantileDetector(high_quantile=0.98) # 可以将此值配置化
        detector.fit(train_residuals_ts)
        print(f"[Celery Task {task_id}] Anomaly detector fitted.")

        # --- 7. 保存模型、检测器和Scaler ---
        print(f"[Celery Task {task_id}] Saving model, detector, and scaler...")
        model_version = datetime.now().strftime("%Y%m%d%H%M%S") # 基于时间戳的版本号
        model_dir_for_asset = os.path.join(MODELS_SAVE_BASE_DIR, f"lgbm_energy_model_{asset_id}")
        os.makedirs(model_dir_for_asset, exist_ok=True)
        
        model_filename = f"model_{model_version}.joblib"
        model_save_path_abs = os.path.join(model_dir_for_asset, model_filename)
        joblib.dump(model_energy, model_save_path_abs)
        
        detector_filename = f"detector_{model_version}.joblib"
        detector_save_path_abs = os.path.join(model_dir_for_asset, detector_filename)
        joblib.dump(detector, detector_save_path_abs)

        # 保存 scaler_energy
        scaler_filename = f"scaler_{model_version}.joblib"
        scaler_save_path_abs = os.path.join(model_dir_for_asset, scaler_filename)
        joblib.dump(scaler_energy, scaler_save_path_abs)

        # 存储相对于项目根目录的路径 (如果需要，可以更新数据库字段来存储这些路径)
        model_relative_path = os.path.relpath(model_save_path_abs, PROJECT_ROOT)
        detector_relative_path = os.path.relpath(detector_save_path_abs, PROJECT_ROOT)
        scaler_relative_path = os.path.relpath(scaler_save_path_abs, PROJECT_ROOT)
        
        print(f"[Celery Task {task_id}] Model saved to: {model_save_path_abs}")
        print(f"[Celery Task {task_id}] Detector saved to: {detector_save_path_abs}")
        print(f"[Celery Task {task_id}] Scaler saved to: {scaler_save_path_abs}")

        # --- 8. 更新数据库 ---
        print(f"[Celery Task {task_id}] Updating database...")
        with engine.connect() as connection:
            # 1. 将该资产下所有旧模型设置为不活跃
            stmt_deactivate_old = update(Model).where(
                Model.asset_id == asset_id,
                Model.is_active == True
            ).values(is_active=False)
            connection.execute(stmt_deactivate_old)

            # 2. 插入新模型记录并设置为活跃
            stmt_insert_new = insert(Model).values(
                asset_id=asset_id,
                model_type="LightGBM",
                model_version=model_version,
                model_path=model_relative_path,
                scaler_path=scaler_relative_path,
                scaler_cov_path=None, # LightGBM does not use scaler_cov_path in this example
                is_active=True,
                description=f"LightGBM model trained on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                metrics=metrics,
            )
            connection.execute(stmt_insert_new)
            connection.commit()
        print(f"[Celery Task {task_id}] Database updated successfully. New model {model_version} is active.")

        # --- 9. 清理临时文件 ---
        os.remove(temp_csv_path)
        print(f"[Celery Task {task_id}] Cleaned up temporary file: {temp_csv_path}")

        print(f"[Celery Task {task_id}] Training task for asset {asset_id} completed successfully.")
        return {"status": "success", "asset_id": asset_id, "model_version": model_version, "metrics": metrics}

    except requests.exceptions.RequestException as e:
        print(f"[Celery Task {task_id}] ERROR: Failed to download data. Error: {e}")
        return {"status": "failed", "asset_id": asset_id, "error": f"Data download failed: {e}"}
    except Exception as e:
        print(f"[Celery Task {task_id}] CRITICAL ERROR during training for asset {asset_id}: {e}")
        return {"status": "failed", "asset_id": asset_id, "error": f"Training failed: {e}"}
