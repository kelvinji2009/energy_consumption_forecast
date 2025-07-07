import os
import sys
import boto3
import pandas as pd
import joblib
from io import BytesIO
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from celery_worker.celery_app import celery_app
from database.database import engine, Model
from sqlalchemy.orm import sessionmaker
from core.training_service import train_model

# --- S3/MinIO Configuration ---
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = "models"

# --- Database Session ---
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_s3_client():
    """Creates a boto3 S3 client."""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY
    )

@celery_app.task(bind=True, name="train_model_task")
def train_model_task(self, model_id: int, asset_id: str, s3_data_path: str, model_type: str):
    """
    Celery task to train a model, save artifacts to S3, and update the database.
    """
    task_id = self.request.id
    print(f"[Task {task_id}] Starting training for model_id: {model_id}, asset_id: {asset_id}")
    
    db = SessionLocal()
    s3_client = get_s3_client()

    try:
        # 1. Update model status to TRAINING
        db.query(Model).filter(Model.id == model_id).update({"status": "TRAINING"})
        db.commit()

        # 2. Download data from S3
        # --- FIX: Make path handling more robust by stripping accidental bucket prefix ---
        key_path = s3_data_path
        bucket_prefix = f"{S3_BUCKET_NAME}/"
        if key_path.startswith(bucket_prefix):
            key_path = key_path[len(bucket_prefix):]
            print(f"[Task {task_id}] Stripped bucket prefix from path. New key: '{key_path}'")

        print(f"[Task {task_id}] Downloading data from s3://{S3_BUCKET_NAME}/{key_path}")
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key_path)
        data_content = response['Body'].read()
        df = pd.read_csv(BytesIO(data_content), index_col='timestamp', parse_dates=True)

        # 3. Call the core training service
        model, scaler_target, scaler_cov, metrics = train_model(model_type=model_type, data=df)

        # 4. Upload model and scalers to S3
        model_version = datetime.now().strftime("%Y%m%d%H%M%S")
        s3_prefix = f"{asset_id}/{model_id}_{model_version}"
        
        model_key = f"{s3_prefix}/model.joblib"
        scaler_target_key = f"{s3_prefix}/scaler_target.joblib"
        scaler_cov_key = f"{s3_prefix}/scaler_cov.joblib"

        for key, obj in zip([model_key, scaler_target_key, scaler_cov_key], [model, scaler_target, scaler_cov]):
            with BytesIO() as buffer:
                joblib.dump(obj, buffer)
                buffer.seek(0)
                s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=key, Body=buffer)
            print(f"[Task {task_id}] Uploaded artifact to s3://{S3_BUCKET_NAME}/{key}")

        # 5. Update database with final status and paths
        update_values = {
            "status": "COMPLETED",
            "model_version": model_version,
            "model_path": model_key,
            "scaler_path": scaler_target_key,
            "scaler_cov_path": scaler_cov_key,
            "training_data_path": s3_data_path,
            "metrics": metrics,
            "description": f"LGBM model trained on {datetime.now().strftime('%Y-%m-%d')}"
        }
        db.query(Model).filter(Model.id == model_id).update(update_values)
        db.commit()

        print(f"[Task {task_id}] Training task for model_id {model_id} completed successfully.")
        return {"status": "success", "model_id": model_id, "metrics": metrics}

    except Exception as e:
        print(f"[Task {task_id}] CRITICAL ERROR during training for model_id {model_id}: {e}")
        db.query(Model).filter(Model.id == model_id).update({"status": "FAILED"})
        db.commit()
        # Re-raise the exception to let Celery know the task failed
        raise
    finally:
        db.close()