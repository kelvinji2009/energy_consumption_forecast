
import os
import uuid
import bcrypt
import boto3
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Project imports
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from database.database import Asset, Model, ApiKey
from celery_worker.tasks import train_model_task
from api_server.dependencies import verify_api_key

# --- S3/MinIO Configuration ---
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = "models"
S3_TRAINING_DATA_FOLDER = "training-data"

def get_s3_client():
    """Creates a boto3 S3 client."""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY
    )

# --- Pydantic Models for Admin API ---

# Assets
class AssetBase(BaseModel):
    name: str = Field(..., max_length=255)
    description: Optional[str] = None

class AssetCreate(AssetBase):
    id: str = Field(..., max_length=255, description="Unique asset ID, e.g., production_line_A")

class AssetResponse(AssetCreate):
    created_at: datetime
    updated_at: datetime
    class Config: from_attributes = True

# Models
class ModelBase(BaseModel):
    asset_id: str
    model_type: str = Field(..., max_length=50)
    description: Optional[str] = None

class ModelResponse(ModelBase):
    id: int
    model_version: Optional[str] = None
    status: str
    model_path: Optional[str] = None
    scaler_path: Optional[str] = None
    scaler_past_cov_path: Optional[str] = None
    detector_path: Optional[str] = None
    training_data_path: Optional[str] = None
    is_active: bool
    created_at: datetime
    metrics: Optional[dict] = None
    class Config: from_attributes = True

# API Keys
class ApiKeyCreate(BaseModel):
    description: Optional[str] = None

class ApiKeyResponse(BaseModel):
    id: uuid.UUID
    key_hash: str
    description: Optional[str] = None
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime] = None
    class Config: from_attributes = True

class ApiKeyCreateResponse(ApiKeyResponse):
    key: str

# --- NEW: Training Job Models ---
class TrainingJobCreate(BaseModel):
    asset_id: str
    model_type: str = Field("LightGBM", description="The type of model to train.")
    s3_data_path: str = Field(..., description="The path (key) to the training CSV file in the S3 bucket.")
    description: Optional[str] = "LGBM model trained via API."
    n_epochs: int = Field(20, description="Number of training epochs for the model.")

class TrainingJobResponse(BaseModel):
    message: str
    model_id: int
    task_id: str
    asset_id: str
    status: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None

# --- Celery App import for task status ---
from celery_worker.celery_app import celery_app
from celery.result import AsyncResult

# --- Admin API Router ---
router = APIRouter(prefix="/admin", tags=["Admin Operations"], dependencies=[Depends(verify_api_key)])


# --- Dependency for Database Session ---
from database.database import engine
def get_db_session():
    with Session(engine) as session:
        yield session

# --- Asset Endpoints (No changes) ---
@router.post("/assets", response_model=AssetResponse, status_code=status.HTTP_201_CREATED)
def create_asset(asset: AssetCreate, db: Session = Depends(get_db_session)):
    new_asset = Asset(**asset.model_dump())
    db.add(new_asset)
    db.commit()
    db.refresh(new_asset)
    return new_asset

@router.get("/assets", response_model=List[AssetResponse])
def read_assets(db: Session = Depends(get_db_session)):
    return db.query(Asset).all()

# --- Model Endpoints (Read-only parts, no changes) ---
@router.get("/models", response_model=List[ModelResponse])
def read_models(db: Session = Depends(get_db_session), asset_id: Optional[str] = None):
    query = db.query(Model)
    if asset_id:
        query = query.filter(Model.asset_id == asset_id)
    return query.all()

@router.get("/models/{model_id}", response_model=ModelResponse)
def read_model(model_id: int, db: Session = Depends(get_db_session)):
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    return model

# --- NEW: Training Job Endpoint ---
@router.post("/training-jobs", response_model=TrainingJobResponse, status_code=status.HTTP_202_ACCEPTED)
def create_training_job(job_request: TrainingJobCreate, db: Session = Depends(get_db_session)):
    """
    Creates a new model training job.
    This endpoint creates a model record in the database with 'PENDING' status
    and queues a Celery task to perform the actual training.
    """
    asset = db.query(Asset).filter(Asset.id == job_request.asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset with id '{job_request.asset_id}' not found.")

    new_model = Model(
        asset_id=job_request.asset_id,
        model_type=job_request.model_type,
        status="PENDING",
        description=job_request.description,
        model_version=f"pending_{uuid.uuid4().hex[:8]}",
        created_at=datetime.now()
    )
    db.add(new_model)
    db.commit()
    db.refresh(new_model)

    task = train_model_task.delay(
        model_id=new_model.id,
        asset_id=new_model.asset_id,
        s3_data_path=job_request.s3_data_path,
        model_type=job_request.model_type,
        n_epochs=job_request.n_epochs
    )
    
    print(f"[Admin API] Queued training task {task.id} for model {new_model.id}")

    return TrainingJobResponse(
        message="Training job created and queued successfully.",
        model_id=new_model.id,
        task_id=str(task.id),
        asset_id=new_model.asset_id,
        status=new_model.status
    )

@router.post("/training-jobs-from-csv", response_model=TrainingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_training_job_from_csv(
    db: Session = Depends(get_db_session),
    asset_id: str = Form(...),
    model_type: str = Form(...),
    n_epochs: int = Form(20),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    """
    Creates a training job by uploading a CSV file.
    The file is first uploaded to S3, then the training task is queued.
    """
    # 1. Validate Asset
    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset with id '{asset_id}' not found.")

    # 2. Upload file to S3
    s3_client = get_s3_client()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    s3_key = f"{S3_TRAINING_DATA_FOLDER}/{asset_id}_{timestamp}_{file.filename}"
    
    try:
        s3_client.upload_fileobj(file.file, S3_BUCKET_NAME, s3_key)
        print(f"Successfully uploaded training data to s3://{S3_BUCKET_NAME}/{s3_key}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file to S3: {e}")

    # 3. Create Model DB record
    final_description = description or f"UI-initiated training for {asset_id} with {model_type}"
    new_model = Model(
        asset_id=asset_id,
        model_type=model_type,
        status="PENDING",
        description=final_description,
        model_version=f"pending_{uuid.uuid4().hex[:8]}",
        training_data_path=s3_key, # Store the path to the uploaded data
        created_at=datetime.now()
    )
    db.add(new_model)
    db.commit()
    db.refresh(new_model)

    # 4. Dispatch Celery Task
    task = train_model_task.delay(
        model_id=new_model.id,
        asset_id=new_model.asset_id,
        s3_data_path=s3_key,
        model_type=model_type,
        n_epochs=n_epochs
    )
    
    print(f"[Admin API] Queued training task {task.id} for model {new_model.id} from uploaded CSV.")

    return TrainingJobResponse(
        message="Training job from uploaded CSV created and queued successfully.",
        model_id=new_model.id,
        task_id=str(task.id),
        asset_id=new_model.asset_id,
        status=new_model.status
    )

@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse, summary="获取异步任务状态")
def get_task_status(task_id: str):
    """
    Retrieves the status and result of a Celery task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None,
    }
    
    # If the task is in progress, add the 'status' from the meta info
    if task_result.status == 'PROGRESS' and isinstance(task_result.info, dict):
        response['result'] = task_result.info
    # If the task failed, the result is the exception, convert it to a string
    elif task_result.status == 'FAILURE':
        response['result'] = {'error': str(task_result.result)}

    return response

# --- Other Endpoints (API Keys, etc. - no changes) ---
# (Assuming other endpoints like API key management remain the same)

# --- API Key Endpoints ---

@router.post("/api-keys", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(key_create: ApiKeyCreate, db: Session = Depends(get_db_session)):
    """
    Creates a new API key. 
    The key is returned in plain text ONCE. 
    The system stores only its bcrypt hash.
    """
    new_key_str = str(uuid.uuid4())
    
    # Hash the key using bcrypt
    hashed_key = bcrypt.hashpw(new_key_str.encode('utf-8'), bcrypt.gensalt())
    key_hash = hashed_key.decode('utf-8')

    new_api_key = ApiKey(
        key_hash=key_hash,
        description=key_create.description,
        is_active=True,
        created_at=datetime.now()
    )
    db.add(new_api_key)
    db.commit()
    db.refresh(new_api_key)
    
    return ApiKeyCreateResponse(
        id=new_api_key.id,
        key_hash=new_api_key.key_hash,
        key=new_key_str,
        description=new_api_key.description,
        is_active=new_api_key.is_active,
        created_at=new_api_key.created_at,
        expires_at=new_api_key.expires_at
    )

@router.get("/api-keys", response_model=List[ApiKeyResponse])
def read_api_keys(db: Session = Depends(get_db_session)):
    keys = db.query(ApiKey).all()
    return keys

@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_api_key(key_id: uuid.UUID, db: Session = Depends(get_db_session)):
    key_record = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if key_record is None:
        raise HTTPException(status_code=404, detail="API Key not found.")
    db.delete(key_record)
    db.commit()
    return
