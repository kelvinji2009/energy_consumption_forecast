
import os
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Project imports
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from database.database import Asset, Model, ApiKey
from celery_worker.tasks import train_model_task # UPDATED: Import new task

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
    scaler_cov_path: Optional[str] = None
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

# --- Admin API Router ---
router = APIRouter(prefix="/admin", tags=["Admin Operations"])

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

# --- Other Endpoints (API Keys, etc. - no changes) ---
# (Assuming other endpoints like API key management remain the same)

# --- API Key Endpoints ---

@router.post("/api-keys", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(key_create: ApiKeyCreate, db: Session = Depends(get_db_session)):
    new_key_str = str(uuid.uuid4())
    key_hash = new_key_str # In a real app, use bcrypt etc. for hashing

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
