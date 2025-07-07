import os
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy import insert, select, update, delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session # Import Session

# 导入数据库组件
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from database.database import engine, Asset, Model, ApiKey # Import ORM models

# 导入Celery任务
from celery_worker.tasks import train_model_for_asset

# --- Pydantic Models for Admin API ---

# Assets
class AssetBase(BaseModel):
    name: str = Field(..., max_length=255)
    description: Optional[str] = None

class AssetCreate(AssetBase):
    id: str = Field(..., max_length=255, description="资产的唯一ID，例如 production_line_A")

class AssetResponse(AssetCreate):
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True # 兼容SQLAlchemy返回的对象

# Models
class ModelBase(BaseModel):
    asset_id: str
    model_type: str = Field(..., max_length=50)
    model_version: str = Field(..., max_length=50)
    model_path: str = Field(..., description="模型文件在文件系统中的相对路径")
    scaler_path: Optional[str] = Field(None, description="值缩放器文件的路径")
    scaler_cov_path: Optional[str] = Field(None, description="协变量缩放器文件的路径")
    description: Optional[str] = None
    metrics: Optional[dict] = None

class ModelCreate(ModelBase):
    pass

class ModelResponse(ModelBase):
    id: int # Changed from UUID to int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# API Keys
class ApiKeyCreate(BaseModel):
    description: Optional[str] = None

# 核心修复：修改 ApiKeyResponse，使其返回 key_hash 而不是 key
class ApiKeyResponse(BaseModel):
    id: uuid.UUID
    key_hash: str # 返回密钥的哈希值
    description: Optional[str] = None
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# 新增一个用于创建时返回明文密钥的响应模型
class ApiKeyCreateResponse(ApiKeyResponse):
    key: str # 仅在创建时返回明文密钥

# --- Admin API Router --- 为了避免与main.py中的PROJECT_ROOT冲突，这里不再定义

router = APIRouter(prefix="/admin", tags=["Admin Operations"])

# --- Dependency for Database Session ---
def get_db_session(): # Renamed to avoid conflict and clarify it returns a Session
    with Session(engine) as session:
        yield session

# --- Asset Endpoints ---

@router.post("/assets", response_model=AssetResponse, status_code=status.HTTP_201_CREATED)
def create_asset(asset: AssetCreate, db: Session = Depends(get_db_session)):
    try:
        new_asset = Asset(**asset.model_dump())
        db.add(new_asset)
        db.commit()
        db.refresh(new_asset)
        return new_asset
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Asset with this ID already exists.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create asset: {e}")

@router.get("/assets", response_model=List[AssetResponse])
def read_assets(db: Session = Depends(get_db_session)):
    assets = db.query(Asset).all()
    return assets

@router.get("/assets/{asset_id}", response_model=AssetResponse)
def read_asset(asset_id: str, db: Session = Depends(get_db_session)):
    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found.")
    return asset

@router.put("/assets/{asset_id}", response_model=AssetResponse)
def update_asset(asset_id: str, asset_update: AssetBase, db: Session = Depends(get_db_session)):
    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found.")
    
    for key, value in asset_update.model_dump(exclude_unset=True).items():
        setattr(asset, key, value)
    db.commit()
    db.refresh(asset)
    return asset

@router.delete("/assets/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_asset(asset_id: str, db: Session = Depends(get_db_session)):
    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found.")
    db.delete(asset)
    db.commit()
    return

# --- Model Endpoints ---

@router.post("/models", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
def create_model(model: ModelCreate, db: Session = Depends(get_db_session)):
    try:
        new_model = Model(**model.model_dump())
        db.add(new_model)
        db.commit()
        db.refresh(new_model)
        return new_model
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Model with this version already exists for the asset.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create model: {e}")

@router.get("/models", response_model=List[ModelResponse])
def read_models(db: Session = Depends(get_db_session), asset_id: Optional[str] = None):
    query = db.query(Model)
    if asset_id:
        query = query.filter(Model.asset_id == asset_id)
    models = query.all()
    return models

@router.get("/models/{model_id}", response_model=ModelResponse)
def read_model(model_id: int, db: Session = Depends(get_db_session)):
    model = db.query(Model).filter(Model.id == model_id).first()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    return model

@router.put("/models/{model_id}/activate", response_model=ModelResponse)
def activate_model(model_id: int, db: Session = Depends(get_db_session)):
    model_to_activate = db.query(Model).filter(Model.id == model_id).first()
    if model_to_activate is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    
    asset_id = model_to_activate.asset_id

    try:
        # Deactivate all other models for this asset
        db.query(Model).filter(
            Model.asset_id == asset_id,
            Model.is_active == True
        ).update({'is_active': False}, synchronize_session=False)

        # Activate the selected model
        model_to_activate.is_active = True
        db.commit()
        db.refresh(model_to_activate)
        return model_to_activate
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {e}")

@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: int, db: Session = Depends(get_db_session)):
    model = db.query(Model).filter(Model.id == model_id).first()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    db.delete(model)
    db.commit()
    return

# --- API Key Endpoints ---

@router.post("/api-keys", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(key_create: ApiKeyCreate, db: Session = Depends(get_db_session)):
    new_key_str = str(uuid.uuid4())
    key_hash = new_key_str # In a real app, use bcrypt etc. for hashing

    try:
        new_api_key = ApiKey(
            key_hash=key_hash,
            description=key_create.description,
            is_active=True,
            created_at=datetime.now()
        )
        db.add(new_api_key)
        db.commit()
        db.refresh(new_api_key)
        
        response_data = ApiKeyCreateResponse(
            id=new_api_key.id,
            key_hash=new_api_key.key_hash,
            key=new_key_str,
            description=new_api_key.description,
            is_active=new_api_key.is_active,
            created_at=new_api_key.created_at,
            expires_at=new_api_key.expires_at
        )
        return response_data
    except IntegrityError:
        raise HTTPException(status_code=400, detail="API Key hash collision (highly unlikely).")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create API Key: {e}")

@router.get("/api-keys", response_model=List[ApiKeyResponse])
def read_api_keys(db: Session = Depends(get_db_session)):
    keys = db.query(ApiKey).all()
    return keys

@router.put("/api-keys/{key_id}/toggle-active", response_model=ApiKeyResponse)
def toggle_api_key_active(key_id: uuid.UUID, db: Session = Depends(get_db_session)):
    key_record = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if key_record is None:
        raise HTTPException(status_code=404, detail="API Key not found.")
    
    key_record.is_active = not key_record.is_active
    db.commit()
    db.refresh(key_record)
    return key_record

@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_api_key(key_id: uuid.UUID, db: Session = Depends(get_db_session)):
    key_record = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if key_record is None:
        raise HTTPException(status_code=404, detail="API Key not found.")
    db.delete(key_record)
    db.commit()
    return

# --- Training Task Endpoint ---

class TrainModelRequest(BaseModel):
    asset_id: str
    data_url: str = Field(..., description="MinIO/S3中CSV数据文件的下载链接")

@router.post("/train-model", status_code=status.HTTP_202_ACCEPTED)
def trigger_model_training(request: TrainModelRequest):
    print(f"[Admin API] Received training request for asset {request.asset_id} from URL: {request.data_url}")
    # 将任务发送到Celery队列
    task = train_model_for_asset.delay(request.asset_id, request.data_url)
    print(f"[Admin API] Training task {task.id} queued for asset {request.asset_id}.")
    return {"message": "Training request received and queued.", "task_id": str(task.id), "asset_id": request.asset_id}