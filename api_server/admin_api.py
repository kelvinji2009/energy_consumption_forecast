import os
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy import insert, select, update, delete, Connection
from sqlalchemy.exc import IntegrityError

# 导入数据库组件
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from database.database import engine, assets_table, models_table, api_keys_table

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
    version: str = Field(..., max_length=50)
    path: str = Field(..., description="模型文件在文件系统中的相对路径")
    model_type: str = Field(..., max_length=50)
    metrics: Optional[dict] = None

class ModelCreate(ModelBase):
    pass

class ModelResponse(ModelBase):
    id: uuid.UUID
    is_active: bool
    trained_at: datetime

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
def get_db_connection():
    with engine.connect() as connection:
        yield connection

# --- Asset Endpoints ---

@router.post("/assets", response_model=AssetResponse, status_code=status.HTTP_201_CREATED)
def create_asset(asset: AssetCreate, db: Connection = Depends(get_db_connection)):
    try:
        stmt = insert(assets_table).values(**asset.model_dump())
        result = db.execute(stmt)
        db.commit()
        new_asset = db.execute(select(assets_table).where(assets_table.c.id == asset.id)).fetchone()
        return AssetResponse.model_validate(new_asset)
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Asset with this ID already exists.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create asset: {e}")

@router.get("/assets", response_model=List[AssetResponse])
def read_assets(db: Connection = Depends(get_db_connection)):
    stmt = select(assets_table)
    assets = db.execute(stmt).fetchall()
    return [AssetResponse.model_validate(asset) for asset in assets]

@router.get("/assets/{asset_id}", response_model=AssetResponse)
def read_asset(asset_id: str, db: Connection = Depends(get_db_connection)):
    stmt = select(assets_table).where(assets_table.c.id == asset_id)
    asset = db.execute(stmt).fetchone()
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found.")
    return AssetResponse.model_validate(asset)

@router.put("/assets/{asset_id}", response_model=AssetResponse)
def update_asset(asset_id: str, asset_update: AssetBase, db: Connection = Depends(get_db_connection)):
    stmt = update(assets_table).where(assets_table.c.id == asset_id).values(**asset_update.model_dump(exclude_unset=True))
    result = db.execute(stmt)
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Asset not found.")
    db.commit()
    updated_asset = db.execute(select(assets_table).where(assets_table.c.id == asset_id)).fetchone()
    return AssetResponse.model_validate(updated_asset)

@router.delete("/assets/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_asset(asset_id: str, db: Connection = Depends(get_db_connection)):
    stmt = delete(assets_table).where(assets_table.c.id == asset_id)
    result = db.execute(stmt)
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Asset not found.")
    db.commit()
    return

# --- Model Endpoints ---

@router.post("/models", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
def create_model(model: ModelCreate, db: Connection = Depends(get_db_connection)):
    try:
        stmt = insert(models_table).values(**model.model_dump(), is_active=False)
        result = db.execute(stmt)
        db.commit()
        new_model = db.execute(select(models_table).where(models_table.c.id == result.inserted_primary_key[0])).fetchone()
        return ModelResponse.model_validate(new_model)
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Model with this version already exists for the asset.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create model: {e}")

@router.get("/models", response_model=List[ModelResponse])
def read_models(db: Connection = Depends(get_db_connection), asset_id: Optional[str] = None):
    stmt = select(models_table)
    if asset_id:
        stmt = stmt.where(models_table.c.asset_id == asset_id)
    models = db.execute(stmt).fetchall()
    return [ModelResponse.model_validate(model) for model in models]

@router.get("/models/{model_id}", response_model=ModelResponse)
def read_model(model_id: uuid.UUID, db: Connection = Depends(get_db_connection)):
    stmt = select(models_table).where(models_table.c.id == model_id)
    model = db.execute(stmt).fetchone()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    return ModelResponse.model_validate(model)

@router.put("/models/{model_id}/activate", response_model=ModelResponse)
def activate_model(model_id: uuid.UUID, db: Connection = Depends(get_db_connection)):
    stmt_select = select(models_table).where(models_table.c.id == model_id)
    model_to_activate = db.execute(stmt_select).fetchone()
    if model_to_activate is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    
    asset_id = model_to_activate.asset_id

    try:
        stmt_deactivate_all = update(models_table).where(
            models_table.c.asset_id == asset_id,
            models_table.c.is_active == True
        ).values(is_active=False)
        db.execute(stmt_deactivate_all)

        stmt_activate_one = update(models_table).where(models_table.c.id == model_id).values(is_active=True)
        db.execute(stmt_activate_one)
        db.commit()

        updated_model = db.execute(stmt_select).fetchone()
        return ModelResponse.model_validate(updated_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {e}")

@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: uuid.UUID, db: Connection = Depends(get_db_connection)):
    stmt = delete(models_table).where(models_table.c.id == model_id)
    result = db.execute(stmt)
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Model not found.")
    db.commit()
    return

# --- API Key Endpoints ---

@router.post("/api-keys", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(key_create: ApiKeyCreate, db: Connection = Depends(get_db_connection)):
    new_key = str(uuid.uuid4())
    key_hash = new_key # 实际应用中应使用 bcrypt 等进行哈希

    try:
        stmt = insert(api_keys_table).values(
            id=uuid.uuid4(),
            key_hash=key_hash,
            description=key_create.description,
            is_active=True,
            created_at=datetime.now()
        )
        result = db.execute(stmt)
        db.commit()
        
        # 核心修复：返回 ApiKeyCreateResponse，包含明文密钥
        response_data = ApiKeyCreateResponse(
            id=result.inserted_primary_key[0],
            key_hash=key_hash, # 确保这里是key_hash
            key=new_key, # 明文密钥
            description=key_create.description,
            is_active=True,
            created_at=datetime.now(),
            expires_at=None
        )
        return response_data
    except IntegrityError:
        raise HTTPException(status_code=400, detail="API Key hash collision (highly unlikely).")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create API Key: {e}")

@router.get("/api-keys", response_model=List[ApiKeyResponse])
def read_api_keys(db: Connection = Depends(get_db_connection)):
    stmt = select(api_keys_table)
    keys = db.execute(stmt).fetchall()
    # 核心修复：确保从数据库行映射到 ApiKeyResponse 时，key_hash ��段被正确填充
    return [ApiKeyResponse.model_validate(key, from_attributes=True) for key in keys]

@router.put("/api-keys/{key_id}/toggle-active", response_model=ApiKeyResponse)
def toggle_api_key_active(key_id: uuid.UUID, db: Connection = Depends(get_db_connection)):
    stmt_select = select(api_keys_table).where(api_keys_table.c.id == key_id)
    key_record = db.execute(stmt_select).fetchone()
    if key_record is None:
        raise HTTPException(status_code=404, detail="API Key not found.")
    
    new_active_status = not key_record.is_active
    stmt_update = update(api_keys_table).where(api_keys_table.c.id == key_id).values(is_active=new_active_status)
    db.execute(stmt_update)
    db.commit()

    updated_key = db.execute(stmt_select).fetchone()
    # 核心修复：确保从数据库行映射到 ApiKeyResponse 时，key_hash 字段被正确填充
    return ApiKeyResponse.model_validate(updated_key, from_attributes=True)

@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_api_key(key_id: uuid.UUID, db: Connection = Depends(get_db_connection)):
    stmt = delete(api_keys_table).where(api_keys_table.c.id == key_id)
    result = db.execute(stmt)
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="API Key not found.")
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