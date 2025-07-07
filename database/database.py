from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Text, Integer, func
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/energy_forecast_db")

# --- Declarative Base ---
# Alembic and SQLAlchemy ORM use this as a base for model classes
Base = declarative_base()

# --- Table Definitions as Classes ---

class Asset(Base):
    __tablename__ = 'assets'
    id = Column(String, primary_key=True, index=True, comment="资产的唯一标识符，例如生产线ID")
    name = Column(String(255), nullable=False, comment="资产的友好名称")
    description = Column(Text, nullable=True, comment="资产的描述")
    created_at = Column(DateTime, nullable=False, default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now(), comment="最后更新时间")

class Model(Base):
    __tablename__ = 'models'
    id = Column(Integer, primary_key=True, autoincrement=True, comment="模型的唯一ID")
    asset_id = Column(String, nullable=False, index=True, comment="关联的资产ID (外键到 assets.id)")
    model_type = Column(String(50), nullable=False, comment="模型类型，例如 LightGBM, TFT, LSTM, TiDE")
    model_version = Column(String(50), nullable=False, comment="模型版本，例如训练时间戳 20250706100000")
    
    # --- NEW: Training Status ---
    status = Column(String(50), nullable=False, default='PENDING', comment="训练状态: PENDING, TRAINING, COMPLETED, FAILED")

    # --- MODIFIED: Paths are now S3 object keys ---
    model_path = Column(Text, nullable=True, comment="主模型文件在S3中的路径 (key)")
    scaler_path = Column(Text, nullable=True, comment="值缩放器文件在S3中的路径 (key)")
    scaler_cov_path = Column(Text, nullable=True, comment="协变量缩放器文件在S3中的路径 (key)")
    scaler_past_cov_path = Column(Text, nullable=True, comment="过去协变量缩放器文件在S3中的路径 (key)")
    
    # --- NEW: Traceability ---
    training_data_path = Column(Text, nullable=True, comment="训练数据在S3中的路径 (key)")

    is_active = Column(Boolean, nullable=False, default=False, comment="是否是当前资产的默认推荐模型")
    created_at = Column(DateTime, nullable=False, default=func.now(), comment="模型记录的创建时间")
    description = Column(Text, nullable=True, comment="模型的用户友好描述")
    metrics = Column(JSONB, nullable=True, comment="训练指标，例如 MAPE, RMSE")

class ApiKey(Base):
    __tablename__ = 'api_keys'
    id = Column(UUID(as_uuid=True), primary_key=True, default=func.gen_random_uuid(), comment="API密钥的唯一ID")
    key_hash = Column(String(255), nullable=False, unique=True, comment="API密钥的哈希值")
    description = Column(Text, nullable=True, comment="密钥用途描述")
    is_active = Column(Boolean, nullable=False, default=True, comment="密钥是否活跃")
    created_at = Column(DateTime, nullable=False, default=func.now(), comment="创建时间")
    expires_at = Column(DateTime, nullable=True, comment="过期时间 (可选)")

# --- Engine ---
engine = create_engine(DATABASE_URL)

# --- Main function for direct execution (optional) ---
def main():
    """A function to manually create tables if needed. Note: Alembic is the preferred way."""
    print(f"Attempting to connect to database: {DATABASE_URL.split('@')[-1]}")
    try:
        Base.metadata.create_all(engine)
        print("Database tables created or verified successfully.")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        print("Please ensure your PostgreSQL server is running and accessible.")

if __name__ == "__main__":
    main()