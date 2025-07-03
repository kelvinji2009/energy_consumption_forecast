
from sqlalchemy import create_engine, MetaData, Table, Column, String, Boolean, DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import os

# --- Configuration ---
# 数据库连接字符串。在生产环境中，这应该通过环境变量配置。
# 示例：postgresql://user:password@host:port/dbname
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/energy_forecast_db")

# --- Database Engine and Metadata ---
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# --- Table Definitions ---

# assets 表：存储产线/车间的信息
assets_table = Table(
    "assets",
    metadata,
    Column("id", String, primary_key=True, index=True, comment="资产的唯一标识符，例如生产线ID"),
    Column("name", String(255), nullable=False, comment="资产的友好名称"),
    Column("description", Text, nullable=True, comment="资产的描述"),
    Column("created_at", DateTime, nullable=False, default=func.now(), comment="创建时间"),
    Column("updated_at", DateTime, nullable=False, default=func.now(), onupdate=func.now(), comment="最后更新时间"),
    comment="存储产线/车间等资产的基本信息"
)

# models 表：存储训练好的模型信息
models_table = Table(
    "models",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=func.gen_random_uuid(), comment="模型的唯一ID"),
    Column("asset_id", String, nullable=False, comment="关联的资产ID (外键到 assets.id)"),
    Column("version", String(50), nullable=False, comment="模型版本，例如 v1.0"),
    Column("path", Text, nullable=False, comment="模型文件在文件系统中的路径"),
    Column("is_active", Boolean, nullable=False, default=False, comment="是否是当前资产的活跃模型"),
    Column("trained_at", DateTime, nullable=False, default=func.now(), comment="模型训练完成时间"),
    Column("metrics", JSONB, nullable=True, comment="训练指标，例如 MAPE, RMSE"),
    Column("model_type", String(50), nullable=False, comment="模型类型，例如 LightGBM, TFT, LSTM, TiDE"),
    comment="存储训练好的模型及其元数据"
)

# api_keys 表：存储API密钥
api_keys_table = Table(
    "api_keys",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=func.gen_random_uuid(), comment="API密钥的唯一ID"),
    Column("key_hash", String(255), nullable=False, unique=True, comment="API密钥的哈希值 (不直接存储明文密钥)"),
    Column("description", Text, nullable=True, comment="密钥用途描述"),
    Column("is_active", Boolean, nullable=False, default=True, comment="密钥是否活跃"),
    Column("created_at", DateTime, nullable=False, default=func.now(), comment="创建时间"),
    Column("expires_at", DateTime, nullable=True, comment="过期时间 (可选)"),
    comment="存储API密钥的哈希值和相关信息"
)

# --- Function to Create Tables ---

def create_tables():
    """在数据库中创建所有定义的表。"""
    print(f"Attempting to connect to database: {DATABASE_URL.split('@')[-1]}")
    try:
        metadata.create_all(engine)
        print("Database tables created or already exist.")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        print("Please ensure your PostgreSQL server is running and accessible.")

if __name__ == "__main__":
    # 这是一个示例，您需要确保PostgreSQL服务正在运行
    # 并且 DATABASE_URL 环境变量已正确设置
    create_tables()
