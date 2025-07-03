
from database.database import engine, assets_table, models_table
from sqlalchemy import insert, select
import os
import uuid
from datetime import datetime

# --- Configuration ---
# 假设模型文件相对于项目根目录的路径
MODEL_RELATIVE_PATH = os.path.join("demo", "models", "lgbm_energy_model", "model.joblib")

# --- Initial Data ---
DEFAULT_ASSET_ID = "production_line_A"
DEFAULT_ASSET_NAME = "生产线A"
DEFAULT_MODEL_VERSION = "v1.0"
DEFAULT_MODEL_TYPE = "LightGBM"

def insert_initial_data():
    """插入初始资产和模型数据到数据库。"""
    print("--- Inserting initial data ---")
    with engine.connect() as connection:
        # 1. 检查并插入默认资产
        stmt_check_asset = select(assets_table).where(assets_table.c.id == DEFAULT_ASSET_ID)
        existing_asset = connection.execute(stmt_check_asset).fetchone()

        if not existing_asset:
            stmt_insert_asset = insert(assets_table).values(
                id=DEFAULT_ASSET_ID,
                name=DEFAULT_ASSET_NAME,
                description="主要的生产线能耗预测资产",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            connection.execute(stmt_insert_asset)
            print(f"Inserted asset: {DEFAULT_ASSET_ID}")
        else:
            print(f"Asset {DEFAULT_ASSET_ID} already exists.")

        # 2. 检查并插入默认模型
        # 确保模型路径是相对于项目根目录的，因为API服务会根据这个路径加载
        stmt_check_model = select(models_table).where(
            models_table.c.asset_id == DEFAULT_ASSET_ID,
            models_table.c.version == DEFAULT_MODEL_VERSION
        )
        existing_model = connection.execute(stmt_check_model).fetchone()

        if not existing_model:
            stmt_insert_model = insert(models_table).values(
                id=uuid.uuid4(), # 生成一个新的UUID
                asset_id=DEFAULT_ASSET_ID,
                version=DEFAULT_MODEL_VERSION,
                path=MODEL_RELATIVE_PATH,
                is_active=True, # 标记为活跃模型
                trained_at=datetime.now(),
                metrics={"mape": 0.0, "rmse": 0.0}, # 占位符，实际应从训练结果获取
                model_type=DEFAULT_MODEL_TYPE
            )
            connection.execute(stmt_insert_model)
            print(f"Inserted active model {DEFAULT_MODEL_VERSION} for asset {DEFAULT_ASSET_ID}.")
        else:
            print(f"Model {DEFAULT_MODEL_VERSION} for asset {DEFAULT_ASSET_ID} already exists. Ensuring it is active.")
            # 如果模型已存在，确保它是活跃的
            from sqlalchemy import update
            stmt_update_model = update(models_table).where(
                models_table.c.asset_id == DEFAULT_ASSET_ID,
                models_table.c.version == DEFAULT_MODEL_VERSION
            ).values(is_active=True)
            connection.execute(stmt_update_model)

        connection.commit()
        print("--- Initial data insertion complete ---")

if __name__ == "__main__":
    insert_initial_data()
