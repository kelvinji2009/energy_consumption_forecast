
from database.database import engine, assets_table, models_table, api_keys_table
from sqlalchemy import insert, select, update
import os
import uuid
from datetime import datetime

# --- Configuration ---
MODEL_RELATIVE_PATH = os.path.join("demo", "models", "lgbm_energy_model", "model.joblib")

# --- Initial Data ---
DEFAULT_ASSET_ID = "production_line_A"
DEFAULT_ASSET_NAME = "生产线A"
DEFAULT_MODEL_VERSION = "v1.0"
DEFAULT_MODEL_TYPE = "LightGBM"
# This key is hardcoded in the frontend's ForecastView.jsx for demo purposes
DEFAULT_API_KEY = "3369df94-7513-459e-be83-104bdb046b85"

def insert_initial_data():
    """插入初始资产、模型和API密钥数据到数据库。"""
    print("--- Inserting initial data ---")
    with engine.connect() as connection:
        # 1. 检查并插入默认资产
        stmt_check_asset = select(assets_table).where(assets_table.c.id == DEFAULT_ASSET_ID)
        if not connection.execute(stmt_check_asset).fetchone():
            stmt_insert_asset = insert(assets_table).values(
                id=DEFAULT_ASSET_ID, name=DEFAULT_ASSET_NAME, description="主要的生产线能耗预测资产"
            )
            connection.execute(stmt_insert_asset)
            print(f"Inserted asset: {DEFAULT_ASSET_ID}")
        else:
            print(f"Asset {DEFAULT_ASSET_ID} already exists.")

        # 2. 检查并插入默认模型
        stmt_check_model = select(models_table).where(
            models_table.c.asset_id == DEFAULT_ASSET_ID,
            models_table.c.version == DEFAULT_MODEL_VERSION
        )
        if not connection.execute(stmt_check_model).fetchone():
            stmt_insert_model = insert(models_table).values(
                id=uuid.uuid4(), asset_id=DEFAULT_ASSET_ID, version=DEFAULT_MODEL_VERSION,
                path=MODEL_RELATIVE_PATH, is_active=True, metrics={"mape": 0.0},
                model_type=DEFAULT_MODEL_TYPE
            )
            connection.execute(stmt_insert_model)
            print(f"Inserted active model {DEFAULT_MODEL_VERSION} for asset {DEFAULT_ASSET_ID}.")
        else:
            print(f"Model {DEFAULT_MODEL_VERSION} for asset {DEFAULT_ASSET_ID} already exists.")

        connection.commit()
        print("--- Initial data insertion complete ---")

if __name__ == "__main__":
    insert_initial_data()

