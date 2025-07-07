import os
import sys
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(PROJECT_ROOT)

from database.database import engine, Model, Asset

def insert_all_sample_models():
    with Session(engine) as session:
        # Ensure a default asset exists
        asset_id = "production_line_A"
        existing_asset = session.execute(select(Asset).filter_by(id=asset_id)).scalar_one_or_none()
        if not existing_asset:
            print(f"Asset '{asset_id}' not found. Creating a dummy asset.")
            new_asset = Asset(id=asset_id, name="Production Line A", description="Sample Production Line")
            session.add(new_asset)
            session.commit()
            session.refresh(new_asset)

        models_to_insert = []

        # --- LightGBM Models ---
        lgbm_base_path = "demo/models/lgbm_energy_model_production_line_A"
        # Find the latest model, scaler, detector files by timestamp in their names
        lgbm_files = os.listdir(os.path.join(PROJECT_ROOT, lgbm_base_path))
        
        latest_model_version = None
        latest_model_file = None
        latest_scaler_file = None
        latest_scaler_cov_file = None

        for f in lgbm_files:
            if f.startswith("model_") and f.endswith(".joblib"):
                version_str = f.replace("model_", "").replace(".joblib", "")
                if not latest_model_version or version_str > latest_model_version:
                    latest_model_version = version_str
                    latest_model_file = f
                    latest_scaler_file = f.replace("model_", "scaler_")
                    latest_scaler_cov_file = f.replace("model_", "scaler_cov_")
        
        if latest_model_file:
            models_to_insert.append({
                "asset_id": asset_id,
                "model_type": "LightGBM",
                "model_version": latest_model_version,
                "model_path": os.path.join(lgbm_base_path, latest_model_file),
                "scaler_path": os.path.join(lgbm_base_path, latest_scaler_file) if os.path.exists(os.path.join(PROJECT_ROOT, lgbm_base_path, latest_scaler_file)) else None,
                "scaler_cov_path": os.path.join(lgbm_base_path, latest_scaler_cov_file) if os.path.exists(os.path.join(PROJECT_ROOT, lgbm_base_path, latest_scaler_cov_file)) else None,
                "description": f"LightGBM model (version {latest_model_version})",
                "is_active": True # Set this one as active for initial testing
            })

        # --- LSTM Model ---
        lstm_base_path = "demo/models/lstm_energy_model"
        lstm_model_file = os.path.join(lstm_base_path, "_model.pth.tar")
        if os.path.exists(os.path.join(PROJECT_ROOT, lstm_model_file)):
            models_to_insert.append({
                "asset_id": asset_id,
                "model_type": "LSTM",
                "model_version": datetime.now().strftime("%Y%m%d%H%M%S") + "_lstm_v1.0", # Ensure unique version
                "model_path": lstm_model_file,
                "scaler_path": None,
                "scaler_cov_path": None,
                "description": "LSTM model (default version)",
                "is_active": False
            })

        # --- TFT Model ---
        tft_base_path = "demo/models/tft_energy_model"
        tft_model_file = os.path.join(tft_base_path, "_model.pth.tar")
        if os.path.exists(os.path.join(PROJECT_ROOT, tft_model_file)):
            models_to_insert.append({
                "asset_id": asset_id,
                "model_type": "TFT",
                "model_version": datetime.now().strftime("%Y%m%d%H%M%S") + "_tft_v1.0", # Ensure unique version
                "model_path": tft_model_file,
                "scaler_path": None,
                "scaler_cov_path": None,
                "description": "TFT model (default version)",
                "is_active": False
            })
        
        # --- TiDE Model ---
        tide_base_path = "demo/models/tide_energy_model"
        tide_model_file = os.path.join(tide_base_path, "_model.pth.tar")
        if os.path.exists(os.path.join(PROJECT_ROOT, tide_model_file)):
            models_to_insert.append({
                "asset_id": asset_id,
                "model_type": "TiDE",
                "model_version": datetime.now().strftime("%Y%m%d%H%M%S") + "_tide_v1.0", # Ensure unique version
                "model_path": tide_model_file,
                "scaler_path": None,
                "scaler_cov_path": None,
                "description": "TiDE model (default version)",
                "is_active": False
            })

        # Insert models, checking for duplicates
        for model_data in models_to_insert:
            # Check if a model with the same asset_id, model_type, and model_version already exists
            existing_model = session.execute(
                select(Model).filter_by(
                    asset_id=model_data["asset_id"],
                    model_type=model_data["model_type"],
                    model_version=model_data["model_version"]
                )
            ).scalar_one_or_none()

            if existing_model:
                print(f"Model {model_data['model_type']} version {model_data['model_version']} for asset {model_data['asset_id']} already exists. Skipping.")
            else:
                new_model = Model(**model_data)
                session.add(new_model)
                session.commit()
                session.refresh(new_model)
                print(f"Successfully inserted {new_model.model_type} model with ID: {new_model.id} and Version: {new_model.model_version}")

if __name__ == "__main__":
    insert_all_sample_models()