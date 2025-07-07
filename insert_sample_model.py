import os
import sys
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(PROJECT_ROOT)

from database.database import engine, Model, Asset

def insert_sample_model():
    with Session(engine) as session:
        # Check if asset exists, if not, create a dummy one
        asset_id = "production_line_A"
        existing_asset = session.execute(select(Asset).filter_by(id=asset_id)).scalar_one_or_none()
        if not existing_asset:
            print(f"Asset '{asset_id}' not found. Creating a dummy asset.")
            new_asset = Asset(id=asset_id, name="Production Line A", description="Sample Production Line")
            session.add(new_asset)
            session.commit()
            session.refresh(new_asset)

        # Define paths to the sample model files
        model_base_path = "demo/models/lgbm_energy_model"
        model_path = os.path.join(model_base_path, "model.joblib")
        scaler_path = os.path.join(model_base_path, "scaler.joblib")
        scaler_cov_path = os.path.join(model_base_path, "scaler_cov.joblib")

        # Check if model files actually exist
        if not os.path.exists(os.path.join(PROJECT_ROOT, model_path)):
            print(f"Error: Model file not found at {model_path}. Please ensure the demo models are present.")
            return

        # Deactivate any existing active models for this asset
        session.query(Model).filter(
            Model.asset_id == asset_id,
            Model.is_active == True
        ).update({'is_active': False}, synchronize_session=False)
        session.commit()

        # Insert the new model record
        new_model = Model(
            asset_id=asset_id,
            model_type="LightGBM",
            model_version=datetime.now().strftime("%Y%m%d%H%M%S") + "_initial_lgbm",
            model_path=model_path,
            scaler_path=scaler_path,
            scaler_cov_path=scaler_cov_path,
            is_active=True,
            description="Initial LightGBM model for Production Line A",
            metrics={'mape': 5.0, 'rmse': 10.0} # Sample metrics
        )
        session.add(new_model)
        session.commit()
        session.refresh(new_model)

        print(f"Successfully inserted sample model with ID: {new_model.id} and Version: {new_model.model_version}")

if __name__ == "__main__":
    insert_sample_model()
