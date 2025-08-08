import os
import sys
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from datetime import datetime, timezone

# Add project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from database.database import Asset, Base
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/energy_forecast_db")

DEFAULT_ASSET_ID = "production_line_A"
DEFAULT_ASSET_NAME = "Production Line A"

def main():
    """
    Checks if the default asset exists and creates it if it doesn't.
    This ensures that a new environment always has at least one asset to work with.
    """
    print("--- Running initial data seeder ---")
    try:
        engine = create_engine(DATABASE_URL)
        with Session(engine) as session:
            # Check if the default asset already exists
            stmt = select(Asset).where(Asset.id == DEFAULT_ASSET_ID)
            existing_asset = session.execute(stmt).scalar_one_or_none()

            if existing_asset:
                print(f"Default asset '{DEFAULT_ASSET_ID}' already exists. Skipping creation.")
            else:
                print(f"Creating default asset: '{DEFAULT_ASSET_ID}'")
                new_asset = Asset(
                    id=DEFAULT_ASSET_ID,
                    name=DEFAULT_ASSET_NAME,
                    description="Default asset created automatically on initial setup.",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                session.add(new_asset)
                session.commit()
                print("Default asset created successfully.")

    except Exception as e:
        print(f"An error occurred during data seeding: {e}")
        # In a real production environment, you might want to handle this more robustly.
        sys.exit(1)

    print("--- Initial data seeder finished ---")

if __name__ == "__main__":
    main()