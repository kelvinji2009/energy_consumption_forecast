import os
import bcrypt
from fastapi import Depends, HTTPException, status, Header
from sqlalchemy import select
from sqlalchemy.orm import Session
from typing import Optional

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from database.database import engine, ApiKey

# Dependency to get DB session
def get_db():
    with Session(engine) as session:
        yield session

async def verify_api_key(x_api_key: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """
    Verifies the provided API key from the 'X-API-Key' header.
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is missing. Please include it in the 'X-API-Key' header."
        )

    # 1. Fetch all active API key hashes from the database
    stmt = select(ApiKey.key_hash).where(ApiKey.is_active == True)
    active_key_hashes = db.execute(stmt).scalars().all()

    if not active_key_hashes:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No active API keys configured in the system."
        )

    # 2. Check the provided key against each hash
    for key_hash in active_key_hashes:
        if bcrypt.checkpw(x_api_key.encode('utf-8'), key_hash.encode('utf-8')):
            return x_api_key # Return the valid key string

    # 3. If no match is found, raise an error
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated" # Simplified error message
    )