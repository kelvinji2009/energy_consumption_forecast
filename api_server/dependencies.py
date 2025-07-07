
import os
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.orm import Session

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from database.database import engine, ApiKey

# Dependency to get DB session
def get_db():
    with Session(engine) as session:
        yield session

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """
    Verifies the provided API key by comparing it against stored bcrypt hashes.
    """
    api_key_str = credentials.credentials
    
    # 1. Fetch all active API key hashes from the database
    stmt = select(ApiKey.key_hash).where(ApiKey.is_active == True)
    active_key_hashes = db.execute(stmt).scalars().all()

    # 2. Check the provided key against each hash
    for key_hash in active_key_hashes:
        if bcrypt.checkpw(api_key_str.encode('utf-8'), key_hash.encode('utf-8')):
            return api_key_str # Return the valid key string

    # 3. If no match is found, raise an error
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, 
        detail="Invalid or inactive API Key"
    )
