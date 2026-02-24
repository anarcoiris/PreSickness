from __future__ import annotations
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Depends, status, Header
from fastapi.security import OAuth2PasswordBearer
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import jwt
from passlib.context import CryptContext

import db

# ------------------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------------------

class Settings(BaseSettings):
    secret_key: str = Field(default_factory=lambda: secrets.token_hex(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours
    
    api_gateway_url: str = "http://localhost:8000"
    ml_inference_url: str = "http://localhost:8001"
    nlp_agent_url: str = "http://localhost:8002"
    
    google_client_id: str = "" # Set in .env
    google_client_secret: str = "" # Set in .env
    
    upload_dir: Path = Path("uploads")
    
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

settings = Settings()
settings.upload_dir.mkdir(exist_ok=True)


# ------------------------------------------------------------------------------
# SECURITY
# ------------------------------------------------------------------------------

# Force pbkdf2_sha256 to avoid bcrypt issues on this Windows environment
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


async def get_current_patient(
    token: str = Depends(oauth2_scheme),
    x_patient_id: Optional[str] = Header(None, alias="X-Patient-ID")
) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciales inválidas",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        
        user = await db.get_user_by_email(email)
        if user is None:
            raise credentials_exception
            
        # Doctor impersonation Logic
        if x_patient_id and user.get("role") == "doctor":
            has_access = await db.doctor_has_patient_access(user["user_id_hash"], x_patient_id)
            if has_access:
                impersonated_user = await db.get_user_by_id(x_patient_id)
                if impersonated_user:
                    # Return the PATIENT user so endpoints see the patient's ID
                    return impersonated_user
            else:
                raise HTTPException(status_code=403, detail="No tienes acceso a este paciente")
                
        return user
    except jwt.PyJWTError:
        raise credentials_exception


async def get_current_admin(user: dict = Depends(get_current_patient)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Se requieren privilegios de administrador"
        )
    return user
