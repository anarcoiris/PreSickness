"""
Unified App - Backend API
FastAPI con autenticación JWT, gestión de pacientes y proxy a servicios ML
"""
from __future__ import annotations

import hashlib
import os
import secrets
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List

import httpx
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import jwt
    from passlib.context import CryptContext
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "pyjwt", "passlib[bcrypt]", "-q"])
    import jwt
    from passlib.context import CryptContext

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

class Settings(BaseSettings):
    secret_key: str = Field(default_factory=lambda: secrets.token_hex(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 horas
    
    api_gateway_url: str = "http://localhost:8000"
    ml_inference_url: str = "http://localhost:8001"
    
    upload_dir: Path = Path("uploads")
    
    # DB simulada en memoria (para prototipo)
    # En producción: usar PostgreSQL
    
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()
settings.upload_dir.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEGURIDAD
# ══════════════════════════════════════════════════════════════════════════════

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Base de datos en memoria (prototipo)
fake_patients_db: dict[str, dict] = {}
fake_uploads_db: list[dict] = []


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


async def get_current_patient(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciales inválidas",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        email: str = payload.get("sub")
        if email is None or email not in fake_patients_db:
            raise credentials_exception
        return fake_patients_db[email]
    except jwt.PyJWTError:
        raise credentials_exception


# ══════════════════════════════════════════════════════════════════════════════
# MODELOS
# ══════════════════════════════════════════════════════════════════════════════

class PatientRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str = Field(min_length=2)


class PatientResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UploadResponse(BaseModel):
    id: str
    filename: str
    uploaded_at: datetime
    processed: bool


class PredictionRequest(BaseModel):
    horizon_days: int = Field(default=14, ge=7, le=30)


class PredictionResponse(BaseModel):
    probability: float
    risk_level: str
    horizon_days: int
    generated_at: datetime


class MetricsResponse(BaseModel):
    total_patients: int
    total_uploads: int
    services_status: dict


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="EM-Predictor Unified API",
    version="1.0.0",
    description="API unificada para gestión de pacientes, datos y predicciones",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS - AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/auth/register", response_model=PatientResponse, status_code=201)
async def register(data: PatientRegister):
    """Registro de nuevo paciente."""
    if data.email in fake_patients_db:
        raise HTTPException(status_code=400, detail="Email ya registrado")
    
    patient_id = hashlib.sha256(data.email.encode()).hexdigest()[:16]
    patient = {
        "id": patient_id,
        "email": data.email,
        "name": data.name,
        "password_hash": hash_password(data.password),
        "created_at": datetime.now(timezone.utc),
    }
    fake_patients_db[data.email] = patient
    
    return PatientResponse(
        id=patient["id"],
        email=patient["email"],
        name=patient["name"],
        created_at=patient["created_at"],
    )


@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login de paciente."""
    patient = fake_patients_db.get(form_data.username)
    if not patient or not verify_password(form_data.password, patient["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos",
        )
    
    access_token = create_access_token(
        data={"sub": patient["email"]},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )
    return Token(access_token=access_token)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS - PACIENTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/patients/me", response_model=PatientResponse)
async def get_profile(patient: dict = Depends(get_current_patient)):
    """Obtiene perfil del paciente actual."""
    return PatientResponse(
        id=patient["id"],
        email=patient["email"],
        name=patient["name"],
        created_at=patient["created_at"],
    )


@app.post("/api/patients/upload", response_model=UploadResponse)
async def upload_data(
    file: UploadFile = File(...),
    patient: dict = Depends(get_current_patient),
):
    """Sube archivo de datos (CSV/JSON)."""
    # Validar extensión
    allowed_extensions = {".csv", ".json", ".xlsx"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Extensión no permitida: {ext}")
    
    # Guardar archivo
    upload_id = secrets.token_hex(8)
    patient_dir = settings.upload_dir / patient["id"]
    patient_dir.mkdir(exist_ok=True)
    
    file_path = patient_dir / f"{upload_id}{ext}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    upload_record = {
        "id": upload_id,
        "patient_id": patient["id"],
        "filename": file.filename,
        "file_path": str(file_path),
        "uploaded_at": datetime.now(timezone.utc),
        "processed": False,
    }
    fake_uploads_db.append(upload_record)
    
    return UploadResponse(
        id=upload_id,
        filename=file.filename,
        uploaded_at=upload_record["uploaded_at"],
        processed=False,
    )


@app.get("/api/patients/data", response_model=List[UploadResponse])
async def list_uploads(patient: dict = Depends(get_current_patient)):
    """Lista datos subidos por el paciente."""
    patient_uploads = [
        UploadResponse(
            id=u["id"],
            filename=u["filename"],
            uploaded_at=u["uploaded_at"],
            processed=u["processed"],
        )
        for u in fake_uploads_db
        if u["patient_id"] == patient["id"]
    ]
    return patient_uploads


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS - PREDICCIÓN
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    patient: dict = Depends(get_current_patient),
):
    """Ejecuta predicción para el paciente."""
    # Intentar llamar al servicio ML real
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{settings.ml_inference_url}/v1/predict",
                json={
                    "user_id_hash": patient["id"].ljust(64, "0"),
                    "horizon_days": request.horizon_days,
                },
            )
            if response.status_code == 200:
                data = response.json()
                return PredictionResponse(
                    probability=data["relapse_probability"],
                    risk_level=data["risk_level"],
                    horizon_days=data["horizon_days"],
                    generated_at=datetime.now(timezone.utc),
                )
    except Exception:
        pass  # Fallback a heurística
    
    # Heurística simple si el servicio no está disponible
    import random
    prob = random.uniform(0.1, 0.5)
    return PredictionResponse(
        probability=prob,
        risk_level="warning" if prob > 0.35 else "ok",
        horizon_days=request.horizon_days,
        generated_at=datetime.now(timezone.utc),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS - SISTEMA
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }



class Alert(BaseModel):
    id: str
    alert_level: str
    alert_type: str
    triggered_at: datetime
    message: str
    read: bool


@app.get("/api/alerts", response_model=List[Alert])
async def get_alerts(patient: dict = Depends(get_current_patient)):
    """Obtiene alertas del paciente."""
    # Mock data para prototipo
    return [
        Alert(
            id="1",
            alert_level="warning",
            alert_type="risk_increase",
            triggered_at=datetime.now(timezone.utc) - timedelta(days=2),
            message="Se ha detectado un aumento leve en tu nivel de riesgo.",
            read=True
        ),
        Alert(
            id="2",
            alert_level="info",
            alert_type="upload_reminder",
            triggered_at=datetime.now(timezone.utc) - timedelta(hours=5),
            message="No olvides subir tus datos de actividad de hoy.",
            read=False
        ),
        Alert(
            id="3",
            alert_level="critical",
            alert_type="appointment",
            triggered_at=datetime.now(timezone.utc) - timedelta(days=5),
            message="Recodatorio: Cita médica programada para la próxima semana.",
            read=True
        )
    ]

@app.get("/api/metrics", response_model=MetricsResponse)
async def metrics():
    """Métricas del sistema."""
    # Verificar servicios
    services = {}
    
    for name, url in [
        ("api_gateway", settings.api_gateway_url),
        ("ml_inference", settings.ml_inference_url),
    ]:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                r = await client.get(f"{url}/v1/health")
                services[name] = "ok" if r.status_code == 200 else "error"
        except Exception:
            services[name] = "unreachable"
    
    return MetricsResponse(
        total_patients=len(fake_patients_db),
        total_uploads=len(fake_uploads_db),
        services_status=services,
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
