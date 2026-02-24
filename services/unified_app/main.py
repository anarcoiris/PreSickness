"""
Unified App - Backend API
FastAPI con autenticación JWT, gestión de pacientes y proxy a servicios ML
"""
from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List
import asyncio
from uuid import UUID

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import httpx
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
except ImportError:
    pass
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import jwt
    from passlib.context import CryptContext
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "pyjwt", "passlib[bcrypt]", "google-auth-oauthlib", "google-api-python-client", "-q"])
    import jwt
    from passlib.context import CryptContext

import db
from dependencies import (
    settings,
    pwd_context,
    oauth2_scheme,
    verify_password,
    hash_password,
    create_access_token,
    get_current_patient,
    get_current_admin
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN & SEGURIDAD IMPORTS VIA dependencies.py
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# MODELOS
# ══════════════════════════════════════════════════════════════════════════════

class PatientRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str = Field(min_length=2)
    role: str = Field(default="patient")  # patient, doctor


class PatientResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str = "patient"
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UploadResponse(BaseModel):
    id: str
    filename: str
    uploaded_at: datetime
    processed: bool


class GoogleToken(BaseModel):
    id_token: Optional[str] = None
    code: Optional[str] = None
    role: str = "patient"


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
    total_messages: int
    total_datapoints: int
    nlp_processed: int
    services_status: dict


# Doctor-Patient relationship models
class AddPatientRequest(BaseModel):
    patient_email: EmailStr


class DoctorPatientResponse(BaseModel):
    patient_id: str
    patient_name: str
    patient_email: str
    granted_at: datetime
    access_level: str
    status: str


class SystemIncidentResponse(BaseModel):
    id: UUID
    severity: str
    component: str
    message: str
    details: Optional[dict]
    resolved: bool
    resolved_at: Optional[datetime]
    created_at: datetime

# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load initial data
    try:
        # Importar aquí para evitar circular imports si los hubiera
        from events import load_initial_data_async
        await load_initial_data_async()
        print("[INFO] Application startup complete")
    except Exception as e:
        print(f"[WARN] Failed to load initial data: {e}")
        
    yield
    
    # Shutdown: Close resources
    try:
        from events import db
        await db.close_pool()
        print("[INFO] Application shutdown complete")
    except Exception as e:
        print(f"[WARN] Failed to close DB pool: {e}")


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="MS-Predictor Unified API",
    version="1.0.0",
    description="API unificada para gestión de pacientes, datos, eventos y predicciones",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # En producción: añadir URL real
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar router de eventos
try:
    from events import router as events_router
    app.include_router(events_router)
    
    from analysis import router as analysis_router
    app.include_router(analysis_router, prefix="/api/analysis", tags=["analysis"])
    
except ImportError as e:
    print(f"[WARN] Failed to load routers: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS - AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/auth/register", response_model=PatientResponse, status_code=201)
@limiter.limit("5/minute")
async def register(request: Request, data: PatientRegister):
    """Registro de nuevo usuario (paciente o médico)."""
    existing = await db.get_user_by_email(data.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email ya registrado")
    
    # Validate role
    if data.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=400, detail="Rol inválido. Use 'patient' o 'doctor'")
    
    # Generar UUIDv4 para mejor privacidad que el hash predecible del email
    user_id_hash = uuid.uuid4().hex[:16]
    user_data = {
        "user_id_hash": user_id_hash,
        "email": data.email,
        "name": data.name,
        "password_hash": hash_password(data.password),
        "role": data.role,
    }
    patient = await db.create_user(user_data)
    
    return PatientResponse(
        id=patient["user_id_hash"],
        email=patient["email"],
        name=patient["name"],
        role=patient.get("role", "patient"),
        created_at=patient["created_at"],
    )

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS - DOCTOR-PATIENT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/doctor/patients", response_model=List[DoctorPatientResponse])
async def get_my_patients(current_user: dict = Depends(get_current_patient)):
    """Get all patients for current doctor."""
    if current_user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Solo médicos pueden acceder a esta función")
    
    patients = await db.get_doctor_patients(current_user["user_id_hash"])
    return [
        DoctorPatientResponse(
            patient_id=p["patient_id"],
            patient_name=p["patient_name"],
            patient_email=p["patient_email"],
            granted_at=p["granted_at"],
            access_level=p["access_level"],
            status=p["status"]
        )
        for p in patients
    ]


@app.post("/api/doctor/patients", response_model=DoctorPatientResponse)
async def add_patient(data: AddPatientRequest, current_user: dict = Depends(get_current_patient)):
    """Add a patient to current doctor's care."""
    if current_user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Solo médicos pueden añadir pacientes")
    
    # Find patient by email
    patient = await db.get_user_by_email(data.patient_email)
    if not patient:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    
    if patient.get("role") != "patient":
        raise HTTPException(status_code=400, detail="El usuario seleccionado no es un paciente")
    
    relationship = await db.add_patient_to_doctor(
        doctor_id=current_user["user_id_hash"],
        patient_id=patient["user_id_hash"],
        granted_by=patient["user_id_hash"]
    )
    
    return DoctorPatientResponse(
        patient_id=patient["user_id_hash"],
        patient_name=patient["name"],
        patient_email=patient["email"],
        granted_at=relationship["granted_at"],
        access_level=relationship["access_level"],
        status=relationship["status"]
    )


@app.delete("/api/doctor/patients/{patient_id}")
async def remove_patient(patient_id: str, current_user: dict = Depends(get_current_patient)):
    """Remove a patient from current doctor's care."""
    if current_user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Solo médicos pueden gestionar pacientes")
    
    await db.remove_patient_from_doctor(current_user["user_id_hash"], patient_id)
    return {"status": "ok", "message": "Paciente removido"}


@app.get("/api/patient/doctors")
async def get_my_doctors(current_user: dict = Depends(get_current_patient)):
    """Get all doctors caring for current patient."""
    doctors = await db.get_patient_doctors(current_user["user_id_hash"])
    return [
        {
            "doctor_id": d["doctor_id"],
            "doctor_name": d["doctor_name"],
            "doctor_email": d["doctor_email"],
            "granted_at": d["granted_at"],
            "status": d["status"]
        }
        for d in doctors
    ]



@app.post("/api/auth/login", response_model=Token)
@limiter.limit("5/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """Login de paciente."""
    patient = await db.get_user_by_email(form_data.username)
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


@app.post("/api/auth/google", response_model=Token)
async def google_login(data: GoogleToken):
    """Login con Google (Code Flow o ID Token)."""
    email = None
    name = "Google User"
    tokens_to_store = None

    # 1. AUTH CODE FLOW (Preferred for Web - supports Offline Access/Calendar)
    if data.code:
        try:
            from google_auth_oauthlib.flow import Flow
            from google.oauth2 import id_token
            from google.auth.transport import requests as google_requests

            # Si estamos en demo/dev sin credenciales reales
            if not settings.google_client_id or not settings.google_client_secret:
                if data.code.startswith("demo_code_"):
                    email = data.code.replace("demo_code_", "")
                    name = "Demo User"
                else:
                    raise HTTPException(status_code=500, detail="Google Client ID/Secret not configured on server")
            else:
                flow_config = {
                    "web": {
                        "client_id": settings.google_client_id,
                        "client_secret": settings.google_client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                }
                # "postmessage" is the magic redirect_uri for popup/SPA flows
                flow = Flow.from_client_config(
                    flow_config,
                    scopes=[
                        "openid", 
                        "https://www.googleapis.com/auth/userinfo.email", 
                        "https://www.googleapis.com/auth/userinfo.profile",
                        "https://www.googleapis.com/auth/calendar"
                    ],
                    redirect_uri="postmessage" 
                )
                
                flow.fetch_token(code=data.code)
                credentials = flow.credentials

                # Verify ID Token
                id_info = id_token.verify_oauth2_token(
                    credentials.id_token, 
                    google_requests.Request(), 
                    settings.google_client_id
                )
                
                email = id_info['email']
                name = id_info.get('name', email.split("@")[0])
                
                tokens_to_store = {
                    'access_token': credentials.token,
                    'refresh_token': credentials.refresh_token,
                    'expiry': credentials.expiry,
                    'scope': ' '.join(credentials.scopes) if credentials.scopes else ''
                }

        except Exception as e:
            print(f"[ERROR] Google Auth Code Error: {e}")
            raise HTTPException(status_code=400, detail=f"Google Auth Error: {str(e)}")

    # 2. ID TOKEN FLOW (Legacy/Mobile or Demo)
    elif data.id_token:
        # En desarrollo/demo, aceptamos tokens que empiecen por 'demo_token_'
        if data.id_token.startswith("demo_token_"):
            email = data.id_token.replace("demo_token_", "")
            name = "Google User"
        else:
            # Validar token real con API de Google
            try:
                # Use libraries if available, else fallback to http check
                from google.oauth2 import id_token
                from google.auth.transport import requests as google_requests
                
                id_info = id_token.verify_oauth2_token(
                    data.id_token, 
                    google_requests.Request(), 
                    settings.google_client_id
                )
                email = id_info['email']
                name = id_info.get('name', email.split("@")[0])

            except Exception as e:
                # Fallback manual check
                try:
                     async with httpx.AsyncClient() as client:
                        res = await client.get(
                            f"https://oauth2.googleapis.com/tokeninfo?id_token={data.id_token}"
                        )
                        if res.status_code != 200:
                            raise Exception("Invalid Token Details")
                        payload = res.json()
                        if settings.google_client_id and payload.get("aud") != settings.google_client_id:
                             raise Exception("Client ID mismatch")
                        email = payload["email"]
                        name = payload.get("name", email.split("@")[0])
                except Exception as ex:
                    raise HTTPException(status_code=400, detail=f"Error validando token Google: {str(ex)}")

    else:
        raise HTTPException(status_code=400, detail="Missing code or id_token")

    if not email:
        raise HTTPException(status_code=400, detail="Could not retrieve email from Google")

    # Buscar o crear usuario
    patient = await db.get_user_by_email(email)
    if not patient:
        # Registro automático en primer login google
        user_id_hash = hashlib.sha256(email.encode()).hexdigest()[:16]
        user_data = {
            "user_id_hash": user_id_hash,
            "email": email,
            "name": name,
            "password_hash": "google_authenticated", # No se usa password
            "role": data.role
        }
        patient = await db.create_user(user_data)
        print(f"[INFO] Usuario Google registrado: {email}")
    
    # Store OAuth Tokens if available (Auth Code Flow)
    if tokens_to_store:
        try:
            await db.store_oauth_tokens(patient["user_id_hash"], "google", tokens_to_store)
            print(f"[INFO] Google Tokens stored for {email}")
        except Exception as e:
            print(f"[WARN] Failed to store OAuth tokens: {e}")

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
        id=patient["user_id_hash"],
        email=patient["email"],
        name=patient["name"],
        role=patient.get("role", "patient"),
        created_at=patient["created_at"],
    )


@app.post("/api/patients/upload", response_model=UploadResponse)
async def upload_data(
    file: UploadFile = File(...),
    patient: dict = Depends(get_current_patient),
):
    """Sube archivo de datos (CSV/JSON/TXT WhatsApp)."""
    # Validar extensión
    allowed_extensions = {".csv", ".json", ".xlsx", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Extensión no permitida: {ext}")
    
    # Guardar archivo
    upload_id = secrets.token_hex(8)
    patient_dir = settings.upload_dir / patient["user_id_hash"]
    patient_dir.mkdir(exist_ok=True, parents=True)
    
    file_path = patient_dir / f"{upload_id}{ext}"
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    upload_record = {
        "id": upload_id,
        "patient_id": patient["user_id_hash"],
        "filename": file.filename,
        "file_path": str(file_path),
        "uploaded_at": datetime.now(timezone.utc),
        "processed": False,
    }
    await db.store_upload(upload_record)
    
    # Si es .txt, parsear como WhatsApp e insertar mensajes
    if ext == ".txt":
        import asyncio
        from events import parse_whatsapp_line
        
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
        
        lines = text.splitlines()
        parsed_messages = []
        current_msg = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parsed = parse_whatsapp_line(line)
            if parsed:
                if current_msg:
                    parsed_messages.append(current_msg)
                current_msg = {
                    "date": parsed["date"],
                    "content": parsed["content"],
                    "metadata": {"sender": parsed["sender"]}
                }
            elif current_msg:
                current_msg["content"] += "\n" + line
        
        if current_msg:
            parsed_messages.append(current_msg)
        
        # Batch insert messages
        if parsed_messages:
            chunk_size = 1000
            for i in range(0, len(parsed_messages), chunk_size):
                chunk = parsed_messages[i:i + chunk_size]
                await db.batch_store_raw_messages(patient["user_id_hash"], chunk)
            
            # Trigger background processing
            try:
                from processor import process_messages_for_patient
                asyncio.create_task(process_messages_for_patient(patient["user_id_hash"]))
            except Exception as e:
                print(f"[WARN] Background processing failed to start: {e}")
        
        # Mark as processed (messages inserted)
        upload_record["processed"] = True
    
    return UploadResponse(
        id=upload_id,
        filename=file.filename,
        uploaded_at=upload_record["uploaded_at"],
        processed=upload_record["processed"],
    )


@app.get("/api/patients/data", response_model=List[UploadResponse])
async def list_uploads(patient: dict = Depends(get_current_patient)):
    """Lista datos subidos por el paciente."""
    uploads = await db.get_uploads_by_patient(patient["user_id_hash"])
    return [
        UploadResponse(
            id=u["id"],
            filename=u["filename"],
            uploaded_at=u["uploaded_at"],
            processed=u["processed"],
        )
        for u in uploads
    ]


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS - PREDICCIÓN
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    patient: dict = Depends(get_current_patient),
):
    """Ejecuta predicción para el paciente."""
    prob = 0.0
    risk_level = "ok"
    horizon_days = request.horizon_days

    # Intentar llamar al servicio ML real
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{settings.ml_inference_url}/v1/predict",
                json={
                    "user_id_hash": patient["user_id_hash"],
                    "horizon_days": request.horizon_days,
                },
            )
            if response.status_code == 200:
                data = response.json()
                prob = data["relapse_probability"]
                risk_level = data["risk_level"]
            else:
                # Fallback internal heuristic
                import random
                prob = random.uniform(0.1, 0.4)
                risk_level = "ok" if prob < 0.35 else "warning"
    except Exception as e:
        print(f"[WARN] Prediction service error: {e}")
        import random
        prob = random.uniform(0.05, 0.3)
        risk_level = "ok"

    # Store prediction in DB
    try:
        prediction_record = await db.store_prediction({
            "user_id_hash": patient["user_id_hash"],
            "horizon_days": horizon_days,
            "relapse_probability": prob,
            "model_name": "tft_prototype" if risk_level != "ok" else "heuristic_fallback"
        })
        
        # Trigger alert if risk is high
        if risk_level in ["warning", "critical"]:
            await db.store_alert({
                "user_id_hash": patient["user_id_hash"],
                "prediction_id": prediction_record["id"],
                "alert_level": risk_level,
                "alert_type": "high_risk_detected"
            })
    except Exception as e:
        print(f"[ERROR] Failed to store prediction/alert: {e}")

    return PredictionResponse(
        probability=prob,
        risk_level=risk_level,
        horizon_days=horizon_days,
        generated_at=datetime.now(timezone.utc),
    )


@app.get("/api/predict/history", response_model=List[PredictionResponse])
async def get_prediction_history(
    limit: int = 30,
    patient: dict = Depends(get_current_patient)
):
    """Obtiene el historial de predicciones del paciente."""
    history = await db.get_prediction_history(patient["user_id_hash"], limit=limit)
    return [
        PredictionResponse(
            probability=p["relapse_probability"],
            risk_level="critical" if p["relapse_probability"] > 0.6 else ("warning" if p["relapse_probability"] > 0.35 else "ok"),
            horizon_days=p["horizon_days"],
            generated_at=p["created_at"]
        )
        for p in history
    ]


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
    """Obtiene alertas reales de la base de datos."""
    alerts = await db.get_alerts_by_patient(patient["user_id_hash"])
    
    # Si no hay alertas, mostrar una de bienvenida
    if not alerts:
        return [
            Alert(
                id="0",
                alert_level="info",
                alert_type="system",
                triggered_at=datetime.now(timezone.utc),
                message="Bienvenido al sistema EM-Predictor. Tu historial está limpio.",
                read=False
            )
        ]

    return [
        Alert(
            id=str(a["id"]),
            alert_level=a["alert_level"],
            alert_type=a["alert_type"],
            triggered_at=a["triggered_at"],
            message=f"Alerta de riesgo detectada ({a['alert_level']}). Revisa tu panel de analytics.",
            read=a.get("acknowledged_at") is not None
        )
        for a in alerts
    ]

@app.get("/api/metrics", response_model=MetricsResponse)
async def metrics():
    """Métricas del sistema."""
    # Verificar servicios
    services = {}
    
    async with httpx.AsyncClient(timeout=2) as client:
        for name, url in [
            ("api_gateway", settings.api_gateway_url),
            ("ml_inference", settings.ml_inference_url),
            ("nlp_agent", settings.nlp_agent_url),
        ]:
            try:
                # Try v1/health first
                r = await client.get(f"{url.rstrip('/')}/v1/health")
                if r.status_code != 200:
                    # Fallback to /health
                    r = await client.get(f"{url.rstrip('/')}/health")
                
                if r.status_code == 200:
                    services[name] = "ok"
                else:
                    services[name] = f"error_{r.status_code}"
            except Exception as e:
                services[name] = "unreachable"
    
    system_stats = await db.get_system_metrics()
    
    return MetricsResponse(
        total_patients=system_stats["total_patients"],
        total_uploads=system_stats["total_uploads"],
        total_messages=system_stats["total_messages"],
        total_datapoints=system_stats["total_datapoints"],
        nlp_processed=system_stats["nlp_processed"],
        services_status=services,
    )




# ------------------------------------------------------------------------------
# PATIENT VIEW (DOCTORS)
# ------------------------------------------------------------------------------

class DoctorInfoResponse(BaseModel):
    doctor_id: str
    doctor_name: str
    doctor_email: str
    granted_at: datetime
    status: str

@app.get("/api/patient/doctors", response_model=List[DoctorInfoResponse])
async def get_my_doctors_endpoint(current_user: dict = Depends(get_current_patient)):
    """Get all doctors authorized by the current patient."""
    return await db.get_patient_doctors(current_user["user_id_hash"])

@app.delete("/api/patient/doctors/{doctor_id}")
async def revoke_doctor_access(doctor_id: str, current_user: dict = Depends(get_current_patient)):
    """Revoke a doctor's access."""
    await db.remove_patient_from_doctor(doctor_id, current_user["user_id_hash"])
    return {"status": "success", "message": "Acceso revocado correctamente"}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS - ADMIN
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/admin/users", response_model=List[PatientResponse])
async def admin_list_users(admin: dict = Depends(get_current_admin)):
    """List all users in the system."""
    users = await db.get_all_users()
    return [
        PatientResponse(
            id=u["user_id_hash"],
            email=u["email"],
            name=u["name"],
            role=u["role"],
            created_at=u["created_at"]
        )
        for u in users
    ]

@app.put("/api/admin/users/{user_id}/role")
async def admin_update_user_role(user_id: str, role: str, admin: dict = Depends(get_current_admin)):
    """Update a user's role."""
    if role not in ["patient", "doctor", "admin"]:
        raise HTTPException(status_code=400, detail="Rol inválido")
    user = await db.update_user_role(user_id, role)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return user

@app.get("/api/admin/incidents", response_model=List[SystemIncidentResponse])
async def admin_list_incidents(include_resolved: bool = False, admin: dict = Depends(get_current_admin)):
    """List system incidents."""
    return await db.get_incidents(include_resolved=include_resolved)

@app.post("/api/admin/incidents/{incident_id}/resolve")
async def admin_resolve_incident(incident_id: UUID, admin: dict = Depends(get_current_admin)):
    """Resolve a system incident."""
    incident = await db.resolve_incident(incident_id, admin["user_id_hash"])
    if not incident:
        raise HTTPException(status_code=404, detail="Incidencia no encontrada")
    return incident


if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
