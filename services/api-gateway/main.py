"""
API Gateway - Implementación inicial (Agent Backus)
Gestiona ingesta segura de datapoints y publica eventos para el pipeline.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

import asyncpg
import orjson
import redis.asyncio as redis
from aiokafka import AIOKafkaProducer
from cryptography.fernet import Fernet
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    status,
    Header,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
import jwt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from google_auth_oauthlib.flow import Flow
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api-gateway")


class Settings(BaseSettings):
    """Configuración centralizada (Agent Archie)."""

    db_dsn: str = Field(
        default="postgresql://emuser:changeme@postgres:5432/empredictor",
        env="DB_DSN",
    )
    redis_url: str = Field(default="redis://redis:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    kafka_bootstrap_servers: Optional[str] = Field(default=None, env="KAFKA_BROKERS")
    kafka_topic: str = Field(default="ingest.datapoints.v1")
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    device_token_prefix: str = Field(default="device_token")
    allowed_clock_skew_seconds: int = Field(default=300)
    max_payload_age_minutes: int = Field(default=60)
    cache_ttl_seconds: int = Field(default=3600)
    
    jwt_secret: str = Field(default="dev_secret_key_123", env="JWT_SECRET")
    google_client_id: Optional[str] = Field(default=None, env="GOOGLE_CLIENT_ID")
    google_client_secret: Optional[str] = Field(default=None, env="GOOGLE_CLIENT_SECRET")

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()

# === Security primitives ===
fernet_key = (
    settings.encryption_key.encode()
    if settings.encryption_key
    else Fernet.generate_key()
)
if not settings.encryption_key:
    logger.warning(
        "ENCRYPTION_KEY no definido. Se generó uno efímero (solo válido en dev)."
    )
cipher = Fernet(fernet_key)

# === FastAPI App ===
app = FastAPI(title="EM Predictor API Gateway", version="0.2.0")
security = HTTPBearer(auto_error=True)

# Globals inicializados en startup
db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None
kafka_producer: Optional[AIOKafkaProducer] = None


# === Schemas ===
class EncryptedEmbedding(BaseModel):
    """Embedding encriptado proporcionado por el cliente."""

    embedding_encrypted: str = Field(..., min_length=8)
    embedding_dim: int = Field(default=768, ge=32, le=2048)
    salt: str = Field(..., min_length=8, max_length=128)


class NumericFeatures(BaseModel):
    """Features pre-calculadas en cliente (Agent Droid)."""

    sentiment_score: float = Field(ge=-1.0, le=1.0)
    avg_sentence_len: float = Field(ge=0)
    type_token_ratio: float = Field(ge=0, le=1.0)
    num_messages: int = Field(ge=0)
    avg_response_latency_sec: Optional[float] = Field(default=None, ge=0)
    steps: Optional[int] = Field(default=None, ge=0)
    hr_mean: Optional[float] = Field(default=None, ge=0)
    sleep_hours: Optional[float] = Field(default=None, ge=0)
    voice_pitch_mean: Optional[float] = Field(default=None)
    voice_speech_rate: Optional[float] = Field(default=None)
    apps_social_minutes: Optional[int] = Field(default=None, ge=0)


class DataPoint(BaseModel):
    """Payload completo del cliente."""

    user_id_hash: str = Field(min_length=64, max_length=64)
    timestamp: datetime
    device_id_hash: str = Field(min_length=64, max_length=64)
    embedding: EncryptedEmbedding
    numeric_features: NumericFeatures
    signature: str = Field(min_length=64, max_length=128, description="HMAC hex")

    @field_validator("timestamp")
    @classmethod
    def ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp debe incluir zona horaria (UTC).")
        return value.astimezone(timezone.utc)


# === Helpers ===
async def verify_device(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Valida token de dispositivo almacenado en Redis o JWT de usuario web."""
    token = credentials.credentials
    try:
        # Intenta verificar si es un JWT de usuario web
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return payload.get("sub")
    except jwt.PyJWTError:
        pass # fallback to device token in redis

    if not redis_client:
        raise HTTPException(status_code=503, detail="Auth service unavailable")

    token_key = f"{settings.device_token_prefix}:{token}"
    device_id = await redis_client.get(token_key)
    if not device_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    device_id_hash = (
        device_id.decode() if isinstance(device_id, (bytes, bytearray)) else str(device_id)
    )
    return device_id_hash
    
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token", auto_error=False)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        user_id_hash = payload.get("sub")
        if not user_id_hash:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
            
        # Optional: Verify user exists in db
        if db_pool:
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow("SELECT id FROM users WHERE user_id_hash = $1", user_id_hash)
                if not row:
                    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return {"user_id_hash": user_id_hash, "email": payload.get("email")}
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")


async def verify_signature(data: DataPoint) -> None:
    """Verifica integridad mediante HMAC con secreto de dispositivo."""
    device_secret = await get_device_secret(data.device_id_hash)

    canonical = "|".join(
        [
            data.user_id_hash,
            data.device_id_hash,
            data.timestamp.isoformat(),
            data.embedding.salt,
        ]
    )
    expected_sig = hmac.new(
        device_secret.encode(),
        canonical.encode(),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected_sig, data.signature):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid signature"
        )


async def get_device_secret(device_id_hash: str) -> str:
    """Obtiene secreto del dispositivo desde Postgres."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT secret FROM devices WHERE device_id_hash = $1",
            device_id_hash,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Device not registered")
        return row["secret"]


def validate_timestamp(timestamp: datetime) -> None:
    """Valida que el payload no sea demasiado viejo o futuro."""
    now = datetime.now(timezone.utc)
    if timestamp > now + timedelta(seconds=settings.allowed_clock_skew_seconds):
        raise HTTPException(status_code=422, detail="timestamp en el futuro")
    if timestamp < now - timedelta(minutes=settings.max_payload_age_minutes):
        raise HTTPException(status_code=422, detail="timestamp vencido")


def compute_quality_score(features: NumericFeatures) -> float:
    """Heurística simple para QA inicial (Agent Guard)."""
    score = 1.0
    if features.num_messages == 0:
        score -= 0.2
    if features.sentiment_score == 0:
        score -= 0.1
    if features.steps is None and features.sleep_hours is None:
        score -= 0.1
    return max(score, 0.1)


async def store_datapoint(data: DataPoint, quality_score: float) -> None:
    """Inserta datapoint en Timescale y actualiza cache."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database unavailable")
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO datapoints (
                time,
                user_id_hash,
                device_id_hash,
                embedding_encrypted,
                embedding_dim,
                embedding_salt,
                numeric_features,
                data_quality_score
            ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
            """,
            data.timestamp,
            data.user_id_hash,
            data.device_id_hash,
            data.embedding.embedding_encrypted,
            data.embedding.embedding_dim,
            data.embedding.salt,
            json.dumps(data.numeric_features.model_dump(exclude_none=True)),
            quality_score,
        )

    if redis_client:
        cache_key = f"features:{data.user_id_hash}:latest"
        await redis_client.setex(
            cache_key,
            settings.cache_ttl_seconds,
            json.dumps(
                {
                    **data.numeric_features.model_dump(exclude_none=True),
                    "timestamp": data.timestamp.isoformat(),
                    "quality_score": quality_score,
                }
            ),
        )


async def publish_event(payload: Dict[str, Any]) -> None:
    """Publica evento en Kafka para el extractor (Agent Brain)."""
    if not kafka_producer:
        logger.debug("Kafka no está configurado; evento no publicado.")
        return
    try:
        await kafka_producer.send_and_wait(
            settings.kafka_topic,
            orjson.dumps(payload),
        )
    except Exception as exc:
        logger.error("No se pudo publicar evento en Kafka: %s", exc)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret, algorithm="HS256")

class GoogleAuthRequest(BaseModel):
    code: str
    role: str = "patient"
    redirect_uri: str = "postmessage"

@app.post("/api/auth/google")
async def google_auth_endpoint(req: GoogleAuthRequest):
    if not settings.google_client_id:
        raise HTTPException(status_code=500, detail="Google Auth not configured")
        
    try:
        client_config = {
            "web": {
                "client_id": settings.google_client_id,
                "project_id": "presickness",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_secret": settings.google_client_secret or "dummy",
                "redirect_uris": [req.redirect_uri]
            }
        }
        
        flow = Flow.from_client_config(
            client_config,
            scopes=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"]
        )
        flow.redirect_uri = req.redirect_uri
        flow.fetch_token(code=req.code)
        
        # Get user info
        session = flow.authorized_session()
        user_info = session.get('https://www.googleapis.com/oauth2/v2/userinfo').json()
        email = user_info.get("email")
        name = user_info.get("name", "")
        google_id = user_info.get("id", "")
        
        if not email:
             raise HTTPException(status_code=400, detail="Could not retrieve email from Google")
             
        user_id_hash = hashlib.sha256(email.encode()).hexdigest()
        
        # Add to postgres
        if db_pool:
             async with db_pool.acquire() as conn:
                  # Create if not exists
                  await conn.execute(
                      """
                      INSERT INTO users (user_id_hash, email, name, status) 
                      VALUES ($1, $2, $3, 'active')
                      ON CONFLICT (user_id_hash) DO UPDATE SET name = EXCLUDED.name
                      """,
                      user_id_hash, email, name
                  )
                  
                  await conn.execute(
                      """
                      INSERT INTO user_oauth (user_id_hash, provider, access_token, refresh_token)
                      VALUES ($1, 'google', $2, $3)
                      ON CONFLICT (user_id_hash, provider) DO UPDATE SET 
                          access_token = EXCLUDED.access_token,
                          refresh_token = COALESCE(EXCLUDED.refresh_token, user_oauth.refresh_token),
                          updated_at = NOW()
                      """,
                      user_id_hash, flow.credentials.token, flow.credentials.refresh_token
                  )
        
        token = create_access_token({"sub": user_id_hash, "email": email, "role": req.role})
        return {"access_token": token, "token_type": "bearer", "user": {"email": email, "name": name, "user_id_hash": user_id_hash}}
        
    except Exception as e:
        logger.error(f"Google auth error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/patients/me")
async def get_profile(current_user: dict = Depends(get_current_user)):
    if not db_pool:
         return {"email": current_user["email"], "id": current_user["user_id_hash"], "role": "patient"}
         
    async with db_pool.acquire() as conn:
         row = await conn.fetchrow("SELECT name, email, user_id_hash, status FROM users WHERE user_id_hash = $1", current_user["user_id_hash"])
         if row:
              return {"id": row["user_id_hash"], "email": row["email"], "name": row["name"], "role": "patient", "status": row["status"]}
         return {"email": current_user["email"], "id": current_user["user_id_hash"], "role": "patient"}

# === API Routes ===
@app.post("/v1/ingest", status_code=202)
async def ingest_datapoint(
    data: DataPoint,
    auth_device_id: str = Depends(verify_device),
    request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
):
    """Ingesta principal."""
    validate_timestamp(data.timestamp)

    if data.device_id_hash != auth_device_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Device mismatch",
        )

    await verify_signature(data)

    quality_score = compute_quality_score(data.numeric_features)
    await store_datapoint(data, quality_score)

    await publish_event(
        {
            "user_id_hash": data.user_id_hash,
            "device_id_hash": data.device_id_hash,
            "timestamp": data.timestamp.isoformat(),
            "quality_score": quality_score,
        }
    )

    logger.info(
        "Datapoint aceptado. user=%s..., req=%s",
        data.user_id_hash[:6],
        request_id,
    )
    return {"status": "accepted", "request_id": request_id}


@app.get("/v1/health")
async def health_check():
    """Health check sencillo para Docker Healthcheck."""
    db_ok = db_pool is not None
    redis_ok = False
    if redis_client:
        try:
            redis_ok = await redis_client.ping()
        except Exception:
            redis_ok = False

    components = {
        "database": "ok" if db_ok else "error",
        "redis": "ok" if redis_ok else "error",
        "kafka": "ok" if kafka_producer else "disabled",
        "timestamp": datetime.utcnow().isoformat(),
    }
    status_label = (
        "healthy"
        if components["database"] == "ok" and components["redis"] == "ok"
        else "degraded"
    )
    return {"status": status_label, **components}


# === Lifespan ===
@app.on_event("startup")
async def on_startup():
    global db_pool, redis_client, kafka_producer
    logger.info("Inicializando API Gateway...")
    db_pool = await asyncpg.create_pool(dsn=settings.db_dsn, min_size=2, max_size=10)

    redis_kwargs = {}
    if settings.redis_password:
        redis_kwargs["password"] = settings.redis_password
    redis_client = redis.from_url(settings.redis_url, **redis_kwargs)

    if settings.kafka_bootstrap_servers:
        kafka_producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers
        )
        await kafka_producer.start()
        logger.info(
            "Kafka producer inicializado (topic=%s)", settings.kafka_topic
        )
    else:
        logger.warning("Kafka no configurado. Procesamiento asíncrono dependerá de cron.")


@app.on_event("shutdown")
async def on_shutdown():
    global db_pool, redis_client, kafka_producer
    logger.info("Apagando API Gateway...")
    if kafka_producer:
        await kafka_producer.stop()
        kafka_producer = None
    if redis_client:
        await redis_client.close()
        redis_client = None
    if db_pool:
        await db_pool.close()
        db_pool = None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8010, reload=True)


