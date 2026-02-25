import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import jwt
import orjson
from fastapi import APIRouter, Depends, HTTPException, status, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from dependencies import settings, get_current_patient
import db
from ingest_models import DataPoint, NumericFeatures

logger = logging.getLogger("unified-ingest")
router = APIRouter()
security = HTTPBearer(auto_error=True)

# For redis verification
redis_client = None
kafka_producer = None

def get_redis_client():
    global redis_client
    return redis_client

def get_kafka_producer():
    global kafka_producer
    return kafka_producer

async def verify_device(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Valida token de dispositivo almacenado en Redis o JWT de usuario web."""
    token = credentials.credentials
    try:
        # Intenta verificar si es un JWT de usuario web
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload.get("sub")
    except jwt.PyJWTError:
        pass # fallback to device token in redis

    redis_cli = get_redis_client()
    if not redis_cli:
        raise HTTPException(status_code=503, detail="Auth service (Redis) unavailable for device tokens")

    token_key = f"{settings.device_token_prefix}:{token}"
    device_id = await redis_cli.get(token_key)
    if not device_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    device_id_hash = (
        device_id.decode() if isinstance(device_id, (bytes, bytearray)) else str(device_id)
    )
    return device_id_hash

async def verify_signature(data: DataPoint) -> None:
    """Verifica integridad mediante HMAC con secreto de dispositivo."""
    device_secret = await db.get_device_secret(data.device_id_hash)
    if not device_secret:
        raise HTTPException(status_code=404, detail="Device not registered")

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

async def store_datapoint_and_cache(data: DataPoint, quality_score: float) -> None:
    """Inserta datapoint en Postgres y actualiza cache en Redis."""
    
    # Adapt to use specific dict for insertion or map properties directly
    mapped_data = {
        'timestamp': data.timestamp,
        'user_id_hash': data.user_id_hash,
        'device_id_hash': data.device_id_hash,
        'embedding': data.embedding.model_dump(),
        'numeric_features': data.numeric_features.model_dump(exclude_none=True),
        'source_hash': uuid.uuid4().hex  # Simplified unique hash for device uploads
    }
    
    await db.store_device_datapoint(mapped_data, quality_score)

    redis_cli = get_redis_client()
    if redis_cli:
        cache_key = f"features:{data.user_id_hash}:latest"
        await redis_cli.setex(
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

async def publish_event(payload: dict) -> None:
    """Publica evento en Kafka para el extractor (Agent Brain)."""
    producer = get_kafka_producer()
    if not producer:
        logger.debug("Kafka no está configurado; evento no publicado.")
        return
    try:
        await producer.send_and_wait(
            settings.kafka_topic,
            orjson.dumps(payload),
        )
    except Exception as exc:
        logger.error("No se pudo publicar evento en Kafka: %s", exc)

@router.post("/v1/ingest", status_code=202)
async def ingest_datapoint(
    data: DataPoint,
    auth_device_id: str = Depends(verify_device),
    request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
):
    """Ingesta principal unificada."""
    validate_timestamp(data.timestamp)

    if data.device_id_hash != auth_device_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Device mismatch",
        )

    await verify_signature(data)

    quality_score = compute_quality_score(data.numeric_features)
    await store_datapoint_and_cache(data, quality_score)

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
