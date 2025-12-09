"""
ML Inference Service - Agents Brain & Archie
Sirve probabilidades de brote usando el último modelo TFT registrado en MLflow.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

import asyncpg
import mlflow
import mlflow.pyfunc
import numpy as np
import orjson
import pandas as pd
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("ml-inference")
logging.basicConfig(level=logging.INFO)


class Settings(BaseSettings):
    db_dsn: str = Field(
        default="postgresql://emuser:changeme@postgres:5432/empredictor",
        description="DSN de Postgres/Timescale con feature store.",
    )
    redis_url: Optional[str] = Field(default=None, description="Cache de features.")
    redis_password: Optional[str] = None
    mlflow_uri: str = Field(
        default="http://mlflow:5000", description="Tracking server para modelos."
    )
    model_uri: str = Field(
        default="models:/tft_prototype/latest",
        description="URI MLflow del modelo desplegado.",
    )
    default_window_days: int = Field(default=14)
    risk_threshold: float = Field(default=0.35)

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()
mlflow.set_tracking_uri(settings.mlflow_uri)


class PredictionRequest(BaseModel):
    user_id_hash: str = Field(min_length=64, max_length=64)
    window_days: int = Field(default=settings.default_window_days)
    horizon_days: int = Field(default=14, ge=7, le=30)
    features: Optional[Dict[str, float]] = None

    @field_validator("window_days")
    @classmethod
    def validate_window(cls, v: int) -> int:
        if v not in (1, 3, 7, 14, 30):
            raise ValueError("window_days debe ser uno de {1,3,7,14,30}.")
        return v


class PredictionResponse(BaseModel):
    user_id_hash: str
    horizon_days: int
    window_days: int
    relapse_probability: float
    risk_level: str
    model_uri: str


class TFTModelWrapper:
    """Encapsula la carga del modelo desde MLflow y heurística fallback."""

    def __init__(self, model_uri: str):
        self.model_uri = model_uri
        self.model: Optional[mlflow.pyfunc.PyFuncModel] = None

    def load(self) -> None:
        try:
            logger.info("Cargando modelo desde MLflow (%s)...", self.model_uri)
            self.model = mlflow.pyfunc.load_model(self.model_uri)
            logger.info("Modelo cargado correctamente.")
        except Exception as exc:
            logger.warning("No se pudo cargar el modelo (%s). Se usará heurística.", exc)
            self.model = None

    def predict(self, features: Dict[str, float], horizon_days: int) -> float:
        if self.model:
            try:
                df = pd.DataFrame([features])
                preds = self.model.predict(df)
                value = float(preds[0]) if isinstance(preds, (list, np.ndarray)) else float(preds)
                return float(np.clip(value, 0.0, 1.0))
            except Exception as exc:
                logger.error("Fallo en inferencia real, usando heurística. %s", exc)

        return self._heuristic_score(features, horizon_days)

    def _heuristic_score(self, features: Dict[str, float], horizon_days: int) -> float:
        """Baseline simple para no bloquear el pipeline."""
        sentiment = features.get("sentiment_mean") or features.get("sentiment_score") or 0.0
        steps = features.get("steps_mean") or features.get("steps") or 6000
        sleep = features.get("sleep_hours_mean") or features.get("sleep_hours") or 6.5
        stress_proxy = features.get("num_messages_mean") or features.get("num_messages") or 8

        score = 0.5
        score += -0.4 * sentiment  # ánimo bajo incrementa riesgo
        score += 0.00002 * max(0, 10000 - steps)
        score += 0.03 * max(0, 7 - sleep)
        score += 0.01 * (stress_proxy - 8)
        score += 0.02 * ((horizon_days - 14) / 14)

        return float(np.clip(score, 0.01, 0.99))


model_wrapper = TFTModelWrapper(settings.model_uri)
db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None
app = FastAPI(title="EM Predictor - Inference", version="0.1.0")


async def get_cached_features(user_id_hash: str, window_days: int) -> Optional[Dict]:
    if not redis_client:
        return None
    cache_key = f"features:{user_id_hash}:window:{window_days}"
    data = await redis_client.get(cache_key)
    if not data:
        return None
    try:
        return orjson.loads(data)
    except Exception:
        return None


async def fetch_features(user_id_hash: str, window_days: int) -> Optional[Dict]:
    if not db_pool:
        return None
    cached = await get_cached_features(user_id_hash, window_days)
    if cached:
        return cached

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT features
            FROM feature_windows
            WHERE user_id_hash = $1
              AND window_size_days = $2
            ORDER BY window_end DESC
            LIMIT 1
            """,
            user_id_hash,
            window_days,
        )
        if not row:
            return None
        features = dict(row["features"])
        if redis_client:
            cache_key = f"features:{user_id_hash}:window:{window_days}"
            await redis_client.setex(cache_key, 900, orjson.dumps(features))
        return features


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Devuelve probabilidad de brote a N días vista."""
    features = request.features
    if not features:
        features = await fetch_features(request.user_id_hash, request.window_days)
    if not features:
        raise HTTPException(status_code=404, detail="No hay features para ese usuario.")

    probability = model_wrapper.predict(features, request.horizon_days)
    risk_level = "critical" if probability >= settings.risk_threshold else "warning"

    return PredictionResponse(
        user_id_hash=request.user_id_hash,
        horizon_days=request.horizon_days,
        window_days=request.window_days,
        relapse_probability=probability,
        risk_level=risk_level,
        model_uri=settings.model_uri,
    )


@app.get("/v1/health")
async def health():
    return {
        "status": "ok" if db_pool else "degraded",
        "model_loaded": bool(model_wrapper.model),
        "db": bool(db_pool),
        "redis": bool(redis_client),
    }


@app.on_event("startup")
async def startup():
    global db_pool, redis_client
    logger.info("Inicializando servicio de inferencia...")
    model_wrapper.load()
    db_pool = await asyncpg.create_pool(settings.db_dsn, min_size=1, max_size=5)
    if settings.redis_url:
        redis_kwargs = {}
        if settings.redis_password:
            redis_kwargs["password"] = settings.redis_password
        redis_client = redis.from_url(settings.redis_url, **redis_kwargs)
    logger.info("Servicio listo.")


@app.on_event("shutdown")
async def shutdown():
    global db_pool, redis_client
    if db_pool:
        await db_pool.close()
        db_pool = None
    if redis_client:
        await redis_client.close()
        redis_client = None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

