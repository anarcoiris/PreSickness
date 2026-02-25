from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, field_validator

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
