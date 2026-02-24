from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ProcessingRequest(BaseModel):
    message_id: str
    user_id: str
    text: str
    timestamp: Optional[str] = None
    language_hint: Optional[str] = "es"

class SymptomScore(BaseModel):
    prob: float
    logit: float
    uncertainty: float

class EmbeddingData(BaseModel):
    model: str
    dim: int
    vector: List[float]

class LinguisticMeta(BaseModel):
    negation_count: int
    pronoun_ratio: float
    temporal_refs: List[str]
    tokens: int
    sentiment_blob_legacy: Optional[float] = None  # For transitional compatibility/logging

class ProcessingResponse(BaseModel):
    message_id: str
    user_id: str
    timestamp: Optional[str] = None
    language: str
    text_hash: str
    embeddings: EmbeddingData
    symptom_scores: Dict[str, SymptomScore]
    linguistic_meta: LinguisticMeta
    model_version: str
    processing_time_ms: float
