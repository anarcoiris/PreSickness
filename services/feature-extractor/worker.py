"""
Feature Extraction Worker - Agents Backus & Brain
Consume eventos de ingesta, recalcula ventanas temporales y almacena en feature store.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import asyncpg
import librosa
import numpy as np
import orjson
import redis.asyncio as redis
from aiokafka import AIOKafkaConsumer
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer
from textblob import TextBlob

logger = logging.getLogger("feature-extractor")
logging.basicConfig(level=logging.INFO)


class Settings(BaseSettings):
    db_dsn: str = "postgresql://emuser:changeme@postgres:5432/empredictor"
    redis_url: str = "redis://redis:6379/0"
    redis_password: Optional[str] = None
    kafka_bootstrap_servers: Optional[str] = None
    kafka_topic: str = "ingest.datapoints.v1"
    window_sizes: List[int] = [1, 3, 7, 14, 30]
    poll_interval_seconds: int = 60

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()


class FeatureExtractor:
    """Encapsula la lógica de extracción y almacenamiento."""

    def __init__(self, config: Settings):
        self.config = config
        self.redis: Optional[redis.Redis] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    async def initialize(self) -> None:
        self.db_pool = await asyncpg.create_pool(self.config.db_dsn, min_size=2, max_size=10)
        redis_kwargs = {}
        if self.config.redis_password:
            redis_kwargs["password"] = self.config.redis_password
        self.redis = redis.from_url(self.config.redis_url, **redis_kwargs)
        logger.info("Feature extractor inicializado.")

    async def close(self) -> None:
        if self.redis:
            await self.redis.close()
        if self.db_pool:
            await self.db_pool.close()

    def extract_text_features(self, text: str, timestamp: Optional[datetime] = None) -> Dict:
        """
        Extrae features lingüísticas y de actividad de un texto.
        
        Mejoras:
        - Tokenización mejorada para español (maneja ¿¡ y abreviaturas)
        - Features de actividad temporal
        - Soporte bilingüe (ES/EN)
        """
        if not text or not text.strip():
            return self._empty_text_features()

        # Tokenización mejorada para español
        sentences = self._tokenize_sentences(text)
        words = self._tokenize_words(text)
        
        if not words:
            return self._empty_text_features()
        
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity

        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0
        avg_sentence_len = float(np.mean([len(s.split()) for s in sentences])) if sentences else 0

        # Pronombres bilingües (ES + EN)
        pronouns_en = {"i", "me", "my", "mine", "myself"}
        pronouns_es = {"yo", "me", "mi", "mí", "conmigo"}
        pronouns = pronouns_en | pronouns_es
        pronoun_count = sum(1 for w in words if w.lower() in pronouns)
        pronoun_ratio = pronoun_count / len(words) if words else 0

        # Marcadores temporales bilingües
        future_markers = {"will", "gonna", "tomorrow", "next", "soon", 
                         "mañana", "próximo", "luego", "después", "pronto"}
        past_markers = {"was", "were", "had", "yesterday", "ago",
                       "ayer", "antes", "pasado", "fue", "era"}

        future_count = sum(1 for w in words if w.lower() in future_markers)
        past_count = sum(1 for w in words if w.lower() in past_markers)
        
        # Features de estilo
        exclamation_count = text.count("!") + text.count("¡")
        question_count = text.count("?") + text.count("¿")
        ellipsis_count = text.count("...")
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Features temporales (si se proporciona timestamp)
        ts = timestamp or datetime.utcnow()
        hour = ts.hour
        day_of_week = ts.weekday()
        is_night = 1 if (hour < 6 or hour >= 23) else 0
        is_weekend = 1 if day_of_week >= 5 else 0

        return {
            # Features lingüísticas base
            "sentiment_score": float(sentiment),
            "avg_sentence_len": avg_sentence_len,
            "type_token_ratio": float(ttr),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "char_count": len(text),
            "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0,
            
            # Pronombres y orientación temporal
            "pronoun_ratio": float(pronoun_ratio),
            "future_orientation": float(future_count / len(words)) if words else 0,
            "past_orientation": float(past_count / len(words)) if words else 0,
            
            # Estilo y expresividad
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "ellipsis_count": ellipsis_count,
            "caps_ratio": float(caps_ratio),
            
            # Features temporales
            "hour": hour,
            "day_of_week": day_of_week,
            "is_night": is_night,
            "is_weekend": is_weekend,
            
            "extracted_at": datetime.utcnow().isoformat(),
        }
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenización de oraciones mejorada para español.
        Maneja: ¿¡, abreviaturas comunes, puntos suspensivos.
        """
        import re
        
        # Proteger abreviaturas comunes
        abbreviations = {
            "Dr.": "Dr§", "Dra.": "Dra§", "Sr.": "Sr§", "Sra.": "Sra§",
            "Lic.": "Lic§", "Ing.": "Ing§", "etc.": "etc§", "vs.": "vs§",
            "p.ej.": "p§ej§", "i.e.": "i§e§", "e.g.": "e§g§",
        }
        protected = text
        for abbr, replacement in abbreviations.items():
            protected = protected.replace(abbr, replacement)
        
        # Proteger puntos suspensivos
        protected = protected.replace("...", "§§§")
        
        # Separar por puntuación final
        sentences = re.split(r'[.!?¡¿]+', protected)
        
        # Restaurar y limpiar
        result = []
        for s in sentences:
            s = s.replace("§§§", "...").strip()
            for abbr, replacement in abbreviations.items():
                s = s.replace(replacement, abbr)
            if s:
                result.append(s)
        
        return result
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenización de palabras básica."""
        import re
        # Eliminar URLs
        text = re.sub(r'https?://\S+', '', text)
        # Tokenizar por espacios y puntuación
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def _empty_text_features(self) -> Dict:
        return {
            "sentiment_score": 0.0,
            "avg_sentence_len": 0.0,
            "type_token_ratio": 0.0,
            "word_count": 0,
            "sentence_count": 0,
            "char_count": 0,
            "avg_word_length": 0.0,
            "pronoun_ratio": 0.0,
            "future_orientation": 0.0,
            "past_orientation": 0.0,
            "exclamation_count": 0,
            "question_count": 0,
            "ellipsis_count": 0,
            "caps_ratio": 0.0,
            "hour": 0,
            "day_of_week": 0,
            "is_night": 0,
            "is_weekend": 0,
        }

    def compute_embedding(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(384)
        return self.sentence_model.encode(text, convert_to_numpy=True)

    def extract_audio_features(self, audio_path: str) -> Dict:
        """Mantiene compatibilidad con roadmap de audio (Agent Brain)."""
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            pitches, _ = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0

            energy = librosa.feature.rms(y=y)[0]
            energy_mean = float(np.mean(energy))
            energy_std = float(np.std(energy))

            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = float(np.mean(zcr))

            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = mfccs.mean(axis=1).tolist()

            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

            return {
                "pitch_mean": float(pitch_mean),
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "zcr_mean": zcr_mean,
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "mfcc_mean": mfcc_mean,
                "tempo_bpm": float(tempo),
                "duration_sec": float(len(y) / sr),
            }
        except Exception as exc:
            logger.error("Audio feature extraction failed: %s", exc)
            return {
                "pitch_mean": 0.0,
                "energy_mean": 0.0,
                "energy_std": 0.0,
                "zcr_mean": 0.0,
                "spectral_centroid_mean": 0.0,
                "spectral_rolloff_mean": 0.0,
                "mfcc_mean": [0.0] * 13,
                "tempo_bpm": 0.0,
                "duration_sec": 0.0,
            }

    async def compute_windowed_features(self, user_id_hash: str) -> Dict[int, Dict]:
        """
        Calcula features agregadas por ventana temporal.
        
        Incluye:
        - Features lingüísticas (sentiment, TTR, longitud)
        - Features de ACTIVIDAD (mensajes/día, horas activas, ratio nocturno)
        - Features de salud (steps, sleep, HR)
        - Tendencias temporales
        """
        if not self.db_pool:
            raise RuntimeError("DB pool no inicializado")
        results: Dict[int, Dict] = {}

        async with self.db_pool.acquire() as conn:
            for window_days in settings.window_sizes:
                window_start = datetime.now(timezone.utc) - timedelta(days=window_days)
                rows = await conn.fetch(
                    """
                    SELECT time, numeric_features
                    FROM datapoints
                    WHERE user_id_hash = $1
                      AND time >= $2
                    ORDER BY time ASC
                    """,
                    user_id_hash,
                    window_start,
                )

                if not rows:
                    results[window_days] = self._empty_windowed_features(window_days)
                    continue

                numeric_features = [
                    row["numeric_features"] if isinstance(row["numeric_features"], dict) else json.loads(row["numeric_features"])
                    for row in rows
                ]
                timestamps = [row["time"] for row in rows]

                def safe_list(values):
                    return [v for v in values if v is not None]

                # Features lingüísticas base
                sentiment_values = safe_list([f.get("sentiment_score") for f in numeric_features])
                avg_sentence_values = safe_list([f.get("avg_sentence_len") for f in numeric_features])
                ttr_values = safe_list([f.get("type_token_ratio") for f in numeric_features])
                word_count_values = safe_list([f.get("word_count") for f in numeric_features])
                
                # Features de actividad
                num_messages_values = safe_list([f.get("num_messages") for f in numeric_features])
                latency_values = safe_list([f.get("avg_response_latency_sec") for f in numeric_features])
                is_night_values = safe_list([f.get("is_night") for f in numeric_features])
                hour_values = safe_list([f.get("hour") for f in numeric_features])
                
                # Features de salud
                steps_values = safe_list([f.get("steps") for f in numeric_features])
                sleep_values = safe_list([f.get("sleep_hours") for f in numeric_features])
                hr_values = safe_list([f.get("hr_mean") for f in numeric_features])
                
                # Features de estilo
                exclamation_values = safe_list([f.get("exclamation_count") for f in numeric_features])
                question_values = safe_list([f.get("question_count") for f in numeric_features])
                caps_ratio_values = safe_list([f.get("caps_ratio") for f in numeric_features])
                
                # === NUEVAS FEATURES DE ACTIVIDAD ===
                
                # Días únicos con actividad
                unique_days = set(ts.date() for ts in timestamps)
                num_active_days = len(unique_days)
                coverage = num_active_days / window_days if window_days > 0 else 0
                
                # Horas únicas activas (promedio por día)
                hours_per_day = {}
                for ts in timestamps:
                    day = ts.date()
                    if day not in hours_per_day:
                        hours_per_day[day] = set()
                    hours_per_day[day].add(ts.hour)
                active_hours_mean = float(np.mean([len(h) for h in hours_per_day.values()])) if hours_per_day else 0.0
                
                # Ratio nocturno (mensajes entre 23:00-06:00)
                night_ratio = float(np.mean(is_night_values)) if is_night_values else 0.0
                
                # Gap máximo entre mensajes (en horas)
                if len(timestamps) > 1:
                    gaps = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600 
                            for i in range(len(timestamps) - 1)]
                    gap_max_hours = float(max(gaps))
                    gap_mean_hours = float(np.mean(gaps))
                else:
                    gap_max_hours = 0.0
                    gap_mean_hours = 0.0
                
                # Mensajes por día
                messages_per_day = {}
                for ts in timestamps:
                    day = ts.date()
                    messages_per_day[day] = messages_per_day.get(day, 0) + 1
                messages_daily_values = list(messages_per_day.values())
                messages_per_day_mean = float(np.mean(messages_daily_values)) if messages_daily_values else 0.0
                messages_per_day_std = float(np.std(messages_daily_values)) if messages_daily_values else 0.0
                
                # Tendencia de actividad (slope de mensajes/día)
                if len(messages_daily_values) >= 2:
                    messages_trend = self._compute_trend(messages_daily_values)
                else:
                    messages_trend = 0.0

                aggregated = {
                    "window_days": window_days,
                    "num_datapoints": len(rows),
                    
                    # Features lingüísticas
                    "sentiment_mean": float(np.mean(sentiment_values)) if sentiment_values else 0.0,
                    "sentiment_std": float(np.std(sentiment_values)) if sentiment_values else 0.0,
                    "sentiment_trend": self._compute_trend(sentiment_values),
                    "avg_sentence_len_mean": float(np.mean(avg_sentence_values)) if avg_sentence_values else 0.0,
                    "ttr_mean": float(np.mean(ttr_values)) if ttr_values else 0.0,
                    "word_count_mean": float(np.mean(word_count_values)) if word_count_values else 0.0,
                    "word_count_total": int(sum(word_count_values)) if word_count_values else 0,
                    
                    # Features de estilo
                    "exclamation_mean": float(np.mean(exclamation_values)) if exclamation_values else 0.0,
                    "question_mean": float(np.mean(question_values)) if question_values else 0.0,
                    "caps_ratio_mean": float(np.mean(caps_ratio_values)) if caps_ratio_values else 0.0,
                    
                    # === FEATURES DE ACTIVIDAD (NUEVAS) ===
                    "num_active_days": num_active_days,
                    "coverage": float(coverage),
                    "messages_per_day_mean": messages_per_day_mean,
                    "messages_per_day_std": messages_per_day_std,
                    "messages_trend": messages_trend,
                    "active_hours_mean": active_hours_mean,
                    "night_ratio": night_ratio,
                    "gap_max_hours": gap_max_hours,
                    "gap_mean_hours": gap_mean_hours,
                    
                    # Legacy (mantener compatibilidad)
                    "num_messages_total": int(sum(num_messages_values)) if num_messages_values else len(rows),
                    "num_messages_mean": float(np.mean(num_messages_values)) if num_messages_values else 0.0,
                    "response_latency_mean": float(np.mean(latency_values)) if latency_values else 0.0,
                    
                    # Features de salud
                    "steps_mean": float(np.mean(steps_values)) if steps_values else 0.0,
                    "sleep_hours_mean": float(np.mean(sleep_values)) if sleep_values else 0.0,
                    "hr_mean": float(np.mean(hr_values)) if hr_values else 0.0,
                    
                    # Metadata temporal
                    "window_start": window_start.isoformat(),
                    "window_end": datetime.now(timezone.utc).isoformat(),
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                }
                results[window_days] = aggregated
        return results

    def _compute_trend(self, values: List[float]) -> float:
        if not values or len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values)
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)

    def _empty_windowed_features(self, window_days: int) -> Dict:
        return {
            "window_days": window_days,
            "num_datapoints": 0,
            
            # Features lingüísticas
            "sentiment_mean": 0.0,
            "sentiment_std": 0.0,
            "sentiment_trend": 0.0,
            "avg_sentence_len_mean": 0.0,
            "ttr_mean": 0.0,
            "word_count_mean": 0.0,
            "word_count_total": 0,
            
            # Features de estilo
            "exclamation_mean": 0.0,
            "question_mean": 0.0,
            "caps_ratio_mean": 0.0,
            
            # Features de actividad
            "num_active_days": 0,
            "coverage": 0.0,
            "messages_per_day_mean": 0.0,
            "messages_per_day_std": 0.0,
            "messages_trend": 0.0,
            "active_hours_mean": 0.0,
            "night_ratio": 0.0,
            "gap_max_hours": 0.0,
            "gap_mean_hours": 0.0,
            
            # Legacy
            "num_messages_total": 0,
            "num_messages_mean": 0.0,
            "response_latency_mean": 0.0,
            
            # Features de salud
            "steps_mean": 0.0,
            "sleep_hours_mean": 0.0,
            "hr_mean": 0.0,
            
            # Metadata
            "window_start": datetime.now(timezone.utc).isoformat(),
            "window_end": datetime.now(timezone.utc).isoformat(),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    async def store_windowed_features(self, user_id_hash: str, features: Dict[int, Dict]) -> None:
        if not self.db_pool:
            raise RuntimeError("DB pool no inicializado")
        async with self.db_pool.acquire() as conn:
            for window_days, feat_dict in features.items():
                await conn.execute(
                    """
                    INSERT INTO feature_windows (
                        user_id_hash,
                        window_start,
                        window_end,
                        window_size_days,
                        features,
                        num_datapoints
                    )
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                    ON CONFLICT (user_id_hash, window_end, window_size_days)
                    DO UPDATE SET
                        features = EXCLUDED.features,
                        num_datapoints = EXCLUDED.num_datapoints,
                        computed_at = NOW()
                    """,
                    user_id_hash,
                    feat_dict["window_start"],
                    feat_dict["window_end"],
                    window_days,
                    json.dumps(feat_dict),
                    feat_dict["num_datapoints"],
                )

        if self.redis:
            cache_key = f"features:{user_id_hash}:windows"
            await self.redis.setex(cache_key, 3600, json.dumps(features))
        logger.info("Stored windowed features for user %s...", user_id_hash[:6])


class FeatureExtractionWorker:
    """Orquestador principal."""

    def __init__(self, config: Settings):
        self.config = config
        self.extractor = FeatureExtractor(config)
        self._stop = asyncio.Event()

    async def start(self) -> None:
        await self.extractor.initialize()
        await asyncio.gather(self._consume_events(), self._scheduled_backfill())

    async def stop(self) -> None:
        self._stop.set()
        await self.extractor.close()

    async def _consume_events(self) -> None:
        if not self.config.kafka_bootstrap_servers:
            logger.warning("Kafka no configurado. Usando modo solo backfill.")
            return

        consumer = AIOKafkaConsumer(
            self.config.kafka_topic,
            bootstrap_servers=self.config.kafka_bootstrap_servers,
            enable_auto_commit=True,
            value_deserializer=lambda v: orjson.loads(v),
            group_id="feature-extractor",
        )

        await consumer.start()
        logger.info("Kafka consumer listo. Esperando eventos...")
        try:
            async for message in consumer:
                if self._stop.is_set():
                    break
                payload = message.value
                user_id = payload.get("user_id_hash")
                if not user_id:
                    continue
                await self._process_user(user_id)
        finally:
            await consumer.stop()

    async def _scheduled_backfill(self) -> None:
        while not self._stop.is_set():
            await self._process_recent_users()
            await asyncio.wait(
                [self._stop.wait()],
                timeout=self.config.poll_interval_seconds,
            )

    async def _process_recent_users(self) -> None:
        if not self.extractor.db_pool:
            return
        async with self.extractor.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT user_id_hash
                FROM datapoints
                WHERE time >= NOW() - INTERVAL '1 hour'
                """
            )

        for row in rows:
            await self._process_user(row["user_id_hash"])

    async def _process_user(self, user_id_hash: str) -> None:
        features = await self.extractor.compute_windowed_features(user_id_hash)
        await self.extractor.store_windowed_features(user_id_hash, features)


async def main():
    worker = FeatureExtractionWorker(settings)
    try:
        await worker.start()
    except asyncio.CancelledError:
        await worker.stop()
    except KeyboardInterrupt:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())

