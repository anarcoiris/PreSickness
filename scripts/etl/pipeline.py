"""
ETL Pipeline - EM Predictor
Pipeline completo para procesar datos de pacientes y preparar para entrenamiento.

Responsables:
- Agent Backend (Backus): infraestructura y almacenamiento
- Agent ML (Brain): features y preparación para modelo

Flujo:
1. EXTRACT: Cargar mensajes crudos (WhatsApp, Telegram, CSV, etc.)
2. TRANSFORM: Limpiar, tokenizar, extraer features, calcular ventanas
3. LOAD: Almacenar en feature store (Postgres) o exportar (Parquet)

Uso:
    python -m scripts.etl.pipeline \
        --input data/raw/messages.json \
        --events data/raw/events.csv \
        --output data/processed/ \
        --patient-id P001
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("etl-pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ETLConfig:
    """Configuración del pipeline ETL."""

    # Rutas
    input_path: Path
    events_path: Optional[Path] = None
    output_path: Path = Path("data/processed")

    # Paciente
    patient_id: str = "anonymous"
    patient_id_hash: Optional[str] = None

    # Ventanas temporales
    window_sizes: List[int] = field(default_factory=lambda: [1, 3, 7, 14, 30])
    
    # Horizontes de predicción
    prediction_horizons: List[int] = field(default_factory=lambda: [7, 14, 30])

    # Procesamiento de texto
    language: str = "es"
    min_message_length: int = 3  # Ignorar mensajes muy cortos
    
    # Agregación
    aggregation_freq: str = "D"  # Diario
    min_messages_per_window: int = 1

    # Exportación
    export_format: str = "parquet"  # parquet, csv, postgres

    def __post_init__(self):
        if self.patient_id_hash is None:
            self.patient_id_hash = hashlib.sha256(
                self.patient_id.encode()
            ).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTORES (E)
# ══════════════════════════════════════════════════════════════════════════════


class MessageExtractor:
    """Extrae mensajes de diferentes formatos de entrada."""

    SUPPORTED_FORMATS = ["whatsapp", "telegram", "csv", "jsonl", "chatgpt"]

    def __init__(self, config: ETLConfig):
        self.config = config

    def extract(self, path: Path) -> pd.DataFrame:
        """
        Extrae mensajes del archivo de entrada.
        
        Returns:
            DataFrame con columnas: [timestamp, sender, text, metadata]
        """
        suffix = path.suffix.lower()
        
        if suffix == ".json":
            return self._extract_json(path)
        elif suffix == ".jsonl":
            return self._extract_jsonl(path)
        elif suffix == ".csv":
            return self._extract_csv(path)
        elif suffix == ".txt":
            return self._extract_whatsapp_txt(path)
        else:
            raise ValueError(f"Formato no soportado: {suffix}")

    def _extract_json(self, path: Path) -> pd.DataFrame:
        """Extrae de JSON (formato ChatGPT export o similar)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = []
        
        # Detectar formato
        if isinstance(data, list):
            # Lista de conversaciones (ChatGPT)
            for conv in data:
                messages.extend(self._parse_chatgpt_conversation(conv))
        elif isinstance(data, dict):
            if "mapping" in data:
                # Una sola conversación ChatGPT
                messages.extend(self._parse_chatgpt_conversation(data))
            elif "messages" in data:
                # Formato genérico {messages: [...]}
                for msg in data["messages"]:
                    messages.append(self._normalize_message(msg))

        return pd.DataFrame(messages)

    def _parse_chatgpt_conversation(self, conv: dict) -> List[dict]:
        """Parsea una conversación de ChatGPT export."""
        messages = []
        mapping = conv.get("mapping", {})
        
        for node_id, node in mapping.items():
            msg = node.get("message")
            if not msg:
                continue
            
            content = msg.get("content", {})
            parts = content.get("parts", [])
            text = " ".join(str(p) for p in parts if isinstance(p, str))
            
            if not text.strip():
                continue
            
            author = msg.get("author", {})
            role = author.get("role", "unknown") if isinstance(author, dict) else "unknown"
            
            create_time = msg.get("create_time")
            if create_time:
                try:
                    ts = datetime.fromtimestamp(create_time)
                except:
                    ts = datetime.now()
            else:
                ts = datetime.now()
            
            messages.append({
                "timestamp": ts,
                "sender": role,
                "text": text.strip(),
                "metadata": {"node_id": node_id},
            })
        
        return messages

    def _extract_whatsapp_txt(self, path: Path) -> pd.DataFrame:
        """Extrae de export de WhatsApp (.txt)."""
        # Patrón: [dd/mm/yyyy, hh:mm:ss] Nombre: Mensaje
        # o: dd/mm/yyyy, hh:mm - Nombre: Mensaje
        pattern = re.compile(
            r"^\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*[-–]?\s*([^:]+):\s*(.+)$"
        )
        
        messages = []
        current_msg = None
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                match = pattern.match(line)
                if match:
                    # Guardar mensaje anterior si existe
                    if current_msg:
                        messages.append(current_msg)
                    
                    date_str, time_str, sender, text = match.groups()
                    
                    # Parsear fecha
                    try:
                        # Intentar diferentes formatos
                        for fmt in ["%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y"]:
                            try:
                                date = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue
                        
                        # Añadir hora
                        time_parts = time_str.split(":")
                        date = date.replace(
                            hour=int(time_parts[0]),
                            minute=int(time_parts[1]),
                            second=int(time_parts[2]) if len(time_parts) > 2 else 0,
                        )
                    except:
                        date = datetime.now()
                    
                    current_msg = {
                        "timestamp": date,
                        "sender": sender.strip(),
                        "text": text.strip(),
                        "metadata": {},
                    }
                elif current_msg:
                    # Línea de continuación del mensaje anterior
                    current_msg["text"] += " " + line
        
        # Añadir último mensaje
        if current_msg:
            messages.append(current_msg)
        
        return pd.DataFrame(messages)

    def _extract_csv(self, path: Path) -> pd.DataFrame:
        """Extrae de CSV con columnas: timestamp, sender, text."""
        df = pd.read_csv(path)
        
        # Normalizar nombres de columnas
        col_map = {
            "date": "timestamp",
            "datetime": "timestamp",
            "time": "timestamp",
            "from": "sender",
            "author": "sender",
            "user": "sender",
            "message": "text",
            "content": "text",
            "body": "text",
        }
        
        df.columns = [col_map.get(c.lower(), c.lower()) for c in df.columns]
        
        # Asegurar columnas requeridas
        required = ["timestamp", "sender", "text"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Columna requerida no encontrada: {col}")
        
        # Parsear timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Añadir metadata vacía si no existe
        if "metadata" not in df.columns:
            df["metadata"] = [{}] * len(df)
        
        return df[["timestamp", "sender", "text", "metadata"]]

    def _extract_jsonl(self, path: Path) -> pd.DataFrame:
        """Extrae de JSONL (un mensaje por línea)."""
        messages = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    msg = json.loads(line)
                    messages.append(self._normalize_message(msg))
        return pd.DataFrame(messages)

    def _normalize_message(self, msg: dict) -> dict:
        """Normaliza un mensaje a formato estándar."""
        # Buscar timestamp
        ts = None
        for key in ["timestamp", "date", "datetime", "time", "created_at"]:
            if key in msg:
                ts = msg[key]
                break
        
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts)
        elif isinstance(ts, str):
            ts = pd.to_datetime(ts)
        else:
            ts = datetime.now()
        
        # Buscar sender
        sender = None
        for key in ["sender", "from", "author", "user", "role"]:
            if key in msg:
                sender = msg[key]
                break
        sender = sender or "unknown"
        
        # Buscar texto
        text = None
        for key in ["text", "message", "content", "body"]:
            if key in msg:
                text = msg[key]
                break
        text = text or ""
        
        return {
            "timestamp": ts,
            "sender": str(sender),
            "text": str(text),
            "metadata": {k: v for k, v in msg.items() if k not in ["timestamp", "sender", "text"]},
        }


class EventsExtractor:
    """Extrae eventos clínicos (brotes, etc.)."""

    def __init__(self, config: ETLConfig):
        self.config = config

    def extract(self, path: Path) -> pd.DataFrame:
        """
        Extrae eventos clínicos.
        
        Formato esperado CSV:
            date,event_type,severity,notes
            2024-03-15,relapse,moderate,fatiga severa
        
        Returns:
            DataFrame con columnas: [date, event_type, severity, notes]
        """
        if not path or not path.exists():
            logger.warning("No se proporcionó archivo de eventos")
            return pd.DataFrame(columns=["date", "event_type", "severity", "notes"])
        
        df = pd.read_csv(path)
        
        # Normalizar nombres
        col_map = {
            "fecha": "date",
            "tipo": "event_type",
            "type": "event_type",
            "severidad": "severity",
            "notas": "notes",
        }
        df.columns = [col_map.get(c.lower(), c.lower()) for c in df.columns]
        
        # Parsear fechas
        df["date"] = pd.to_datetime(df["date"])
        
        # Valores por defecto
        if "event_type" not in df.columns:
            df["event_type"] = "relapse"
        if "severity" not in df.columns:
            df["severity"] = "unknown"
        if "notes" not in df.columns:
            df["notes"] = ""
        
        logger.info(f"Cargados {len(df)} eventos clínicos")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMADORES (T)
# ══════════════════════════════════════════════════════════════════════════════


class TextProcessor:
    """Procesa y limpia texto."""

    def __init__(self, config: ETLConfig):
        self.config = config
        self._nlp = None

    @property
    def nlp(self):
        """Carga spaCy de forma lazy."""
        if self._nlp is None:
            try:
                import spacy
                model = "es_core_news_sm" if self.config.language == "es" else "en_core_web_sm"
                try:
                    self._nlp = spacy.load(model)
                except OSError:
                    logger.warning(f"Modelo spaCy {model} no encontrado. Usando tokenización básica.")
                    self._nlp = "basic"
            except ImportError:
                logger.warning("spaCy no instalado. Usando tokenización básica.")
                self._nlp = "basic"
        return self._nlp

    def clean_text(self, text: str) -> str:
        """Limpia texto de ruido."""
        if not text or not isinstance(text, str):
            return ""
        
        # Eliminar URLs
        text = re.sub(r"https?://\S+", "", text)
        
        # Eliminar menciones y hashtags (si aplica)
        text = re.sub(r"@\w+", "", text)
        
        # Eliminar emojis (opcional - podrían ser features útiles)
        # text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
        
        # Normalizar espacios
        text = re.sub(r"\s+", " ", text)
        
        # Eliminar caracteres de control
        text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
        
        return text.strip()

    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokeniza texto en oraciones."""
        if not text:
            return []
        
        if self.nlp == "basic":
            # Tokenización básica
            # Manejar signos de puntuación españoles
            text = text.replace("¿", " ¿").replace("¡", " ¡")
            sentences = re.split(r"[.!?]+", text)
            return [s.strip() for s in sentences if s.strip()]
        else:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]

    def tokenize_words(self, text: str) -> List[str]:
        """Tokeniza texto en palabras."""
        if not text:
            return []
        
        if self.nlp == "basic":
            return text.lower().split()
        else:
            doc = self.nlp(text)
            return [token.text.lower() for token in doc if not token.is_space]


class FeatureExtractor:
    """Extrae features de mensajes."""

    def __init__(self, config: ETLConfig):
        self.config = config
        self.text_processor = TextProcessor(config)

    def extract_message_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extrae features de un mensaje individual."""
        text = self.text_processor.clean_text(row["text"])
        
        if len(text) < self.config.min_message_length:
            return None
        
        sentences = self.text_processor.tokenize_sentences(text)
        words = self.text_processor.tokenize_words(text)
        
        if not words:
            return None
        
        # Features básicas
        features = {
            "timestamp": row["timestamp"],
            "sender": row["sender"],
            "text_clean": text,
            
            # Longitud
            "char_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            
            # Diversidad léxica
            "unique_words": len(set(words)),
            "type_token_ratio": len(set(words)) / len(words) if words else 0,
            
            # Puntuación y estilo
            "exclamation_count": text.count("!") + text.count("¡"),
            "question_count": text.count("?") + text.count("¿"),
            "ellipsis_count": text.count("..."),
            "caps_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            
            # Emojis (proxy de expresividad)
            "emoji_count": len(re.findall(r"[\U00010000-\U0010ffff]", text)),
            
            # Hora del día (features temporales)
            "hour": row["timestamp"].hour,
            "day_of_week": row["timestamp"].dayofweek,
            "is_weekend": row["timestamp"].dayofweek >= 5,
            "is_night": row["timestamp"].hour < 6 or row["timestamp"].hour >= 23,
        }
        
        # Sentiment básico (sin dependencias externas)
        features["sentiment_proxy"] = self._basic_sentiment(text)
        
        return features

    def _basic_sentiment(self, text: str) -> float:
        """Sentiment muy básico basado en palabras clave (placeholder)."""
        # En producción: usar TextBlob, transformers, o el MiniLLM
        positive = ["bien", "genial", "feliz", "mejor", "bueno", "gracias", "alegr", "content"]
        negative = ["mal", "fatal", "triste", "peor", "cansad", "dolor", "agotad", "difícil"]
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total


class ActivityAggregator:
    """Agrega features por ventanas temporales."""

    def __init__(self, config: ETLConfig):
        self.config = config

    def aggregate_daily(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Agrega features a nivel diario."""
        if features_df.empty:
            return pd.DataFrame()
        
        features_df = features_df.copy()
        features_df["date"] = features_df["timestamp"].dt.date
        
        # Agrupar por día
        daily = features_df.groupby("date").agg({
            # Conteos
            "word_count": ["sum", "mean", "std"],
            "sentence_count": ["sum", "mean"],
            "char_count": ["sum", "mean"],
            
            # Diversidad
            "type_token_ratio": "mean",
            "unique_words": "sum",
            
            # Estilo
            "exclamation_count": "sum",
            "question_count": "sum",
            "emoji_count": "sum",
            
            # Sentiment
            "sentiment_proxy": ["mean", "std"],
            
            # Actividad
            "timestamp": ["count", "min", "max"],
            "hour": lambda x: len(x.unique()),  # Horas activas
            "is_night": "mean",  # Ratio nocturno
        })
        
        # Aplanar columnas
        daily.columns = ["_".join(col).strip("_") for col in daily.columns]
        daily = daily.rename(columns={
            "timestamp_count": "messages_count",
            "timestamp_min": "first_message",
            "timestamp_max": "last_message",
            "hour_<lambda>": "active_hours",
            "is_night_mean": "night_ratio",
        })
        
        # Calcular gap máximo entre mensajes
        daily["active_span_hours"] = (
            (daily["last_message"] - daily["first_message"]).dt.total_seconds() / 3600
        )
        
        daily = daily.reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        
        return daily

    def compute_windows(
        self, 
        daily_df: pd.DataFrame, 
        target_date: datetime
    ) -> Dict[int, Dict[str, float]]:
        """Calcula features para múltiples ventanas temporales."""
        windows = {}
        
        for window_size in self.config.window_sizes:
            start_date = target_date - timedelta(days=window_size)
            window_data = daily_df[
                (daily_df["date"] >= start_date) & 
                (daily_df["date"] < target_date)
            ]
            
            if len(window_data) < self.config.min_messages_per_window:
                windows[window_size] = self._empty_window_features(window_size)
                continue
            
            features = {
                "window_size_days": window_size,
                "num_active_days": len(window_data),
                "coverage": len(window_data) / window_size,
                
                # Volumen
                "messages_total": window_data["messages_count"].sum(),
                "messages_mean": window_data["messages_count"].mean(),
                "messages_std": window_data["messages_count"].std() or 0,
                
                # Palabras
                "words_total": window_data["word_count_sum"].sum(),
                "words_mean": window_data["word_count_mean"].mean(),
                
                # Diversidad
                "ttr_mean": window_data["type_token_ratio_mean"].mean(),
                
                # Sentiment
                "sentiment_mean": window_data["sentiment_proxy_mean"].mean(),
                "sentiment_std": window_data["sentiment_proxy_std"].mean() or 0,
                
                # Actividad
                "active_hours_mean": window_data["active_hours"].mean(),
                "night_ratio_mean": window_data["night_ratio"].mean(),
                
                # Tendencias
                "messages_trend": self._compute_trend(window_data["messages_count"]),
                "sentiment_trend": self._compute_trend(window_data["sentiment_proxy_mean"]),
            }
            
            windows[window_size] = features
        
        return windows

    def _compute_trend(self, series: pd.Series) -> float:
        """Calcula tendencia lineal (slope normalizado)."""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        # Evitar NaN
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 0.0
        
        slope, _ = np.polyfit(x[mask], y[mask], 1)
        
        # Normalizar por la media
        mean = np.nanmean(y)
        if abs(mean) < 1e-6:
            return 0.0
        
        return float(slope / mean)

    def _empty_window_features(self, window_size: int) -> Dict[str, float]:
        """Features vacíos para ventanas sin datos."""
        return {
            "window_size_days": window_size,
            "num_active_days": 0,
            "coverage": 0.0,
            "messages_total": 0,
            "messages_mean": 0.0,
            "messages_std": 0.0,
            "words_total": 0,
            "words_mean": 0.0,
            "ttr_mean": 0.0,
            "sentiment_mean": 0.0,
            "sentiment_std": 0.0,
            "active_hours_mean": 0.0,
            "night_ratio_mean": 0.0,
            "messages_trend": 0.0,
            "sentiment_trend": 0.0,
        }


class LabelGenerator:
    """Genera labels para entrenamiento supervisado."""

    def __init__(self, config: ETLConfig):
        self.config = config

    def generate_labels(
        self,
        dates: pd.DatetimeIndex,
        events_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Genera labels binarias para cada fecha y horizonte.
        
        Para cada fecha, indica si hay un evento en los próximos N días.
        """
        labels = []
        
        event_dates = set(events_df["date"].dt.date) if not events_df.empty else set()
        
        for date in dates:
            date_labels = {"date": date}
            
            for horizon in self.config.prediction_horizons:
                # Verificar si hay evento en los próximos `horizon` días
                future_dates = [
                    date.date() + timedelta(days=d)
                    for d in range(1, horizon + 1)
                ]
                has_event = any(d in event_dates for d in future_dates)
                date_labels[f"relapse_in_{horizon}d"] = int(has_event)
            
            labels.append(date_labels)
        
        return pd.DataFrame(labels)


# ══════════════════════════════════════════════════════════════════════════════
# LOADER (L)
# ══════════════════════════════════════════════════════════════════════════════


class DataLoader:
    """Carga datos procesados a diferentes destinos."""

    def __init__(self, config: ETLConfig):
        self.config = config

    def save(
        self,
        daily_features: pd.DataFrame,
        window_features: pd.DataFrame,
        labels: pd.DataFrame,
        metadata: Dict[str, Any],
    ):
        """Guarda todos los artefactos procesados."""
        output_dir = self.config.output_path / self.config.patient_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar features diarios
        daily_path = output_dir / f"daily_features.{self.config.export_format}"
        self._save_df(daily_features, daily_path)
        logger.info(f"Features diarios guardados: {daily_path}")
        
        # Guardar features por ventana
        window_path = output_dir / f"window_features.{self.config.export_format}"
        self._save_df(window_features, window_path)
        logger.info(f"Features por ventana guardados: {window_path}")
        
        # Guardar labels
        labels_path = output_dir / f"labels.{self.config.export_format}"
        self._save_df(labels, labels_path)
        logger.info(f"Labels guardados: {labels_path}")
        
        # Guardar metadata
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata guardado: {meta_path}")
        
        # Dataset combinado para entrenamiento
        if not window_features.empty and not labels.empty:
            combined = window_features.merge(labels, on="date", how="left")
            combined_path = output_dir / f"training_dataset.{self.config.export_format}"
            self._save_df(combined, combined_path)
            logger.info(f"Dataset de entrenamiento: {combined_path}")

    def _save_df(self, df: pd.DataFrame, path: Path):
        """Guarda DataFrame en el formato configurado."""
        if self.config.export_format == "parquet":
            df.to_parquet(path, index=False)
        elif self.config.export_format == "csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Formato no soportado: {self.config.export_format}")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════


class ETLPipeline:
    """Pipeline ETL completo."""

    def __init__(self, config: ETLConfig):
        self.config = config
        self.message_extractor = MessageExtractor(config)
        self.events_extractor = EventsExtractor(config)
        self.feature_extractor = FeatureExtractor(config)
        self.aggregator = ActivityAggregator(config)
        self.label_generator = LabelGenerator(config)
        self.loader = DataLoader(config)

    def run(self) -> Dict[str, Any]:
        """Ejecuta el pipeline completo."""
        logger.info(f"Iniciando ETL para paciente: {self.config.patient_id}")
        
        # ═══ EXTRACT ═══
        logger.info("=== FASE EXTRACT ===")
        
        messages_df = self.message_extractor.extract(self.config.input_path)
        logger.info(f"Mensajes extraídos: {len(messages_df)}")
        
        events_df = pd.DataFrame()
        if self.config.events_path and self.config.events_path.exists():
            events_df = self.events_extractor.extract(self.config.events_path)
        
        # ═══ TRANSFORM ═══
        logger.info("=== FASE TRANSFORM ===")
        
        # Extraer features por mensaje
        message_features = []
        for _, row in messages_df.iterrows():
            features = self.feature_extractor.extract_message_features(row)
            if features:
                message_features.append(features)
        
        features_df = pd.DataFrame(message_features)
        logger.info(f"Mensajes con features: {len(features_df)}")
        
        if features_df.empty:
            logger.error("No se pudieron extraer features de ningún mensaje")
            return {"status": "error", "message": "No features extracted"}
        
        # Agregar a nivel diario
        daily_df = self.aggregator.aggregate_daily(features_df)
        logger.info(f"Días con actividad: {len(daily_df)}")
        
        # Calcular ventanas para cada día
        window_records = []
        for date in daily_df["date"]:
            windows = self.aggregator.compute_windows(daily_df, date)
            for window_size, window_features in windows.items():
                record = {"date": date, **window_features}
                window_records.append(record)
        
        window_df = pd.DataFrame(window_records)
        logger.info(f"Registros de ventanas: {len(window_df)}")
        
        # Generar labels
        labels_df = self.label_generator.generate_labels(
            daily_df["date"],
            events_df,
        )
        logger.info(f"Labels generados: {len(labels_df)}")
        
        # Estadísticas de labels
        for horizon in self.config.prediction_horizons:
            col = f"relapse_in_{horizon}d"
            if col in labels_df.columns:
                positive_rate = labels_df[col].mean()
                logger.info(f"  {col}: {positive_rate:.2%} positivos")
        
        # ═══ LOAD ═══
        logger.info("=== FASE LOAD ===")
        
        metadata = {
            "patient_id": self.config.patient_id,
            "patient_id_hash": self.config.patient_id_hash,
            "processed_at": datetime.now().isoformat(),
            "total_messages": len(messages_df),
            "total_days": len(daily_df),
            "date_range": {
                "start": str(daily_df["date"].min()),
                "end": str(daily_df["date"].max()),
            },
            "events_count": len(events_df),
            "window_sizes": self.config.window_sizes,
            "prediction_horizons": self.config.prediction_horizons,
        }
        
        self.loader.save(daily_df, window_df, labels_df, metadata)
        
        logger.info("=== ETL COMPLETADO ===")
        
        return {
            "status": "success",
            "patient_id": self.config.patient_id,
            "messages_processed": len(features_df),
            "days_processed": len(daily_df),
            "events_loaded": len(events_df),
            "output_path": str(self.config.output_path / self.config.patient_id),
        }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="ETL Pipeline para datos de pacientes EM Predictor"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Archivo de mensajes (JSON, JSONL, CSV, TXT)",
    )
    parser.add_argument(
        "--events", "-e",
        type=Path,
        default=None,
        help="Archivo CSV con eventos clínicos",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/processed"),
        help="Directorio de salida",
    )
    parser.add_argument(
        "--patient-id", "-p",
        type=str,
        default="patient_001",
        help="Identificador del paciente",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["parquet", "csv"],
        default="parquet",
        help="Formato de exportación",
    )
    parser.add_argument(
        "--language", "-l",
        choices=["es", "en"],
        default="es",
        help="Idioma de los mensajes",
    )
    
    args = parser.parse_args()
    
    config = ETLConfig(
        input_path=args.input,
        events_path=args.events,
        output_path=args.output,
        patient_id=args.patient_id,
        export_format=args.format,
        language=args.language,
    )
    
    pipeline = ETLPipeline(config)
    result = pipeline.run()
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

