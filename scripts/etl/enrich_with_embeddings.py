#!/usr/bin/env python3
"""
Enriquece el dataset procesado con embeddings LLM.

Añade embeddings de texto a los features existentes para mejorar
las predicciones del modelo TFT.

Uso:
    python enrich_with_embeddings.py --input data/processed/paciente1 --preset balanced
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from embeddings import EmbeddingExtractor, EmbeddingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_daily_features(input_dir: Path) -> pd.DataFrame:
    """Carga features diarios."""
    path = input_dir / "daily_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No encontrado: {path}")
    return pd.read_parquet(path)


def load_messages(input_dir: Path) -> pd.DataFrame:
    """
    Carga mensajes originales si están disponibles.
    Si no, reconstruye textos desde metadata.
    """
    # Intentar cargar mensajes crudos
    messages_path = input_dir / "messages.parquet"
    if messages_path.exists():
        return pd.read_parquet(messages_path)
    
    # Alternativa: usar el archivo original si está en metadata
    metadata_path = input_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        logger.info(f"Metadata: {metadata}")
    
    return None


def compute_daily_embeddings(
    messages_df: pd.DataFrame,
    extractor: EmbeddingExtractor,
    aggregation: str = "mean"
) -> pd.DataFrame:
    """
    Calcula embeddings agregados por día.
    
    Args:
        messages_df: DataFrame con columnas [date, text]
        extractor: Extractor de embeddings
        aggregation: Método de agregación ("mean", "max", "concat")
    
    Returns:
        DataFrame con [date, embedding_0, embedding_1, ...]
    """
    # Asegurar columna de fecha
    if "date" not in messages_df.columns:
        messages_df["date"] = messages_df["timestamp"].dt.date
    
    # Agrupar textos por día
    daily_texts = messages_df.groupby("date")["text"].apply(list).to_dict()
    
    logger.info(f"Procesando {len(daily_texts)} días...")
    
    results = []
    dim = extractor.embedding_dim
    
    for date, texts in daily_texts.items():
        # Filtrar textos válidos
        valid_texts = [t for t in texts if t and len(str(t)) > 3]
        
        if not valid_texts:
            # Embedding cero si no hay textos
            embedding = np.zeros(dim)
        else:
            # Codificar todos los textos del día
            embeddings = extractor.encode(valid_texts)
            
            # Agregar
            if aggregation == "mean":
                embedding = embeddings.mean(axis=0)
            elif aggregation == "max":
                embedding = embeddings.max(axis=0)
            else:
                embedding = embeddings.mean(axis=0)
        
        results.append({
            "date": pd.Timestamp(date),
            **{f"emb_{i}": v for i, v in enumerate(embedding)}
        })
    
    return pd.DataFrame(results)


def enrich_training_dataset(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    preset: str = "balanced",
    backend: str = "sentence-transformers",
) -> dict:
    """
    Enriquece el dataset de entrenamiento con embeddings.
    
    Args:
        input_dir: Directorio con outputs del ETL
        output_dir: Directorio de salida (default: mismo que input)
        preset: Preset del modelo (fast/balanced/quality/spanish)
        backend: Backend a usar
    
    Returns:
        Diccionario con estadísticas
    """
    output_dir = output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar dataset existente
    training_path = input_dir / "training_dataset.parquet"
    if not training_path.exists():
        raise FileNotFoundError(f"No encontrado: {training_path}")
    
    training_df = pd.read_parquet(training_path)
    logger.info(f"Dataset cargado: {training_df.shape}")
    
    # Cargar mensajes si están disponibles
    messages_df = load_messages(input_dir)
    
    if messages_df is None:
        logger.warning("Mensajes no disponibles. Saltando embeddings de texto.")
        return {"status": "skipped", "reason": "no_messages"}
    
    # Crear extractor
    logger.info(f"Inicializando extractor: {backend} / {preset}")
    extractor = EmbeddingExtractor(backend=backend, preset=preset)
    logger.info(f"Modelo: {extractor.model_name} (dim={extractor.embedding_dim})")
    
    # Calcular embeddings por día
    start_time = time.time()
    daily_embeddings = compute_daily_embeddings(messages_df, extractor)
    elapsed = time.time() - start_time
    logger.info(f"Embeddings calculados en {elapsed:.1f}s")
    
    # Merge con training dataset
    training_df["date"] = pd.to_datetime(training_df["date"]).dt.date
    daily_embeddings["date"] = daily_embeddings["date"].dt.date
    
    enriched_df = training_df.merge(daily_embeddings, on="date", how="left")
    
    # Rellenar NaN con 0
    emb_cols = [c for c in enriched_df.columns if c.startswith("emb_")]
    enriched_df[emb_cols] = enriched_df[emb_cols].fillna(0)
    
    logger.info(f"Dataset enriquecido: {enriched_df.shape}")
    
    # Guardar
    output_path = output_dir / "training_dataset_enriched.parquet"
    enriched_df.to_parquet(output_path, index=False)
    logger.info(f"Guardado: {output_path}")
    
    # Guardar metadata de embeddings
    emb_metadata = {
        "backend": backend,
        "preset": preset,
        "model": extractor.model_name,
        "embedding_dim": extractor.embedding_dim,
        "processing_time_seconds": elapsed,
        "num_days": len(daily_embeddings),
    }
    
    with open(output_dir / "embeddings_metadata.json", "w") as f:
        json.dump(emb_metadata, f, indent=2)
    
    return {
        "status": "success",
        "output_path": str(output_path),
        "embedding_dim": extractor.embedding_dim,
        "processing_time": elapsed,
        "original_shape": training_df.shape,
        "enriched_shape": enriched_df.shape,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Enriquece dataset con embeddings LLM"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Directorio con outputs del ETL"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Directorio de salida (default: mismo que input)"
    )
    parser.add_argument(
        "--preset", "-p",
        choices=["fast", "balanced", "quality", "spanish"],
        default="balanced",
        help="Preset del modelo"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["sentence-transformers", "openai"],
        default="sentence-transformers",
        help="Backend a usar"
    )
    
    args = parser.parse_args()
    
    result = enrich_training_dataset(
        input_dir=args.input,
        output_dir=args.output,
        preset=args.preset,
        backend=args.backend,
    )
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
