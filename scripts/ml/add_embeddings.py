#!/usr/bin/env python3
"""
Pipeline de Embeddings para enriquecer features.

Procesa mensajes del paciente y genera embeddings usando Sentence Transformers.

Uso:
    python add_embeddings.py --data-path data/processed/paciente1 --preset fast
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("⚠️ sentence-transformers no instalado")


PRESETS = {
    "fast": "distiluse-base-multilingual-cased-v1",
    "balanced": "paraphrase-multilingual-mpnet-base-v2",
    "quality": "intfloat/multilingual-e5-large",
}


def load_messages(data_path: Path, input_path: Path = None) -> pd.DataFrame:
    """Carga mensajes del archivo de WhatsApp original."""
    # Intentar cargar mensajes guardados
    messages_path = data_path / "messages.parquet"
    if messages_path.exists():
        return pd.read_parquet(messages_path)
    
    # Buscar archivo original
    if input_path and input_path.exists():
        return parse_whatsapp_export(input_path)
    
    # Buscar en datos/
    datos_path = data_path.parent.parent / "datos" / "paciente1_whatsapp.txt"
    if datos_path.exists():
        return parse_whatsapp_export(datos_path)
    
    return None


def parse_whatsapp_export(path: Path) -> pd.DataFrame:
    """Parsea export de WhatsApp."""
    import re
    
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2})(?::\d{2})?\s*(?:[ap]\.?\s*m\.?\s*)?-\s*([^:]+):\s*(.+)"
    
    messages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(pattern, line.strip(), re.IGNORECASE)
            if match:
                date_str, time_str, sender, text = match.groups()
                messages.append({
                    "date": date_str,
                    "time": time_str,
                    "sender": sender.strip(),
                    "text": text.strip()
                })
    
    if not messages:
        return None
    
    df = pd.DataFrame(messages)
    
    # Parsear fechas
    for fmt in ["%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y"]:
        try:
            df["date"] = pd.to_datetime(df["date"], format=fmt)
            break
        except:
            continue
    
    return df


def compute_daily_embeddings(
    messages_df: pd.DataFrame,
    model: SentenceTransformer,
    target_sender: str = "<?>",
) -> pd.DataFrame:
    """
    Calcula embeddings agregados por día.
    """
    if messages_df is None or messages_df.empty:
        return None
    
    # Filtrar solo mensajes del paciente
    if target_sender:
        patient_msgs = messages_df[messages_df["sender"] == target_sender]
    else:
        patient_msgs = messages_df
    
    if patient_msgs.empty:
        print("⚠️ No se encontraron mensajes del paciente")
        return None
    
    # Filtrar mensajes de sistema
    system_patterns = [
        "<Multimedia omitido>",
        "Los mensajes y las llamadas están cifrados",
        "imagen omitida",
        "audio omitido",
    ]
    
    for pattern in system_patterns:
        patient_msgs = patient_msgs[~patient_msgs["text"].str.contains(pattern, case=False, na=False)]
    
    # Agrupar por día
    patient_msgs = patient_msgs.copy()
    if not pd.api.types.is_datetime64_any_dtype(patient_msgs["date"]):
        patient_msgs["date"] = pd.to_datetime(patient_msgs["date"])
    
    patient_msgs["date_only"] = patient_msgs["date"].dt.date
    
    daily_texts = patient_msgs.groupby("date_only")["text"].apply(list).to_dict()
    
    print(f"📅 Procesando {len(daily_texts)} días...")
    
    dim = model.get_sentence_embedding_dimension()
    results = []
    
    for date, texts in daily_texts.items():
        valid_texts = [t for t in texts if t and len(str(t)) > 3]
        
        if not valid_texts:
            embedding = np.zeros(dim)
        else:
            # Prefijo para modelos E5
            if "e5" in model._modules.get("0", "").__class__.__name__.lower():
                valid_texts = [f"query: {t}" for t in valid_texts]
            
            embeddings = model.encode(valid_texts, show_progress_bar=False)
            embedding = embeddings.mean(axis=0)
        
        results.append({
            "date": pd.Timestamp(date),
            **{f"emb_{i}": v for i, v in enumerate(embedding)}
        })
    
    return pd.DataFrame(results)


def enrich_dataset_with_embeddings(
    data_path: Path,
    preset: str = "fast",
    output: Path = None,
) -> pd.DataFrame:
    """
    Enriquece el training dataset con embeddings.
    """
    if not ST_AVAILABLE:
        print("❌ sentence-transformers requerido")
        return None
    
    output_dir = output or data_path
    
    # Cargar mensajes
    messages_df = load_messages(data_path)
    
    if messages_df is None:
        print("⚠️ No se pudieron cargar mensajes. Generando embeddings dummy...")
        return create_dummy_embeddings(data_path, output_dir)
    
    print(f"📨 Mensajes cargados: {len(messages_df)}")
    
    # Cargar modelo
    model_name = PRESETS.get(preset, preset)
    print(f"🔧 Cargando modelo: {model_name}...")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"📐 Dimensión: {dim}")
    
    # Calcular embeddings diarios
    daily_embeddings = compute_daily_embeddings(messages_df, model)
    
    if daily_embeddings is None:
        return create_dummy_embeddings(data_path, output_dir)
    
    print(f"✅ Embeddings calculados: {len(daily_embeddings)} días")
    
    # Cargar training dataset
    df = load_training_dataset(data_path)
    
    # Merge
    df["date"] = pd.to_datetime(df["date"]).dt.date
    daily_embeddings["date"] = pd.to_datetime(daily_embeddings["date"]).dt.date
    
    enriched = df.merge(daily_embeddings, on="date", how="left")
    
    # Fill NaN embeddings
    emb_cols = [c for c in enriched.columns if c.startswith("emb_")]
    enriched[emb_cols] = enriched[emb_cols].fillna(0)
    
    # Guardar
    output_path = output_dir / "training_dataset_with_embeddings.parquet"
    enriched.to_parquet(output_path, index=False)
    print(f"💾 Guardado: {output_path}")
    
    # Metadata
    meta = {
        "model": model_name,
        "dim": dim,
        "days_with_embeddings": int((enriched[emb_cols[0]] != 0).sum()),
    }
    with open(output_dir / "embeddings_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    return enriched


def load_training_dataset(data_path: Path) -> pd.DataFrame:
    """Carga el mejor training dataset disponible."""
    options = [
        "training_dataset_engineered.parquet",
        "training_dataset_clusters.parquet",
        "training_dataset.parquet",
    ]
    
    for name in options:
        path = data_path / name
        if path.exists():
            print(f"📁 Base dataset: {name}")
            return pd.read_parquet(path)
    
    raise FileNotFoundError("No training dataset found")


def create_dummy_embeddings(data_path: Path, output_dir: Path) -> pd.DataFrame:
    """Crea dataset con embeddings aleatorios para testing."""
    print("📦 Creando embeddings dummy para testing...")
    
    df = load_training_dataset(data_path)
    
    # Añadir 32 columnas de embeddings aleatorios (seed fijo)
    np.random.seed(42)
    dim = 32
    
    for i in range(dim):
        df[f"emb_{i}"] = np.random.randn(len(df)) * 0.1
    
    output_path = output_dir / "training_dataset_with_embeddings.parquet"
    df.to_parquet(output_path, index=False)
    print(f"💾 Guardado (dummy): {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Add Embeddings to Dataset")
    parser.add_argument("--data-path", "-d", type=Path, required=True)
    parser.add_argument("--preset", "-p", default="fast",
                       choices=["fast", "balanced", "quality"])
    parser.add_argument("--output", "-o", type=Path, default=None)
    
    args = parser.parse_args()
    
    print(f"📁 Data path: {args.data_path}")
    print(f"🎯 Preset: {args.preset}")
    
    result = enrich_dataset_with_embeddings(
        args.data_path,
        preset=args.preset,
        output=args.output,
    )
    
    if result is not None:
        print(f"\n✅ Dataset enriquecido: {result.shape}")


if __name__ == "__main__":
    main()
