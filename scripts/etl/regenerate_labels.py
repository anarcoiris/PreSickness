#!/usr/bin/env python3
"""
Re-genera labels usando solo clusters de brote confirmados.

Uso:
    python regenerate_labels.py --data-path data/processed/paciente1 --clusters datos/paciente1_events_auto_clusters.csv
"""

import argparse
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd


def load_clusters(path: Path) -> pd.DataFrame:
    """Carga clusters y filtra solo los probables brotes."""
    df = pd.read_csv(path)
    
    # Filtrar solo clusters con is_probable_relapse = True
    df = df[df["is_probable_relapse"] == True].copy()
    
    # Convertir fechas
    df["peak_date"] = pd.to_datetime(df["peak_date"])
    
    return df


def regenerate_labels(
    window_features: pd.DataFrame,
    clusters_df: pd.DataFrame,
    horizons: list = [7, 14, 30],
) -> pd.DataFrame:
    """
    Regenera labels usando fechas de clusters.
    """
    labels = []
    
    # Usar peak_date de clusters como eventos
    event_dates = set(clusters_df["peak_date"].dt.date)
    
    for date in window_features["date"]:
        if isinstance(date, str):
            date = pd.Timestamp(date)
        
        row = {"date": date}
        
        for horizon in horizons:
            # Verificar si hay evento en próximos N días
            future_dates = [
                (date + timedelta(days=d)).date() 
                for d in range(1, horizon + 1)
            ]
            has_event = any(d in event_dates for d in future_dates)
            row[f"relapse_in_{horizon}d"] = int(has_event)
        
        labels.append(row)
    
    return pd.DataFrame(labels)


def main():
    parser = argparse.ArgumentParser(description="Regenerar labels con clusters")
    parser.add_argument("--data-path", "-d", type=Path, required=True)
    parser.add_argument("--clusters", "-c", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=None)
    
    args = parser.parse_args()
    
    # Cargar datos
    window_path = args.data_path / "window_features.parquet"
    window_df = pd.read_parquet(window_path)
    print(f"Window features: {window_df.shape}")
    
    # Cargar clusters
    clusters_df = load_clusters(args.clusters)
    print(f"Clusters probables: {len(clusters_df)}")
    print(f"Fechas de pico: {clusters_df['peak_date'].tolist()}")
    
    # Regenerar labels
    labels_df = regenerate_labels(window_df, clusters_df)
    
    # Estadísticas
    for h in [7, 14, 30]:
        col = f"relapse_in_{h}d"
        rate = labels_df[col].mean()
        print(f"{col}: {rate:.2%} positivos")
    
    # Guardar
    output_dir = args.output or args.data_path
    
    # Labels
    labels_path = output_dir / "labels_clusters.parquet"
    labels_df.to_parquet(labels_path, index=False)
    print(f"Labels guardados: {labels_path}")
    
    # Training dataset combinado
    combined = window_df.merge(labels_df, on="date", how="left")
    combined_path = output_dir / "training_dataset_clusters.parquet"
    combined.to_parquet(combined_path, index=False)
    print(f"Training dataset: {combined_path}")
    
    return labels_df


if __name__ == "__main__":
    main()
