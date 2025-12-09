#!/usr/bin/env python3
"""
Feature Engineering Avanzado para EM-Predictor.

Añade:
- Lag features (valores de días anteriores)
- Rolling statistics (media, std, tendencia en ventanas)
- Features de cambio (diferencias, ratios)

Uso:
    python feature_engineering.py --data-path data/processed/paciente1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_lag_features(df: pd.DataFrame, columns: list, lags: list = [1, 3, 7]) -> pd.DataFrame:
    """
    Añade features con valores de días anteriores.
    
    Args:
        df: DataFrame con columna 'date' ordenado
        columns: Columnas a las que añadir lags
        lags: Lista de días de lag (ej: [1, 3, 7])
    
    Returns:
        DataFrame con nuevas columnas lag_*
    """
    df = df.sort_values("date").copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    return df


def add_rolling_features(
    df: pd.DataFrame, 
    columns: list, 
    windows: list = [3, 7, 14]
) -> pd.DataFrame:
    """
    Añade estadísticas en ventanas móviles.
    
    Args:
        df: DataFrame ordenado por date
        columns: Columnas para calcular rolling
        windows: Tamaños de ventana en días
    
    Returns:
        DataFrame con nuevas columnas rolling_*
    """
    df = df.sort_values("date").copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        for window in windows:
            # Media móvil
            df[f"{col}_roll{window}_mean"] = df[col].rolling(window=window, min_periods=1).mean()
            
            # Desviación estándar móvil
            df[f"{col}_roll{window}_std"] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
            
            # Tendencia (pendiente lineal)
            df[f"{col}_roll{window}_trend"] = df[col].rolling(window=window, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
            ).fillna(0)
    
    return df


def add_change_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Añade features de cambio (diferencias y ratios).
    """
    df = df.sort_values("date").copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        # Cambio absoluto día a día
        df[f"{col}_diff1"] = df[col].diff(1).fillna(0)
        
        # Cambio porcentual
        df[f"{col}_pct1"] = df[col].pct_change(1).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Cambio en 7 días
        df[f"{col}_diff7"] = df[col].diff(7).fillna(0)
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features de interacción entre variables.
    """
    df = df.copy()
    
    # Interacciones útiles para predicción de brotes
    if "sentiment_proxy_mean" in df.columns and "messages_count" in df.columns:
        df["sentiment_x_volume"] = df["sentiment_proxy_mean"] * df["messages_count"]
    
    if "word_count_mean" in df.columns and "type_token_ratio_mean" in df.columns:
        df["complexity_score"] = df["word_count_mean"] * df["type_token_ratio_mean"]
    
    if "night_ratio" in df.columns and "messages_count" in df.columns:
        df["night_activity"] = df["night_ratio"] * df["messages_count"]
    
    return df


def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    """
    # Identificar columnas numéricas para procesar
    exclude = ["date", "first_message", "last_message", 
               "relapse_in_7d", "relapse_in_14d", "relapse_in_30d"]
    
    numeric_cols = [c for c in df.columns if c not in exclude 
                   and df[c].dtype in ["float64", "int64", "float32", "int32"]]
    
    # Seleccionar columnas principales para features avanzados
    # (evitar explosión de dimensionalidad)
    main_cols = numeric_cols[:8]  # Top 8 features
    
    if verbose:
        print(f"📊 Columnas originales: {len(df.columns)}")
        print(f"🔧 Features principales para engineering: {main_cols}")
    
    original_cols = len(df.columns)
    
    # Lag features
    df = add_lag_features(df, main_cols, lags=[1, 3, 7])
    
    # Rolling features
    df = add_rolling_features(df, main_cols[:5], windows=[3, 7])
    
    # Change features
    df = add_change_features(df, main_cols[:5])
    
    # Interactions
    df = add_interaction_features(df)
    
    if verbose:
        new_cols = len(df.columns) - original_cols
        print(f"✅ Features nuevos añadidos: {new_cols}")
        print(f"📊 Total columnas: {len(df.columns)}")
    
    # Llenar NaN con 0
    df = df.fillna(0)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering")
    parser.add_argument("--data-path", "-d", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=None)
    
    args = parser.parse_args()
    output_dir = args.output or args.data_path
    
    # Cargar datos
    clusters_path = args.data_path / "training_dataset_clusters.parquet"
    if clusters_path.exists():
        df = pd.read_parquet(clusters_path)
        print(f"📁 Cargado: training_dataset_clusters.parquet")
    else:
        df = pd.read_parquet(args.data_path / "training_dataset.parquet")
        print(f"📁 Cargado: training_dataset.parquet")
    
    print(f"📊 Shape original: {df.shape}")
    
    # Engineer features
    df_engineered = engineer_features(df, verbose=True)
    
    # Guardar
    output_path = output_dir / "training_dataset_engineered.parquet"
    df_engineered.to_parquet(output_path, index=False)
    print(f"\n💾 Guardado: {output_path}")
    
    # Mostrar nuevas columnas
    new_cols = [c for c in df_engineered.columns if c not in df.columns]
    print(f"\n📋 Nuevas columnas ({len(new_cols)}):")
    for col in new_cols[:20]:
        print(f"   - {col}")
    if len(new_cols) > 20:
        print(f"   ... y {len(new_cols) - 20} más")


if __name__ == "__main__":
    main()
