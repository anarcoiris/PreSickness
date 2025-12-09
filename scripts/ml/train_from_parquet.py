#!/usr/bin/env python3
"""
Training Pipeline para datos procesados por ETL (Parquet).

Versión simplificada que no requiere conexión a DB.
Usa directamente los outputs del pipeline ETL.

Uso:
    python train_from_parquet.py --data-path data/processed/paciente1
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════


def load_training_data(data_path: Path) -> pd.DataFrame:
    """Carga dataset de entrenamiento desde Parquet."""
    training_path = data_path / "training_dataset.parquet"
    
    if not training_path.exists():
        raise FileNotFoundError(f"No encontrado: {training_path}")
    
    df = pd.read_parquet(training_path)
    logger.info(f"Dataset cargado: {df.shape}")
    
    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "relapse_in_14d",
    exclude_cols: Optional[List[str]] = None,
) -> tuple:
    """
    Prepara features (X) y target (y) para entrenamiento.
    
    Returns:
        X, y, feature_names
    """
    exclude_cols = exclude_cols or []
    
    # Columnas a excluir (no son features)
    non_feature_cols = [
        "date", "first_message", "last_message",
        "relapse_in_7d", "relapse_in_14d", "relapse_in_30d",
    ] + exclude_cols
    
    # Identificar columnas de features
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    
    # Filtrar columnas numéricas
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Features seleccionados: {len(numeric_cols)}")
    
    # Preparar X e y
    X = df[numeric_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None
    
    # Manejar NaN
    X = X.fillna(0)
    
    return X, y, numeric_cols


# ══════════════════════════════════════════════════════════════════════════════
# MODELOS BASELINE
# ══════════════════════════════════════════════════════════════════════════════


def train_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict:
    """
    Entrena múltiples modelos baseline y compara métricas.
    
    Returns:
        Diccionario con resultados por modelo
    """
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_depth=10,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Entrenando {name}...")
        
        # Entrenar
        if name == "LogisticRegression":
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Métricas
        try:
            auroc = roc_auc_score(y_val, y_pred_proba)
            auprc = average_precision_score(y_val, y_pred_proba)
            brier = brier_score_loss(y_val, y_pred_proba)
        except Exception as e:
            logger.warning(f"Error calculando métricas para {name}: {e}")
            auroc, auprc, brier = 0.5, 0.5, 0.25
        
        results[name] = {
            "model": model,
            "scaler": scaler if name == "LogisticRegression" else None,
            "metrics": {
                "auroc": round(auroc, 4),
                "auprc": round(auprc, 4),
                "brier_score": round(brier, 4),
            }
        }
        
        logger.info(f"  {name}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, Brier={brier:.4f}")
    
    return results


def get_feature_importance(model, feature_names: List[str], model_name: str) -> pd.DataFrame:
    """Extrae importancia de features del modelo."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()
    
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TFT (OPCIONAL)
# ══════════════════════════════════════════════════════════════════════════════


def train_tft_if_available(
    df: pd.DataFrame,
    target_col: str,
    config: Dict,
) -> Optional[Dict]:
    """
    Intenta entrenar TFT si pytorch_forecasting está disponible.
    """
    try:
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping
        import torch
    except ImportError:
        logger.warning("pytorch_forecasting no instalado. Saltando TFT.")
        return None
    
    logger.info("Preparando dataset para TFT...")
    
    # Añadir columnas necesarias
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["time_idx"] = range(len(df))
    df["group"] = "patient1"  # Single patient for now
    
    # Features numéricos
    exclude = ["date", "time_idx", "group", target_col, 
               "relapse_in_7d", "relapse_in_14d", "relapse_in_30d",
               "first_message", "last_message"]
    
    numeric_features = [c for c in df.columns if c not in exclude 
                       and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
    # Limitar a 20 features para evitar problemas
    numeric_features = numeric_features[:20]
    
    # Rellenar NaN
    df[numeric_features] = df[numeric_features].fillna(0)
    df[target_col] = df[target_col].fillna(0).astype(float)
    
    # Split temporal
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    if len(train_df) < 30 or len(val_df) < 10:
        logger.warning("Dataset muy pequeño para TFT")
        return None
    
    try:
        # Crear dataset
        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=target_col,
            group_ids=["group"],
            min_encoder_length=7,
            max_encoder_length=min(30, len(train_df) // 2),
            min_prediction_length=1,
            max_prediction_length=7,
            time_varying_unknown_reals=numeric_features,
            target_normalizer=GroupNormalizer(groups=["group"]),
            add_relative_time_idx=True,
            add_target_scales=True,
        )
        
        validation = TimeSeriesDataSet.from_dataset(
            training, val_df, predict=True, stop_randomization=True
        )
        
        # Dataloaders
        train_dataloader = training.to_dataloader(train=True, batch_size=16, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=16, num_workers=0)
        
        # Modelo
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=config.get("learning_rate", 0.01),
            hidden_size=config.get("hidden_size", 16),
            attention_head_size=config.get("attention_head_size", 2),
            dropout=config.get("dropout", 0.1),
            hidden_continuous_size=config.get("hidden_continuous_size", 8),
            loss=QuantileLoss(),
            reduce_on_plateau_patience=2,
        )
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=config.get("max_epochs", 10),
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            gradient_clip_val=0.1,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
            enable_progress_bar=True,
            logger=False,
        )
        
        # Train
        logger.info("Entrenando TFT...")
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
        # Evaluar
        predictions = tft.predict(val_dataloader, mode="raw")
        
        return {
            "model": tft,
            "trainer": trainer,
            "metrics": {"status": "trained", "epochs": trainer.current_epoch}
        }
        
    except Exception as e:
        logger.error(f"Error entrenando TFT: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelos desde Parquet")
    parser.add_argument(
        "--data-path", "-d",
        type=Path,
        required=True,
        help="Directorio con outputs del ETL"
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        default="relapse_in_14d",
        choices=["relapse_in_7d", "relapse_in_14d", "relapse_in_30d"],
        help="Columna objetivo"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción para validación"
    )
    parser.add_argument(
        "--try-tft",
        action="store_true",
        help="Intentar entrenar TFT (requiere pytorch_forecasting)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Directorio para guardar resultados"
    )
    
    args = parser.parse_args()
    
    # Cargar datos
    df = load_training_data(args.data_path)
    
    # Verificar columna target
    if args.target not in df.columns:
        logger.error(f"Columna {args.target} no encontrada. Disponibles: {df.columns.tolist()}")
        return
    
    # Preparar features
    X, y, feature_names = prepare_features(df, target_col=args.target)
    
    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"Positive rate: {y.mean():.2%}")
    
    # Split temporal (mantener orden para series temporales)
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Entrenar baselines
    print("\n" + "="*60)
    print("ENTRENAMIENTO DE MODELOS BASELINE")
    print("="*60)
    
    results = train_baselines(X_train, y_train, X_val, y_val)
    
    # Mejor modelo
    best_model_name = max(results, key=lambda k: results[k]["metrics"]["auroc"])
    best_metrics = results[best_model_name]["metrics"]
    
    print("\n" + "="*60)
    print(f"MEJOR MODELO: {best_model_name}")
    print(f"  AUROC: {best_metrics['auroc']}")
    print(f"  AUPRC: {best_metrics['auprc']}")
    print(f"  Brier: {best_metrics['brier_score']}")
    print("="*60)
    
    # Feature importance
    best_model = results[best_model_name]["model"]
    importances = get_feature_importance(best_model, feature_names, best_model_name)
    
    if not importances.empty:
        print("\nTOP 10 FEATURES MÁS IMPORTANTES:")
        print(importances.head(10).to_string(index=False))
    
    # TFT opcional
    if args.try_tft:
        print("\n" + "="*60)
        print("INTENTANDO ENTRENAR TFT...")
        print("="*60)
        
        tft_result = train_tft_if_available(df, args.target, {
            "max_epochs": 10,
            "learning_rate": 0.01,
        })
        
        if tft_result:
            results["TFT"] = tft_result
    
    # Guardar resultados
    output_dir = args.output or args.data_path
    
    results_summary = {
        "data_path": str(args.data_path),
        "target": args.target,
        "samples": len(df),
        "features": len(feature_names),
        "positive_rate": float(y.mean()),
        "models": {
            name: res["metrics"] for name, res in results.items()
            if "metrics" in res
        },
        "best_model": best_model_name,
        "feature_importance": importances.head(20).to_dict("records") if not importances.empty else [],
    }
    
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\nResultados guardados en: {results_path}")
    
    return results


if __name__ == "__main__":
    main()
