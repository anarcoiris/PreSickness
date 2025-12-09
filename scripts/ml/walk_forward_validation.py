#!/usr/bin/env python3
"""
Walk-Forward Validation para EM-Predictor.

Simula escenario real de predicción: entrena con datos pasados,
evalúa en datos futuros, avanza la ventana.

Uso:
    python walk_forward_validation.py --data-path data/processed/paciente1
"""

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def load_data(data_path: Path) -> pd.DataFrame:
    """Carga dataset con labels de clusters."""
    # Intentar primero el dataset con clusters
    clusters_path = data_path / "training_dataset_clusters.parquet"
    if clusters_path.exists():
        df = pd.read_parquet(clusters_path)
        print(f"✅ Usando dataset con clusters: {df.shape}")
    else:
        df = pd.read_parquet(data_path / "training_dataset.parquet")
        print(f"⚠️ Usando dataset original: {df.shape}")
    
    return df


def get_features(df: pd.DataFrame, target: str = "relapse_in_14d") -> tuple:
    """Extrae features y target."""
    exclude = [
        "date", "first_message", "last_message",
        "relapse_in_7d", "relapse_in_14d", "relapse_in_30d"
    ]
    
    features = [c for c in df.columns if c not in exclude 
                and df[c].dtype in ["float64", "int64", "float32", "int32"]]
    
    return features, target


def walk_forward_validation(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    train_window: int = 60,  # días de entrenamiento
    test_window: int = 14,   # días de test
    step: int = 7,           # avance por iteración
    model_class=RandomForestClassifier,
    model_params: Dict = None,
) -> pd.DataFrame:
    """
    Ejecuta walk-forward validation.
    
    Args:
        df: DataFrame con features y target
        features: Lista de columnas de features
        target: Columna objetivo
        train_window: Días para entrenar
        test_window: Días para evaluar
        step: Días para avanzar ventana
        model_class: Clase del modelo
        model_params: Parámetros del modelo
    
    Returns:
        DataFrame con métricas por fold
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    
    model_params = model_params or {}
    
    results = []
    dates = df["date"].unique()
    
    print(f"\n📊 Walk-Forward Validation")
    print(f"   Train window: {train_window} días")
    print(f"   Test window: {test_window} días")
    print(f"   Step: {step} días")
    print(f"   Total fechas: {len(dates)}")
    
    fold = 0
    start_idx = 0
    
    while start_idx + train_window + test_window <= len(dates):
        train_end_idx = start_idx + train_window
        test_end_idx = train_end_idx + test_window
        
        train_dates = dates[start_idx:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]
        
        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"].isin(test_dates)]
        
        if len(train_df) < 10 or len(test_df) < 5:
            start_idx += step
            continue
        
        X_train = train_df[features].fillna(0)
        y_train = train_df[target]
        X_test = test_df[features].fillna(0)
        y_test = test_df[target]
        
        # Verificar que hay ambas clases
        if len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
            start_idx += step
            continue
        
        # Entrenar
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Predecir
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas
        try:
            auroc = roc_auc_score(y_test, y_proba)
            auprc = average_precision_score(y_test, y_proba)
            brier = brier_score_loss(y_test, y_proba)
        except:
            auroc, auprc, brier = 0.5, 0.5, 0.25
        
        results.append({
            "fold": fold,
            "train_start": pd.Timestamp(train_dates[0]),
            "train_end": pd.Timestamp(train_dates[-1]),
            "test_start": pd.Timestamp(test_dates[0]),
            "test_end": pd.Timestamp(test_dates[-1]),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_positive_rate": y_train.mean(),
            "test_positive_rate": y_test.mean(),
            "auroc": auroc,
            "auprc": auprc,
            "brier": brier,
        })
        
        fold += 1
        start_idx += step
    
    return pd.DataFrame(results)


def plot_walk_forward_results(results_df: pd.DataFrame, output_path: Path):
    """Genera plots de resultados de walk-forward."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # AUROC over time
    ax1 = axes[0]
    ax1.plot(results_df["test_start"], results_df["auroc"], "b-o", label="AUROC", linewidth=2)
    ax1.axhline(y=0.65, color="r", linestyle="--", label="Target (0.65)")
    ax1.axhline(y=results_df["auroc"].mean(), color="g", linestyle=":", label=f"Mean ({results_df['auroc'].mean():.3f})")
    ax1.fill_between(results_df["test_start"], 0.5, results_df["auroc"], alpha=0.3)
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(0.4, 1.0)
    ax1.legend(loc="upper right")
    ax1.set_title("Walk-Forward Validation: Evolución de AUROC")
    ax1.grid(True, alpha=0.3)
    
    # AUPRC over time
    ax2 = axes[1]
    ax2.plot(results_df["test_start"], results_df["auprc"], "g-o", label="AUPRC", linewidth=2)
    ax2.axhline(y=results_df["auprc"].mean(), color="orange", linestyle=":", label=f"Mean ({results_df['auprc'].mean():.3f})")
    ax2.fill_between(results_df["test_start"], 0, results_df["auprc"], alpha=0.3, color="green")
    ax2.set_ylabel("AUPRC")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    # Positive rate
    ax3 = axes[2]
    ax3.bar(results_df["test_start"], results_df["test_positive_rate"], 
            width=5, alpha=0.7, label="Test Positive Rate")
    ax3.axhline(y=results_df["test_positive_rate"].mean(), color="r", linestyle=":", 
                label=f"Mean ({results_df['test_positive_rate'].mean():.2%})")
    ax3.set_ylabel("Positive Rate")
    ax3.set_xlabel("Test Period Start")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"📈 Plot guardado: {output_path}")


def plot_metrics_summary(results_df: pd.DataFrame, output_path: Path):
    """Genera summary boxplot de métricas."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ["auroc", "auprc", "brier"]
    data = [results_df[m].values for m in metrics]
    
    bp = ax.boxplot(data, labels=["AUROC", "AUPRC", "Brier Score"], patch_artist=True)
    
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Añadir puntos individuales
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.5, s=30)
    
    ax.axhline(y=0.65, color="r", linestyle="--", alpha=0.5, label="AUROC Target")
    ax.set_ylabel("Score")
    ax.set_title(f"Walk-Forward Validation Summary ({len(results_df)} folds)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"📊 Summary plot guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument("--data-path", "-d", type=Path, required=True)
    parser.add_argument("--target", "-t", default="relapse_in_14d")
    parser.add_argument("--train-window", type=int, default=60)
    parser.add_argument("--test-window", type=int, default=14)
    parser.add_argument("--step", type=int, default=7)
    parser.add_argument("--output", "-o", type=Path, default=None)
    
    args = parser.parse_args()
    output_dir = args.output or args.data_path
    
    # Cargar datos
    df = load_data(args.data_path)
    features, target = get_features(df, args.target)
    print(f"Features: {len(features)}")
    print(f"Target: {target} (positive rate: {df[target].mean():.2%})")
    
    # Walk-forward con RandomForest
    print("\n" + "="*60)
    print("RandomForest Walk-Forward")
    print("="*60)
    
    rf_results = walk_forward_validation(
        df, features, target,
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.step,
        model_class=RandomForestClassifier,
        model_params={"n_estimators": 100, "max_depth": 10, "class_weight": "balanced", "n_jobs": -1}
    )
    
    if not rf_results.empty:
        print(f"\n📈 Resultados ({len(rf_results)} folds):")
        print(f"   AUROC: {rf_results['auroc'].mean():.4f} ± {rf_results['auroc'].std():.4f}")
        print(f"   AUPRC: {rf_results['auprc'].mean():.4f} ± {rf_results['auprc'].std():.4f}")
        print(f"   Brier: {rf_results['brier'].mean():.4f} ± {rf_results['brier'].std():.4f}")
        
        # Guardar resultados
        rf_results.to_csv(output_dir / "walk_forward_results.csv", index=False)
        print(f"\n💾 Resultados guardados: {output_dir / 'walk_forward_results.csv'}")
        
        # Plots
        plot_walk_forward_results(rf_results, output_dir / "walk_forward_auroc.png")
        plot_metrics_summary(rf_results, output_dir / "walk_forward_summary.png")
    else:
        print("⚠️ No se pudieron generar suficientes folds")
    
    return rf_results


if __name__ == "__main__":
    main()
