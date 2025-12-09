#!/usr/bin/env python3
"""
Optuna Tuning Simple para EM-Predictor.
Versión simplificada que maneja correctamente los errores.

Uso:
    python optuna_simple.py --data-path data/processed/paciente1 --n-trials 30
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("⚠️ Instala optuna: pip install optuna")
    exit(1)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def load_data(data_path: Path, target: str = "relapse_in_14d"):
    """Carga y prepara datos."""
    # Usar dataset con clusters si existe
    clusters_path = data_path / "training_dataset_clusters.parquet"
    if clusters_path.exists():
        df = pd.read_parquet(clusters_path)
    else:
        df = pd.read_parquet(data_path / "training_dataset.parquet")
    
    exclude = ["date", "first_message", "last_message",
               "relapse_in_7d", "relapse_in_14d", "relapse_in_30d"]
    
    features = [c for c in df.columns if c not in exclude 
                and df[c].dtype in ["float64", "int64", "float32", "int32"]]
    
    X = df[features].fillna(0)
    y = df[target]
    
    return X, y


def temporal_cv_score(model, X, y, n_splits=5):
    """Cross-validation temporal (no shuffle)."""
    scores = []
    fold_size = len(X) // n_splits
    
    for i in range(1, n_splits):
        train_end = i * fold_size
        val_end = train_end + fold_size
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        
        if len(y_train.unique()) < 2 or len(y_val.unique()) < 2:
            continue
        
        try:
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_proba)
            scores.append(score)
        except Exception:
            pass
    
    return np.mean(scores) if scores else 0.5


def objective_rf(trial, X, y):
    """Objetivo para RandomForest."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }
    model = RandomForestClassifier(**params)
    return temporal_cv_score(model, X, y)


def objective_gbm(trial, X, y):
    """Objetivo para GradientBoosting."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "random_state": 42,
    }
    model = GradientBoostingClassifier(**params)
    return temporal_cv_score(model, X, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", "-d", type=Path, required=True)
    parser.add_argument("--target", "-t", default="relapse_in_14d")
    parser.add_argument("--n-trials", "-n", type=int, default=30)
    parser.add_argument("--output", "-o", type=Path, default=None)
    
    args = parser.parse_args()
    output_dir = args.output or args.data_path
    
    # Cargar datos
    X, y = load_data(args.data_path, args.target)
    print(f"📊 Dataset: {X.shape}")
    print(f"🎯 Target: {args.target} ({y.mean():.1%} positivos)")
    
    results = {}
    
    # Optimizar RF
    print(f"\n{'='*50}")
    print("🔧 Optimizando RandomForest...")
    print("="*50)
    
    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(lambda t: objective_rf(t, X, y), n_trials=args.n_trials, show_progress_bar=True)
    
    print(f"✅ Mejor CV AUROC: {study_rf.best_value:.4f}")
    print(f"📋 Params: {study_rf.best_params}")
    
    # Evaluar en holdout
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    
    best_rf = RandomForestClassifier(**study_rf.best_params, class_weight="balanced", n_jobs=-1, random_state=42)
    best_rf.fit(X_train, y_train)
    rf_auroc = roc_auc_score(y_val, best_rf.predict_proba(X_val)[:, 1])
    print(f"📊 Holdout AUROC: {rf_auroc:.4f}")
    
    results["rf"] = {
        "cv_auroc": round(study_rf.best_value, 4),
        "holdout_auroc": round(rf_auroc, 4),
        "params": study_rf.best_params,
    }
    
    # Optimizar GBM
    print(f"\n{'='*50}")
    print("🔧 Optimizando GradientBoosting...")
    print("="*50)
    
    study_gbm = optuna.create_study(direction="maximize")
    study_gbm.optimize(lambda t: objective_gbm(t, X, y), n_trials=args.n_trials, show_progress_bar=True)
    
    print(f"✅ Mejor CV AUROC: {study_gbm.best_value:.4f}")
    print(f"📋 Params: {study_gbm.best_params}")
    
    best_gbm = GradientBoostingClassifier(**study_gbm.best_params, random_state=42)
    best_gbm.fit(X_train, y_train)
    gbm_auroc = roc_auc_score(y_val, best_gbm.predict_proba(X_val)[:, 1])
    print(f"📊 Holdout AUROC: {gbm_auroc:.4f}")
    
    results["gbm"] = {
        "cv_auroc": round(study_gbm.best_value, 4),
        "holdout_auroc": round(gbm_auroc, 4),
        "params": study_gbm.best_params,
    }
    
    # Resumen
    best = max(results.items(), key=lambda x: x[1]["holdout_auroc"])
    
    print(f"\n{'='*50}")
    print(f"🏆 MEJOR: {best[0].upper()} con AUROC {best[1]['holdout_auroc']:.4f}")
    print("="*50)
    
    # Guardar
    with open(output_dir / "optuna_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Guardado: {output_dir / 'optuna_results.json'}")


if __name__ == "__main__":
    main()
