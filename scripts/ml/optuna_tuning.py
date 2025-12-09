#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning para EM-Predictor.

Optimiza hiperparámetros de modelos usando Optuna con pruning.

Uso:
    python optuna_tuning.py --data-path data/processed/paciente1 --n-trials 50
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna no instalado. Ejecuta: pip install optuna")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def load_data(data_path: Path) -> pd.DataFrame:
    """Carga dataset."""
    clusters_path = data_path / "training_dataset_clusters.parquet"
    if clusters_path.exists():
        return pd.read_parquet(clusters_path)
    return pd.read_parquet(data_path / "training_dataset.parquet")


def prepare_data(df: pd.DataFrame, target: str = "relapse_in_14d") -> tuple:
    """Prepara X e y para entrenamiento."""
    exclude = [
        "date", "first_message", "last_message",
        "relapse_in_7d", "relapse_in_14d", "relapse_in_30d"
    ]
    
    features = [c for c in df.columns if c not in exclude 
                and df[c].dtype in ["float64", "int64", "float32", "int32"]]
    
    X = df[features].fillna(0)
    y = df[target]
    
    return X, y, features


def create_rf_objective(X: pd.DataFrame, y: pd.Series, cv: int = 5):
    """Crea función objetivo para RandomForest."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
            "n_jobs": -1,
            "random_state": 42,
        }
        
        model = RandomForestClassifier(**params)
        
        scorer = make_scorer(roc_auc_score, needs_proba=True)
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        try:
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scorer)
            return scores.mean()
        except:
            return 0.5
    
    return objective


def create_gbm_objective(X: pd.DataFrame, y: pd.Series, cv: int = 5):
    """Crea función objetivo para GradientBoosting."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": 42,
        }
        
        model = GradientBoostingClassifier(**params)
        
        scorer = make_scorer(roc_auc_score, needs_proba=True)
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        try:
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scorer)
            return scores.mean()
        except:
            return 0.5
    
    return objective


def create_logreg_objective(X: pd.DataFrame, y: pd.Series, cv: int = 5):
    """Crea función objetivo para LogisticRegression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            "C": trial.suggest_float("C", 0.001, 100, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": "saga",
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "max_iter": 1000,
            "random_state": 42,
        }
        
        model = LogisticRegression(**params)
        
        scorer = make_scorer(roc_auc_score, needs_proba=True)
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        try:
            scores = cross_val_score(model, X_scaled, y, cv=cv_splitter, scoring=scorer)
            return scores.mean()
        except:
            return 0.5
    
    return objective


def run_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "rf",
    n_trials: int = 50,
    cv: int = 5,
) -> tuple:
    """Ejecuta optimización con Optuna."""
    
    objectives = {
        "rf": create_rf_objective(X, y, cv),
        "gbm": create_gbm_objective(X, y, cv),
        "logreg": create_logreg_objective(X, y, cv),
    }
    
    if model_type not in objectives:
        raise ValueError(f"Modelo no soportado: {model_type}")
    
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    
    study.optimize(
        objectives[model_type],
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
    )
    
    return study.best_params, study.best_value, study


def train_with_best_params(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    best_params: Dict,
) -> tuple:
    """Entrena modelo con mejores parámetros y evalúa."""
    
    # Split temporal
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    
    if model_type == "rf":
        model = RandomForestClassifier(**best_params, n_jobs=-1, random_state=42)
    elif model_type == "gbm":
        model = GradientBoostingClassifier(**best_params, random_state=42)
    elif model_type == "logreg":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        model = LogisticRegression(**best_params, solver="saga", max_iter=1000, random_state=42)
    
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    auroc = roc_auc_score(y_val, y_proba)
    
    return model, auroc


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
    parser.add_argument("--data-path", "-d", type=Path, required=True)
    parser.add_argument("--target", "-t", default="relapse_in_14d")
    parser.add_argument("--n-trials", "-n", type=int, default=50)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--models", "-m", nargs="+", default=["rf", "gbm"],
                       choices=["rf", "gbm", "logreg"])
    parser.add_argument("--output", "-o", type=Path, default=None)
    
    args = parser.parse_args()
    
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna no disponible. Instala con: pip install optuna")
        return
    
    output_dir = args.output or args.data_path
    
    # Cargar datos
    df = load_data(args.data_path)
    X, y, features = prepare_data(df, args.target)
    
    print(f"📊 Dataset: {X.shape}")
    print(f"🎯 Target: {args.target} (positive rate: {y.mean():.2%})")
    print(f"⚙️ Trials por modelo: {args.n_trials}")
    
    results = {}
    best_overall = {"model": None, "auroc": 0, "params": {}}
    
    for model_type in args.models:
        print(f"\n{'='*60}")
        print(f"🔧 Optimizando {model_type.upper()}...")
        print("="*60)
        
        best_params, best_cv_score, study = run_optimization(
            X, y, model_type, args.n_trials, args.cv
        )
        
        print(f"\n✅ Mejor CV AUROC: {best_cv_score:.4f}")
        print(f"📋 Mejores parámetros: {best_params}")
        
        # Evaluar en holdout
        model, holdout_auroc = train_with_best_params(X, y, model_type, best_params)
        print(f"📊 Holdout AUROC: {holdout_auroc:.4f}")
        
        results[model_type] = {
            "cv_auroc": round(best_cv_score, 4),
            "holdout_auroc": round(holdout_auroc, 4),
            "best_params": best_params,
        }
        
        if holdout_auroc > best_overall["auroc"]:
            best_overall = {
                "model": model_type,
                "auroc": holdout_auroc,
                "params": best_params,
            }
    
    # Resumen
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE OPTIMIZACIÓN")
    print("="*60)
    
    for model_type, res in results.items():
        print(f"\n{model_type.upper()}:")
        print(f"   CV AUROC: {res['cv_auroc']}")
        print(f"   Holdout AUROC: {res['holdout_auroc']}")
    
    print(f"\n🏆 MEJOR MODELO: {best_overall['model'].upper()}")
    print(f"   AUROC: {best_overall['auroc']:.4f}")
    print(f"   Params: {best_overall['params']}")
    
    # Guardar resultados
    results_path = output_dir / "optuna_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "results": results,
            "best_model": best_overall,
            "config": {
                "n_trials": args.n_trials,
                "cv": args.cv,
                "target": args.target,
            }
        }, f, indent=2)
    
    print(f"\n💾 Resultados guardados: {results_path}")
    
    return results


if __name__ == "__main__":
    main()
