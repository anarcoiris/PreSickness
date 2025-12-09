#!/usr/bin/env python3
"""
Ensemble Model Training para EM-Predictor.

Combina RF + GBM + LogReg usando stacking o voting.

Uso:
    python ensemble_model.py --data-path data/processed/paciente1
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.calibration import CalibratedClassifierCV


def load_data(data_path: Path, target: str = "relapse_in_14d"):
    """Carga dataset con features engineered si está disponible."""
    # Prioridad: engineered > clusters > original
    paths = [
        data_path / "training_dataset_engineered.parquet",
        data_path / "training_dataset_clusters.parquet",
        data_path / "training_dataset.parquet",
    ]
    
    for path in paths:
        if path.exists():
            df = pd.read_parquet(path)
            print(f"📁 Usando: {path.name}")
            break
    else:
        raise FileNotFoundError(f"No se encontró dataset en {data_path}")
    
    exclude = ["date", "first_message", "last_message",
               "relapse_in_7d", "relapse_in_14d", "relapse_in_30d"]
    
    features = [c for c in df.columns if c not in exclude 
                and df[c].dtype in ["float64", "int64", "float32", "int32"]]
    
    X = df[features].fillna(0)
    y = df[target]
    
    return X, y, features


def create_base_models(best_params: Dict = None) -> Dict:
    """Crea modelos base con mejores parámetros."""
    best_params = best_params or {}
    
    rf_params = best_params.get("rf", {
        "n_estimators": 135,
        "max_depth": 13,
        "min_samples_split": 7,
        "min_samples_leaf": 2,
    })
    
    gbm_params = best_params.get("gbm", {
        "n_estimators": 97,
        "max_depth": 8,
        "learning_rate": 0.022,
        "min_samples_split": 5,
        "subsample": 0.77,
    })
    
    models = {
        "rf": RandomForestClassifier(
            **rf_params,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        "gbm": GradientBoostingClassifier(
            **gbm_params,
            random_state=42
        ),
        "logreg": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
    }
    
    return models


def create_voting_ensemble(models: Dict, voting: str = "soft") -> VotingClassifier:
    """Crea ensemble de voting."""
    estimators = [(name, model) for name, model in models.items()]
    
    return VotingClassifier(
        estimators=estimators,
        voting=voting,
        n_jobs=-1
    )


def create_stacking_ensemble(models: Dict) -> StackingClassifier:
    """Crea ensemble de stacking con meta-learner."""
    estimators = [(name, model) for name, model in models.items() if name != "logreg"]
    
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
        cv=5,
        n_jobs=-1,
        passthrough=True  # También pasar features originales al meta-learner
    )


def evaluate_model(model, X_train, y_train, X_val, y_val, name: str) -> Dict:
    """Entrena y evalúa un modelo."""
    # Escalar para LogReg si está en el pipeline
    if "logreg" in name.lower():
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model.fit(X_train_scaled, y_train)
        y_proba = model.predict_proba(X_val_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
    
    y_pred = (y_proba >= 0.5).astype(int)
    
    auroc = roc_auc_score(y_val, y_proba)
    auprc = average_precision_score(y_val, y_proba)
    
    return {
        "name": name,
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "model": model,
        "y_proba": y_proba,
    }


def main():
    parser = argparse.ArgumentParser(description="Ensemble Model Training")
    parser.add_argument("--data-path", "-d", type=Path, required=True)
    parser.add_argument("--target", "-t", default="relapse_in_14d")
    parser.add_argument("--output", "-o", type=Path, default=None)
    
    args = parser.parse_args()
    output_dir = args.output or args.data_path
    
    # Cargar datos
    X, y, features = load_data(args.data_path, args.target)
    print(f"📊 Features: {len(features)}, Samples: {len(X)}")
    print(f"🎯 Positive rate: {y.mean():.1%}")
    
    # Split temporal
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    print(f"📈 Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Cargar mejores params de Optuna si existen
    optuna_path = args.data_path / "optuna_results.json"
    if optuna_path.exists():
        with open(optuna_path) as f:
            optuna_results = json.load(f)
        best_params = {
            "rf": optuna_results.get("rf", {}).get("params", {}),
            "gbm": optuna_results.get("gbm", {}).get("params", {}),
        }
        print(f"✅ Cargados params de Optuna")
    else:
        best_params = {}
    
    # Crear modelos base
    base_models = create_base_models(best_params)
    
    results = {}
    
    # Evaluar modelos base
    print(f"\n{'='*60}")
    print("MODELOS BASE")
    print("="*60)
    
    for name, model in base_models.items():
        result = evaluate_model(model, X_train, y_train, X_val, y_val, name)
        results[name] = result
        print(f"{name.upper()}: AUROC={result['auroc']:.4f}, AUPRC={result['auprc']:.4f}")
    
    # Voting Ensemble
    print(f"\n{'='*60}")
    print("ENSEMBLES")
    print("="*60)
    
    # Soft voting
    voting_soft = create_voting_ensemble(create_base_models(best_params), voting="soft")
    result = evaluate_model(voting_soft, X_train, y_train, X_val, y_val, "voting_soft")
    results["voting_soft"] = result
    print(f"Voting (Soft): AUROC={result['auroc']:.4f}, AUPRC={result['auprc']:.4f}")
    
    # Stacking
    try:
        stacking = create_stacking_ensemble(create_base_models(best_params))
        result = evaluate_model(stacking, X_train, y_train, X_val, y_val, "stacking")
        results["stacking"] = result
        print(f"Stacking: AUROC={result['auroc']:.4f}, AUPRC={result['auprc']:.4f}")
    except Exception as e:
        print(f"⚠️ Stacking falló: {e}")
    
    # Manual averaging (más robusto)
    print(f"\n{'='*60}")
    print("AVERAGING MANUAL")
    print("="*60)
    
    # Promedio simple de probabilidades
    probas = []
    for name in ["rf", "gbm"]:
        if name in results:
            probas.append(results[name]["y_proba"])
    
    if probas:
        avg_proba = np.mean(probas, axis=0)
        auroc_avg = roc_auc_score(y_val, avg_proba)
        auprc_avg = average_precision_score(y_val, avg_proba)
        results["manual_avg"] = {"auroc": round(auroc_avg, 4), "auprc": round(auprc_avg, 4)}
        print(f"Manual Average (RF+GBM): AUROC={auroc_avg:.4f}, AUPRC={auprc_avg:.4f}")
    
    # Mejor resultado
    best_name = max(results.keys(), key=lambda k: results[k].get("auroc", 0))
    best_auroc = results[best_name].get("auroc", 0)
    
    print(f"\n{'='*60}")
    print(f"🏆 MEJOR: {best_name.upper()} con AUROC {best_auroc:.4f}")
    print("="*60)
    
    # Guardar resultados
    results_clean = {k: {"auroc": v["auroc"], "auprc": v["auprc"]} 
                    for k, v in results.items() if "auroc" in v}
    
    with open(output_dir / "ensemble_results.json", "w") as f:
        json.dump({"results": results_clean, "best": best_name}, f, indent=2)
    
    print(f"\n💾 Resultados: {output_dir / 'ensemble_results.json'}")
    
    return results


if __name__ == "__main__":
    main()
