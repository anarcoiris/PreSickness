"""
Train Ensemble Models (Classical ML) for MS Relapse Prediction.
Generates ensemble_results.json and saves trained models.

Usage:
    python scripts/train_ensemble.py
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Settings (standalone, no external dependencies)
DB_DSN = os.getenv("DATABASE_URL", "postgresql://emuser:changeme@localhost/empredictor")


def flatten_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert DataFrame with JSONB-style features into flat numpy arrays.
    Returns (X, y) for the 14-day horizon target.
    """
    feature_cols = [
        'sentiment_mean', 'sentiment_std', 'sentiment_trend',
        'avg_sentence_len_mean', 'ttr_mean',
        'num_messages_total', 'num_messages_mean',
        'response_latency_mean',
        'steps_mean', 'sleep_hours_mean', 'hr_mean',
        'window_size_days'
    ]
    
    # Fill missing columns with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    X = df[feature_cols].fillna(0).values
    y = df['relapse_in_14d'].values if 'relapse_in_14d' in df.columns else np.zeros(len(df))
    
    return X, y


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    """Train all ensemble models."""
    models = {}
    
    # 1. Random Forest
    logger.info("Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['rf'] = rf
    
    # 2. Gradient Boosting
    logger.info("Training GradientBoosting...")
    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gbm.fit(X_train, y_train)
    models['gbm'] = gbm
    
    # 3. Logistic Regression
    logger.info("Training LogisticRegression...")
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    models['logreg'] = logreg
    
    # 4. Soft Voting Ensemble
    logger.info("Training VotingClassifier (soft)...")
    voting = VotingClassifier(
        estimators=[('rf', rf), ('gbm', gbm), ('logreg', logreg)],
        voting='soft'
    )
    voting.fit(X_train, y_train)
    models['voting_soft'] = voting
    
    # 5. Stacking
    logger.info("Training StackingClassifier...")
    stacking = StackingClassifier(
        estimators=[('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
                    ('gbm', GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42))],
        final_estimator=LogisticRegression(),
        cv=3
    )
    stacking.fit(X_train, y_train)
    models['stacking'] = stacking
    
    return models


def evaluate_models(models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate all models and return metrics."""
    results = {}
    
    for name, model in models.items():
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(y_test, y_prob)
            auprc = average_precision_score(y_test, y_prob)
            results[name] = {'auroc': round(auroc, 4), 'auprc': round(auprc, 4)}
            logger.info(f"{name}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")
            results[name] = {'auroc': 0.0, 'auprc': 0.0}
    
    # Manual average (simple ensemble)
    all_probs = []
    for name in ['rf', 'gbm', 'logreg']:
        if name in models:
            all_probs.append(models[name].predict_proba(X_test)[:, 1])
    
    if all_probs:
        avg_prob = np.mean(all_probs, axis=0)
        auroc = roc_auc_score(y_test, avg_prob)
        auprc = average_precision_score(y_test, avg_prob)
        results['manual_avg'] = {'auroc': round(auroc, 4), 'auprc': round(auprc, 4)}
        logger.info(f"manual_avg: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    
    return results


async def main():
    output_dir = Path("data/processed/paciente1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load existing features from CSV, otherwise generate synthetic
    features_csv = Path("data/processed/paciente1/features.csv")
    
    if features_csv.exists():
        logger.info(f"Loading features from {features_csv}")
        df = pd.read_csv(features_csv)
    else:
        logger.info("No features CSV found. Generating synthetic data for demo...")
        np.random.seed(42)
        n_samples = 500
        df = pd.DataFrame({
            'sentiment_mean': np.random.randn(n_samples) * 0.3,
            'sentiment_std': np.abs(np.random.randn(n_samples) * 0.1),
            'sentiment_trend': np.random.randn(n_samples) * 0.05,
            'avg_sentence_len_mean': np.random.randint(5, 25, n_samples),
            'ttr_mean': np.random.uniform(0.3, 0.8, n_samples),
            'num_messages_total': np.random.randint(10, 200, n_samples),
            'num_messages_mean': np.random.uniform(5, 50, n_samples),
            'response_latency_mean': np.random.uniform(60, 3600, n_samples),
            'steps_mean': np.random.randint(1000, 15000, n_samples),
            'sleep_hours_mean': np.random.uniform(4, 9, n_samples),
            'hr_mean': np.random.uniform(55, 90, n_samples),
            'window_size_days': np.random.choice([7, 14, 30], n_samples),
            'relapse_in_14d': np.random.binomial(1, 0.15, n_samples)
        })
    
    logger.info(f"Dataset shape: {df.shape}")
    
    X, y = flatten_features(df)
    logger.info(f"Features shape: {X.shape}, Positive rate: {y.mean():.3f}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 1 else None
    )
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    models = train_models(X_train, y_train)
    
    # Evaluate
    results = evaluate_models(models, X_test, y_test)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['auroc'])[0]
    
    # Save results
    output = {
        "results": results,
        "best": best_model,
        "trained_at": datetime.utcnow().isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }
    
    results_path = output_dir / "ensemble_results.json"
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Save models
    for name, model in models.items():
        model_path = output_dir / f"{name}_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")
    
    logger.info(f"Best model: {best_model} with AUROC={results[best_model]['auroc']}")


if __name__ == "__main__":
    asyncio.run(main())
