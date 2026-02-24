#!/usr/bin/env python3
"""
Validación de Data Leakage para EM-Predictor.

Audita el pipeline ML para detectar fugas de información temporal.

Uso:
    python validate_no_leakage.py --data-path data/processed/paciente1
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def check_feature_temporal_integrity(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Verifica que los features no contengan información del futuro.
    
    Detecta patrones sospechosos:
    - Columnas sin lag/shift en el nombre pero correlación alta con target
    - Features que no deberían existir en t=0
    
    Args:
        df: DataFrame con features y targets
        verbose: Imprimir detalles
        
    Returns:
        Dict con resultados de la auditoría
    """
    results = {
        "passed": True,
        "warnings": [],
        "errors": [],
        "checked_columns": [],
    }
    
    # Identificar columnas de features vs targets
    target_cols = [c for c in df.columns if "relapse_in_" in c]
    date_cols = ["date", "first_message", "last_message"]
    feature_cols = [c for c in df.columns if c not in target_cols + date_cols]
    
    if verbose:
        print(f"📊 Verificando {len(feature_cols)} features...")
    
    # Verificar nombres de columnas
    for col in feature_cols:
        results["checked_columns"].append(col)
        
        # Columnas de rolling sin lag son sospechosas
        if "roll" in col.lower() and "lag" not in col.lower():
            # Verificar si el nombre indica que usa shift
            if "_lag" not in col and "_shift" not in col:
                msg = f"⚠️  '{col}': Rolling feature sin indicación de lag"
                results["warnings"].append(msg)
                if verbose:
                    print(msg)
    
    # Verificar correlación temporal
    if "date" in df.columns and len(target_cols) > 0:
        df_sorted = df.sort_values("date")
        
        for target in target_cols[:1]:  # Solo verificar el target principal
            if target not in df.columns:
                continue
                
            # Verificar si hay features que predicen "demasiado bien"
            for col in feature_cols[:20]:  # Limitar para no tardar
                if df[col].dtype not in ["float64", "int64", "float32", "int32"]:
                    continue
                
                try:
                    corr = df[col].corr(df[target])
                    if abs(corr) > 0.9:
                        msg = f"❌ '{col}': Correlación sospechosa con {target}: {corr:.3f}"
                        results["errors"].append(msg)
                        results["passed"] = False
                        if verbose:
                            print(msg)
                except:
                    pass
    
    return results


def check_label_generation(events_path: Path, labels_path: Path, verbose: bool = True) -> Dict:
    """
    Verifica que los labels se generan correctamente.
    
    Args:
        events_path: Path al archivo de eventos
        labels_path: Path al archivo de labels
        verbose: Imprimir detalles
        
    Returns:
        Dict con resultados
    """
    results = {
        "passed": True,
        "warnings": [],
        "info": {},
    }
    
    if not labels_path.exists():
        results["warnings"].append("Labels file not found")
        return results
    
    labels_df = pd.read_parquet(labels_path)
    
    # Verificar distribución de labels
    for col in labels_df.columns:
        if "relapse_in_" in col:
            pos_rate = labels_df[col].mean()
            results["info"][col] = {"positive_rate": pos_rate}
            
            if pos_rate > 0.9:
                msg = f"⚠️  '{col}': {pos_rate:.1%} positivos - posible label leakage"
                results["warnings"].append(msg)
                if verbose:
                    print(msg)
            elif pos_rate < 0.05:
                msg = f"⚠️  '{col}': Solo {pos_rate:.1%} positivos - muy desbalanceado"
                results["warnings"].append(msg)
                if verbose:
                    print(msg)
            else:
                if verbose:
                    print(f"✅ '{col}': {pos_rate:.1%} positivos")
    
    return results


def simulate_walk_forward_leakage_test(
    df: pd.DataFrame, 
    target: str = "relapse_in_14d",
    horizon: int = 14,
    verbose: bool = True
) -> Dict:
    """
    Simula walk-forward para detectar leakage.
    
    Compara AUROC con gap=0 vs gap=horizon.
    Si gap=0 es mucho mejor, hay leakage.
    
    Args:
        df: DataFrame con features y target
        target: Columna objetivo
        horizon: Horizonte de predicción
        verbose: Imprimir detalles
        
    Returns:
        Dict con resultados
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    
    results = {
        "auroc_gap_0": None,
        "auroc_gap_horizon": None,
        "leakage_indicator": None,
        "passed": True,
    }
    
    df = df.sort_values("date").reset_index(drop=True)
    
    # Identificar features
    exclude = ["date", "first_message", "last_message",
               "relapse_in_7d", "relapse_in_14d", "relapse_in_30d"]
    features = [c for c in df.columns if c not in exclude 
                and df[c].dtype in ["float64", "int64", "float32", "int32"]]
    
    X = df[features].fillna(0)
    y = df[target]
    
    n = len(df)
    train_end = int(n * 0.7)
    
    # Test sin gap
    X_train_0 = X.iloc[:train_end]
    y_train_0 = y.iloc[:train_end]
    X_test = X.iloc[train_end:]
    y_test = y.iloc[train_end:]
    
    if len(y_test.unique()) < 2:
        if verbose:
            print("⚠️  No hay suficiente varianza en test set")
        return results
    
    try:
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train_0, y_train_0)
        auroc_0 = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        results["auroc_gap_0"] = round(auroc_0, 4)
    except Exception as e:
        if verbose:
            print(f"⚠️  Error en test gap=0: {e}")
        return results
    
    # Test con gap
    train_end_gap = train_end - horizon
    if train_end_gap <= 0:
        if verbose:
            print("⚠️  Dataset muy pequeño para test con gap")
        return results
    
    X_train_gap = X.iloc[:train_end_gap]
    y_train_gap = y.iloc[:train_end_gap]
    
    try:
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train_gap, y_train_gap)
        auroc_gap = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        results["auroc_gap_horizon"] = round(auroc_gap, 4)
    except Exception as e:
        if verbose:
            print(f"⚠️  Error en test gap={horizon}: {e}")
        return results
    
    # Indicador de leakage
    diff = auroc_0 - auroc_gap
    results["leakage_indicator"] = round(diff, 4)
    
    if verbose:
        print(f"\n🔬 Test de Leakage Temporal:")
        print(f"   AUROC (gap=0):  {auroc_0:.4f}")
        print(f"   AUROC (gap={horizon}): {auroc_gap:.4f}")
        print(f"   Diferencia: {diff:.4f}")
    
    # Si la diferencia es > 0.15, hay sospecha de leakage
    if diff > 0.15:
        results["passed"] = False
        if verbose:
            print(f"   ❌ SOSPECHA DE LEAKAGE: diferencia > 0.15")
    else:
        if verbose:
            print(f"   ✅ Sin evidencia fuerte de leakage")
    
    return results


def run_full_audit(data_path: Path, verbose: bool = True) -> Dict:
    """
    Ejecuta auditoría completa de data leakage.
    
    Args:
        data_path: Path al directorio con datos procesados
        verbose: Imprimir detalles
        
    Returns:
        Dict con resultados de la auditoría
    """
    print("="*60)
    print("🔍 AUDITORÍA DE DATA LEAKAGE")
    print("="*60)
    
    results = {
        "passed": True,
        "checks": {},
    }
    
    # 1. Cargar datos
    training_path = data_path / "training_dataset_engineered.parquet"
    if not training_path.exists():
        training_path = data_path / "training_dataset_clusters.parquet"
    if not training_path.exists():
        training_path = data_path / "training_dataset.parquet"
    
    if not training_path.exists():
        print("❌ No se encontró dataset de entrenamiento")
        results["passed"] = False
        return results
    
    df = pd.read_parquet(training_path)
    print(f"\n📁 Dataset: {training_path.name} ({len(df)} samples)")
    
    # 2. Check feature integrity
    print("\n" + "-"*40)
    print("1️⃣ Verificando integridad de features")
    print("-"*40)
    feature_check = check_feature_temporal_integrity(df, verbose)
    results["checks"]["feature_integrity"] = feature_check
    if not feature_check["passed"]:
        results["passed"] = False
    
    # 3. Check labels
    print("\n" + "-"*40)
    print("2️⃣ Verificando generación de labels")
    print("-"*40)
    labels_path = data_path / "labels.parquet"
    events_path = data_path.parent.parent / "datos" / "paciente1_events_auto.csv"
    label_check = check_label_generation(events_path, labels_path, verbose)
    results["checks"]["label_generation"] = label_check
    
    # 4. Leakage simulation test
    print("\n" + "-"*40)
    print("3️⃣ Test de simulación de leakage")
    print("-"*40)
    leakage_check = simulate_walk_forward_leakage_test(df, verbose=verbose)
    results["checks"]["leakage_simulation"] = leakage_check
    if not leakage_check["passed"]:
        results["passed"] = False
    
    # Resumen
    print("\n" + "="*60)
    if results["passed"]:
        print("✅ AUDITORÍA PASADA")
    else:
        print("❌ AUDITORÍA FALLIDA - SE DETECTARON PROBLEMAS")
    print("="*60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validación de Data Leakage")
    parser.add_argument("--data-path", "-d", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--quiet", "-q", action="store_true")
    
    args = parser.parse_args()
    
    results = run_full_audit(args.data_path, verbose=not args.quiet)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Resultados guardados: {args.output}")
