#!/usr/bin/env python3
"""
Time-Series Cross-Validation con Purging y Embargo.

Implementa validación temporal correcta para evitar data leakage.

- Purging: Elimina últimos N días del train (sus labels miran hacia test)
- Embargo: Gap entre fin de train y inicio de test

Uso:
    from time_series_cv import PurgedTimeSeriesSplit
    
    cv = PurgedTimeSeriesSplit(n_splits=5, gap=14, purge=14)
    for train_idx, test_idx in cv.split(X, dates=dates):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, Optional


class PurgedTimeSeriesSplit:
    """
    Time-series split con purging y embargo.
    
    Evita data leakage temporal asegurando:
    1. Purge: Elimina últimos N días del train (sus labels dependen del test)
    2. Gap: Período de embargo entre train y test
    
    Args:
        n_splits: Número de folds
        gap: Días de separación entre train y test (embargo)
        purge: Días a eliminar del final de train (purging)
        test_size: Días de test por fold (None = automático)
    """
    
    def __init__(
        self, 
        n_splits: int = 5, 
        gap: int = 14, 
        purge: int = 14,
        test_size: Optional[int] = None
    ):
        self.n_splits = n_splits
        self.gap = gap
        self.purge = purge
        self.test_size = test_size
    
    def split(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None, 
        dates: Optional[pd.Series] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Genera índices de train/test para cada fold.
        
        Args:
            X: Features DataFrame
            y: Target (opcional, no usado)
            dates: Serie de fechas correspondientes a X
            
        Yields:
            (train_indices, test_indices) para cada fold
        """
        n_samples = len(X)
        
        if dates is not None:
            # Ordenar por fechas
            dates = pd.to_datetime(dates)
            sort_idx = dates.argsort()
            dates_sorted = dates.iloc[sort_idx].reset_index(drop=True)
        else:
            sort_idx = np.arange(n_samples)
            dates_sorted = None
        
        # Calcular tamaño de test si no está especificado
        test_size = self.test_size
        if test_size is None:
            # Aproximadamente 20% del dataset para test total
            test_size = max(n_samples // (self.n_splits * 2), 14)
        
        # Calcular punto de inicio del primer test
        # Dejar suficiente espacio para train
        min_train_size = n_samples // 3
        
        for fold in range(self.n_splits):
            # Calcular índices
            test_end = n_samples - fold * test_size
            test_start = test_end - test_size
            
            if test_start < min_train_size:
                break
            
            # Train va desde inicio hasta test_start - gap - purge
            train_end = test_start - self.gap
            train_end_purged = train_end - self.purge
            
            if train_end_purged <= 0:
                continue
            
            train_idx = sort_idx[:train_end_purged]
            test_idx = sort_idx[test_start:test_end]
            
            if len(train_idx) < 10 or len(test_idx) < 5:
                continue
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def check_temporal_leakage(
    train_dates: pd.Series,
    test_dates: pd.Series,
    horizon: int = 14
) -> dict:
    """
    Verifica que no hay overlap temporal entre train y test.
    
    Args:
        train_dates: Fechas del set de entrenamiento
        test_dates: Fechas del set de test
        horizon: Horizonte de predicción en días
        
    Returns:
        Dict con resultados de la verificación
    """
    train_dates = pd.to_datetime(train_dates)
    test_dates = pd.to_datetime(test_dates)
    
    train_max = train_dates.max()
    test_min = test_dates.min()
    
    gap_days = (test_min - train_max).days
    
    results = {
        "train_max": train_max,
        "test_min": test_min,
        "gap_days": gap_days,
        "required_gap": horizon,
        "has_leakage": gap_days < horizon,
        "overlap_dates": [],
    }
    
    # Buscar overlap directo
    overlap = set(train_dates.dt.date) & set(test_dates.dt.date)
    if overlap:
        results["overlap_dates"] = sorted(list(overlap))
        results["has_leakage"] = True
    
    return results


def validate_cv_splits(
    X: pd.DataFrame,
    dates: pd.Series,
    cv,
    horizon: int = 14,
    verbose: bool = True
) -> bool:
    """
    Valida que los splits de CV no tienen data leakage.
    
    Args:
        X: Features DataFrame
        dates: Serie de fechas
        cv: Cross-validator (e.g., PurgedTimeSeriesSplit)
        horizon: Horizonte de predicción
        verbose: Imprimir detalles
        
    Returns:
        True si pasa la validación
    """
    all_passed = True
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, dates=dates)):
        train_dates = dates.iloc[train_idx]
        test_dates = dates.iloc[test_idx]
        
        check = check_temporal_leakage(train_dates, test_dates, horizon)
        
        if check["has_leakage"]:
            all_passed = False
            if verbose:
                print(f"❌ Fold {fold}: LEAKAGE DETECTADO")
                print(f"   Gap: {check['gap_days']} días (requiere {horizon})")
                if check["overlap_dates"]:
                    print(f"   Overlap: {check['overlap_dates'][:5]}...")
        else:
            if verbose:
                print(f"✅ Fold {fold}: OK (gap={check['gap_days']} días)")
    
    return all_passed


if __name__ == "__main__":
    # Test básico
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--gap", type=int, default=14)
    parser.add_argument("--purge", type=int, default=14)
    args = parser.parse_args()
    
    # Crear datos de prueba
    dates = pd.date_range("2024-01-01", periods=args.n_samples, freq="D")
    X = pd.DataFrame({"feature": np.random.randn(args.n_samples)})
    
    print(f"📊 Dataset: {len(X)} samples")
    print(f"📅 Rango: {dates.min()} - {dates.max()}")
    print(f"\n🔄 PurgedTimeSeriesSplit(n_splits={args.n_splits}, gap={args.gap}, purge={args.purge})\n")
    
    cv = PurgedTimeSeriesSplit(
        n_splits=args.n_splits,
        gap=args.gap,
        purge=args.purge
    )
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, dates=pd.Series(dates))):
        train_dates = dates[train_idx]
        test_dates = dates[test_idx]
        
        print(f"Fold {fold}:")
        print(f"  Train: {len(train_idx)} samples ({train_dates.min().date()} - {train_dates.max().date()})")
        print(f"  Test:  {len(test_idx)} samples ({test_dates.min().date()} - {test_dates.max().date()})")
        print(f"  Gap:   {(test_dates.min() - train_dates.max()).days} días")
        print()
    
    # Validar
    print("\n" + "="*50)
    print("🔍 Validación de leakage:")
    validate_cv_splits(X, pd.Series(dates), cv, horizon=14)
