#!/usr/bin/env python3
"""
Cluster de Señales Temporales para EM-Predictor

Agrupa eventos detectados por proximidad temporal para identificar
períodos de alta actividad de síntomas que pueden indicar un brote inminente.

La idea: más señales cercanas en el tiempo => mayor probabilidad de brote.

Uso:
    python cluster_signals.py datos/paciente1_events_auto.csv --output datos/paciente1_clusters.csv
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np


@dataclass
class SignalCluster:
    """Representa un cluster de señales temporales."""
    start_date: datetime
    end_date: datetime
    peak_date: datetime
    total_signals: int
    unique_event_types: int
    event_types: Dict[str, int]
    max_severity: str
    severity_score: float
    density: float  # signals per day
    is_probable_relapse: bool


def load_events(path: Path) -> pd.DataFrame:
    """Carga eventos desde CSV."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def severity_to_score(severity: str) -> float:
    """Convierte severidad a score numérico."""
    mapping = {
        "severe": 3.0,
        "moderate": 2.0,
        "mild": 1.0,
    }
    return mapping.get(severity.lower(), 0.5)


def calculate_daily_signal_score(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula un score de señal diario basado en:
    - Número de eventos
    - Severidad de eventos
    - Diversidad de tipos de evento
    """
    daily_scores = []
    
    for date, group in events_df.groupby(events_df["date"].dt.date):
        event_types = group["event_type"].unique().tolist()
        severities = group["severity"].apply(severity_to_score)
        
        # Score = sum of severities * diversity bonus
        base_score = severities.sum()
        diversity_bonus = 1 + (len(event_types) - 1) * 0.2  # +20% por cada tipo adicional
        total_score = base_score * diversity_bonus
        
        # Penalizar eventos de contraparte (si está en notas)
        counterpart_ratio = group["notes"].str.contains("contraparte", case=False, na=False).mean()
        target_bonus = 1 + (1 - counterpart_ratio) * 0.5  # Más peso a mensajes del paciente
        
        final_score = total_score * target_bonus
        
        daily_scores.append({
            "date": pd.Timestamp(date),
            "event_count": len(group),
            "event_types": event_types,
            "unique_types": len(event_types),
            "max_severity": group["severity"].apply(severity_to_score).max(),
            "signal_score": final_score,
        })
    
    return pd.DataFrame(daily_scores)


def find_clusters(
    daily_scores: pd.DataFrame,
    gap_days: int = 3,
    min_cluster_score: float = 5.0,
) -> List[SignalCluster]:
    """
    Encuentra clusters de señales basándose en proximidad temporal.
    
    Args:
        daily_scores: DataFrame con scores diarios
        gap_days: Máximo de días entre señales para considerarlas del mismo cluster
        min_cluster_score: Score mínimo para considerar un cluster significativo
    
    Returns:
        Lista de SignalCluster
    """
    if daily_scores.empty:
        return []
    
    # Ordenar por fecha
    daily_scores = daily_scores.sort_values("date").reset_index(drop=True)
    
    clusters = []
    current_cluster_start = None
    current_cluster_data = []
    
    for i, row in daily_scores.iterrows():
        if current_cluster_start is None:
            # Iniciar nuevo cluster
            current_cluster_start = row["date"]
            current_cluster_data = [row]
        else:
            # Verificar si continúa el cluster
            last_date = current_cluster_data[-1]["date"]
            gap = (row["date"] - last_date).days
            
            if gap <= gap_days:
                # Continúa el cluster
                current_cluster_data.append(row)
            else:
                # Cerrar cluster actual y empezar uno nuevo
                cluster = _create_cluster(current_cluster_data, min_cluster_score)
                if cluster:
                    clusters.append(cluster)
                
                current_cluster_start = row["date"]
                current_cluster_data = [row]
    
    # Cerrar último cluster
    if current_cluster_data:
        cluster = _create_cluster(current_cluster_data, min_cluster_score)
        if cluster:
            clusters.append(cluster)
    
    return clusters


def _create_cluster(data: List[dict], min_score: float) -> SignalCluster:
    """Crea un SignalCluster a partir de datos diarios."""
    total_score = sum(d["signal_score"] for d in data)
    
    if total_score < min_score:
        return None
    
    dates = [d["date"] for d in data]
    start_date = min(dates)
    end_date = max(dates)
    days_span = (end_date - start_date).days + 1
    
    # Encontrar el día pico (más señales)
    peak_day = max(data, key=lambda x: x["signal_score"])
    
    # Agregar tipos de evento
    all_types = defaultdict(int)
    for d in data:
        for t in d["event_types"]:
            all_types[t] += 1
    
    # Calcular densidad
    density = sum(d["event_count"] for d in data) / days_span
    
    # Determinar si es probable brote
    # Criterios: alta densidad + múltiples tipos + alta severidad
    is_probable = (
        density >= 2.0 and 
        len(all_types) >= 3 and
        any(d["max_severity"] >= 2.0 for d in data)
    )
    
    return SignalCluster(
        start_date=start_date,
        end_date=end_date,
        peak_date=peak_day["date"],
        total_signals=sum(d["event_count"] for d in data),
        unique_event_types=len(all_types),
        event_types=dict(all_types),
        max_severity=["mild", "moderate", "severe"][int(max(d["max_severity"] for d in data)) - 1],
        severity_score=total_score,
        density=density,
        is_probable_relapse=is_probable,
    )


def generate_cluster_report(clusters: List[SignalCluster]) -> dict:
    """Genera reporte de clusters."""
    probable_relapses = [c for c in clusters if c.is_probable_relapse]
    
    return {
        "total_clusters": len(clusters),
        "probable_relapses": len(probable_relapses),
        "clusters": [
            {
                "start_date": c.start_date.strftime("%Y-%m-%d"),
                "end_date": c.end_date.strftime("%Y-%m-%d"),
                "peak_date": c.peak_date.strftime("%Y-%m-%d"),
                "duration_days": (c.end_date - c.start_date).days + 1,
                "total_signals": c.total_signals,
                "unique_event_types": c.unique_event_types,
                "event_types": c.event_types,
                "max_severity": c.max_severity,
                "severity_score": round(c.severity_score, 2),
                "density": round(c.density, 2),
                "is_probable_relapse": c.is_probable_relapse,
            }
            for c in sorted(clusters, key=lambda x: x.severity_score, reverse=True)
        ]
    }


def generate_labels_from_clusters(
    clusters: List[SignalCluster],
    date_range: Tuple[datetime, datetime],
    horizons: List[int] = [7, 14, 30],
) -> pd.DataFrame:
    """
    Genera labels de predicción basándose en clusters.
    
    Para cada fecha en el rango, indica si hay un cluster probable en los próximos N días.
    """
    start_date, end_date = date_range
    dates = pd.date_range(start_date, end_date, freq="D")
    
    # Obtener fechas de clusters probables
    relapse_dates = set()
    for c in clusters:
        if c.is_probable_relapse:
            relapse_dates.add(c.peak_date.date())
    
    labels = []
    for date in dates:
        row = {"date": date}
        
        for horizon in horizons:
            # Verificar si hay relapse en los próximos N días
            future_dates = [date.date() + timedelta(days=d) for d in range(1, horizon + 1)]
            has_relapse = any(d in relapse_dates for d in future_dates)
            row[f"relapse_cluster_in_{horizon}d"] = int(has_relapse)
        
        labels.append(row)
    
    return pd.DataFrame(labels)


def print_summary(clusters: List[SignalCluster]):
    """Imprime resumen de clusters."""
    print("\n" + "="*70)
    print("📊 ANÁLISIS DE CLUSTERS DE SEÑALES")
    print("="*70)
    
    probable = [c for c in clusters if c.is_probable_relapse]
    
    print(f"\n📈 Total clusters detectados: {len(clusters)}")
    print(f"🔴 Probables brotes: {len(probable)}")
    
    if probable:
        print("\n🚨 PERÍODOS DE PROBABLE BROTE:")
        for i, c in enumerate(sorted(probable, key=lambda x: x.peak_date), 1):
            print(f"\n   #{i} {c.start_date.strftime('%Y-%m-%d')} → {c.end_date.strftime('%Y-%m-%d')}")
            print(f"       Pico: {c.peak_date.strftime('%Y-%m-%d')}")
            print(f"       Señales: {c.total_signals} | Densidad: {c.density:.1f}/día")
            print(f"       Severidad máx: {c.max_severity} | Score: {c.severity_score:.1f}")
            print(f"       Tipos: {', '.join(c.event_types.keys())}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Agrupa señales por proximidad temporal para predecir brotes"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Archivo CSV de eventos (output de extract_events.py)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Archivo de salida para clusters (CSV)"
    )
    parser.add_argument(
        "--gap-days", "-g",
        type=int,
        default=3,
        help="Máximo días entre señales para considerarlas del mismo cluster (default: 3)"
    )
    parser.add_argument(
        "--min-score", "-m",
        type=float,
        default=5.0,
        help="Score mínimo para considerar un cluster significativo (default: 5.0)"
    )
    parser.add_argument(
        "--json-report",
        action="store_true",
        help="Generar reporte JSON detallado"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Archivo no encontrado: {args.input}")
        return 1
    
    # Cargar eventos
    events_df = load_events(args.input)
    print(f"📄 Cargados {len(events_df)} eventos")
    
    # Calcular scores diarios
    daily_scores = calculate_daily_signal_score(events_df)
    print(f"📅 {len(daily_scores)} días con actividad")
    
    # Encontrar clusters
    clusters = find_clusters(
        daily_scores,
        gap_days=args.gap_days,
        min_cluster_score=args.min_score,
    )
    
    # Mostrar resumen
    print_summary(clusters)
    
    # Guardar outputs
    if args.output:
        output_path = args.output
    else:
        output_path = args.input.with_name(args.input.stem + "_clusters.csv")
    
    # Guardar clusters como CSV
    cluster_data = []
    for c in clusters:
        cluster_data.append({
            "start_date": c.start_date.strftime("%Y-%m-%d"),
            "end_date": c.end_date.strftime("%Y-%m-%d"),
            "peak_date": c.peak_date.strftime("%Y-%m-%d"),
            "total_signals": c.total_signals,
            "unique_types": c.unique_event_types,
            "max_severity": c.max_severity,
            "severity_score": c.severity_score,
            "density": c.density,
            "is_probable_relapse": c.is_probable_relapse,
        })
    
    pd.DataFrame(cluster_data).to_csv(output_path, index=False)
    print(f"✅ Clusters guardados: {output_path}")
    
    if args.json_report:
        report = generate_cluster_report(clusters)
        report_path = output_path.with_suffix(".json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✅ Reporte JSON: {report_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
