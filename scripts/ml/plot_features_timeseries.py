#!/usr/bin/env python3
"""
Plot de Features en Serie Temporal para EM-Predictor.

Visualiza evolución de features clave junto con eventos de brote.

Uso:
    python plot_features_timeseries.py --data-path data/processed/paciente1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


def load_data(data_path: Path) -> tuple:
    """Carga features diarios y clusters."""
    daily_df = pd.read_parquet(data_path / "daily_features.parquet")
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    
    # Cargar clusters si existen
    clusters_path = data_path.parent.parent / "datos" / "paciente1_events_auto_clusters.csv"
    if clusters_path.exists():
        clusters_df = pd.read_csv(clusters_path)
        clusters_df["start_date"] = pd.to_datetime(clusters_df["start_date"])
        clusters_df["end_date"] = pd.to_datetime(clusters_df["end_date"])
        clusters_df["peak_date"] = pd.to_datetime(clusters_df["peak_date"])
    else:
        clusters_df = pd.DataFrame()
    
    return daily_df, clusters_df


def plot_feature_timeseries(
    df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    features: list,
    output_path: Path,
    title: str = "Feature Evolution"
):
    """
    Plotea múltiples features en serie temporal con marcadores de brote.
    """
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features), sharex=True)
    
    if n_features == 1:
        axes = [axes]
    
    dates = df["date"]
    
    # Colores para features
    colors = plt.cm.tab10(np.linspace(0, 1, n_features))
    
    for i, (feature, color) in enumerate(zip(features, colors)):
        ax = axes[i]
        
        if feature not in df.columns:
            ax.text(0.5, 0.5, f"Feature '{feature}' no encontrado", 
                   transform=ax.transAxes, ha="center")
            continue
        
        values = df[feature].values
        
        # Plot principal
        ax.plot(dates, values, color=color, linewidth=1.5, alpha=0.8)
        ax.fill_between(dates, 0, values, color=color, alpha=0.2)
        
        # Media móvil
        window = min(7, len(values) // 3)
        if window > 1:
            rolling_mean = pd.Series(values).rolling(window=window, center=True).mean()
            ax.plot(dates, rolling_mean, color="black", linewidth=2, 
                   linestyle="--", alpha=0.7, label=f"Media móvil ({window}d)")
        
        # Marcar clusters de brote
        if not clusters_df.empty:
            for _, cluster in clusters_df.iterrows():
                if cluster["is_probable_relapse"]:
                    start = cluster["start_date"]
                    end = cluster["end_date"]
                    peak = cluster["peak_date"]
                    
                    # Sombrear período de brote
                    y_min, y_max = ax.get_ylim()
                    rect = Rectangle(
                        (mdates.date2num(start), y_min),
                        mdates.date2num(end) - mdates.date2num(start),
                        y_max - y_min,
                        facecolor="red", alpha=0.15, edgecolor="red", linewidth=1
                    )
                    ax.add_patch(rect)
                    
                    # Marcar pico
                    ax.axvline(x=peak, color="red", linestyle=":", alpha=0.7, linewidth=1.5)
        
        ax.set_ylabel(feature.replace("_", " ").title(), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    
    # Formato de fechas
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlabel("Fecha")
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"📊 Plot guardado: {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, features: list, output_path: Path):
    """Genera heatmap de correlaciones entre features."""
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 2:
        print("⚠️ No hay suficientes features para heatmap")
        return
    
    corr_matrix = df[available_features].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    
    # Labels
    ax.set_xticks(range(len(available_features)))
    ax.set_yticks(range(len(available_features)))
    ax.set_xticklabels([f.replace("_", "\n") for f in available_features], fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels([f.replace("_", "\n") for f in available_features], fontsize=8)
    
    # Añadir valores
    for i in range(len(available_features)):
        for j in range(len(available_features)):
            val = corr_matrix.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)
    
    plt.colorbar(im, label="Correlación")
    plt.title("Correlación entre Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"🔥 Heatmap guardado: {output_path}")


def plot_feature_distributions(df: pd.DataFrame, features: list, output_path: Path):
    """Genera histogramas de distribución de features."""
    available_features = [f for f in features if f in df.columns]
    n_features = len(available_features)
    
    if n_features == 0:
        return
    
    cols = min(4, n_features)
    rows = (n_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(available_features):
        ax = axes[i]
        values = df[feature].dropna()
        
        ax.hist(values, bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(x=values.mean(), color="red", linestyle="--", label=f"Mean: {values.mean():.2f}")
        ax.axvline(x=values.median(), color="green", linestyle=":", label=f"Median: {values.median():.2f}")
        
        ax.set_title(feature.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # Ocultar axes vacíos
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Distribución de Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"📊 Distribuciones guardadas: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Features Timeseries")
    parser.add_argument("--data-path", "-d", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=None)
    
    args = parser.parse_args()
    output_dir = args.output or args.data_path
    
    # Cargar datos
    daily_df, clusters_df = load_data(args.data_path)
    print(f"📅 Días: {len(daily_df)}")
    print(f"🔴 Clusters de brote: {len(clusters_df[clusters_df['is_probable_relapse'] == True]) if not clusters_df.empty else 0}")
    
    # Features principales para plotear
    main_features = [
        "messages_count",
        "sentiment_proxy_mean",
        "word_count_mean",
        "type_token_ratio_mean",
        "night_ratio",
    ]
    
    # Verificar cuáles existen
    available = [f for f in main_features if f in daily_df.columns]
    print(f"Features disponibles: {available}")
    
    if not available:
        print("Usando primeras 5 columnas numéricas...")
        numeric_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()
        available = [c for c in numeric_cols if c != "date"][:5]
    
    # Plot timeseries
    plot_feature_timeseries(
        daily_df, clusters_df, available,
        output_dir / "features_timeseries.png",
        title="Evolución de Features del Paciente"
    )
    
    # Correlation heatmap
    all_numeric = daily_df.select_dtypes(include=[np.number]).columns.tolist()
    plot_correlation_heatmap(daily_df, all_numeric[:15], output_dir / "features_correlation.png")
    
    # Distributions
    plot_feature_distributions(daily_df, all_numeric[:12], output_dir / "features_distribution.png")
    
    print("\n✅ Plots generados!")


if __name__ == "__main__":
    main()
