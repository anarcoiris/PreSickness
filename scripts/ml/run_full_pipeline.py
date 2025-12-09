#!/usr/bin/env python3
"""
Pipeline Completo de Mejora de Modelo.

Ejecuta todos los pasos de mejora en orden:
1. Feature Engineering (lags, rolling, interactions)
2. Embeddings (Sentence Transformers)
3. Ensemble (RF + GBM + Voting/Stacking)

Uso:
    python run_full_pipeline.py --data-path data/processed/paciente1
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_script(script_path: str, args: list):
    """Ejecuta un script Python con argumentos."""
    cmd = [sys.executable, script_path] + args
    print(f"\n🔧 Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run Full Improvement Pipeline")
    parser.add_argument("--data-path", "-d", type=Path, required=True)
    parser.add_argument("--skip-embeddings", action="store_true", 
                       help="Skip embedding generation (slow)")
    parser.add_argument("--embedding-preset", default="fast",
                       choices=["fast", "balanced", "quality"])
    
    args = parser.parse_args()
    
    scripts_dir = Path(__file__).parent
    
    print("="*70)
    print("🚀 PIPELINE COMPLETO DE MEJORA")
    print("="*70)
    
    results = {}
    
    # 1. Feature Engineering
    print("\n" + "="*70)
    print("📊 PASO 1: Feature Engineering")
    print("="*70)
    
    success = run_script(
        str(scripts_dir / "feature_engineering.py"),
        ["--data-path", str(args.data_path)]
    )
    results["feature_engineering"] = "✅ OK" if success else "❌ Error"
    
    # 2. Embeddings (opcional)
    if not args.skip_embeddings:
        print("\n" + "="*70)
        print("🔤 PASO 2: Embeddings")
        print("="*70)
        
        success = run_script(
            str(scripts_dir / "add_embeddings.py"),
            ["--data-path", str(args.data_path), "--preset", args.embedding_preset]
        )
        results["embeddings"] = "✅ OK" if success else "❌ Error"
    else:
        print("\n⏭️ Saltando embeddings (--skip-embeddings)")
        results["embeddings"] = "⏭️ Skipped"
    
    # 3. Ensemble Model
    print("\n" + "="*70)
    print("🤖 PASO 3: Ensemble Model")
    print("="*70)
    
    success = run_script(
        str(scripts_dir / "ensemble_model.py"),
        ["--data-path", str(args.data_path)]
    )
    results["ensemble"] = "✅ OK" if success else "❌ Error"
    
    # Resumen
    print("\n" + "="*70)
    print("📋 RESUMEN DEL PIPELINE")
    print("="*70)
    
    for step, status in results.items():
        print(f"  {step}: {status}")
    
    # Cargar resultados finales
    ensemble_path = args.data_path / "ensemble_results.json"
    if ensemble_path.exists():
        with open(ensemble_path) as f:
            ensemble_results = json.load(f)
        
        print(f"\n🏆 Mejor modelo: {ensemble_results.get('best', 'N/A')}")
        
        if "results" in ensemble_results:
            best_auroc = max(r.get("auroc", 0) for r in ensemble_results["results"].values())
            print(f"📊 Mejor AUROC: {best_auroc:.4f}")
    
    print("\n✅ Pipeline completado!")


if __name__ == "__main__":
    main()
