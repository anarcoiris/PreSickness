"""
Master script to reproduce the complete training pipeline.

Usage:
    python scripts/reproduce_pipeline.py [--skip-tft] [--skip-ensemble]

This script:
1. Runs train_tft.py (Deep Learning - Temporal Fusion Transformer)
2. Runs train_ensemble.py (Classical ML - RF, GBM, etc.)
3. Compares and reports results
"""
import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path


def run_script(script_name: str) -> bool:
    """Run a Python script and return success status."""
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        script_path = Path(__file__).parent.parent / script_name
    
    print(f"\n{'='*60}")
    print(f"Running: {script_path.name}")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(Path(__file__).parent.parent)
    )
    
    return result.returncode == 0


def load_results() -> dict:
    """Load all available training results."""
    results = {}
    
    # Ensemble results
    ensemble_path = Path("data/processed/paciente1/ensemble_results.json")
    if ensemble_path.exists():
        with open(ensemble_path) as f:
            results['ensemble'] = json.load(f)
    
    # TFT results (from MLflow or checkpoint)
    # TODO: Add MLflow query for TFT metrics
    
    return results


def print_summary(results: dict):
    """Print comparison summary."""
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    
    if 'ensemble' in results:
        print("\n📊 Classical ML Ensemble:")
        for model, metrics in results['ensemble'].get('results', {}).items():
            print(f"  {model:15} | AUROC: {metrics.get('auroc', 0):.4f} | AUPRC: {metrics.get('auprc', 0):.4f}")
        
        best = results['ensemble'].get('best', 'N/A')
        print(f"\n  🏆 Best Model: {best}")
    
    print("\n" + "="*60)


async def main():
    parser = argparse.ArgumentParser(description="Reproduce complete training pipeline")
    parser.add_argument('--skip-tft', action='store_true', help='Skip TFT training')
    parser.add_argument('--skip-ensemble', action='store_true', help='Skip ensemble training')
    args = parser.parse_args()
    
    success = True
    
    # 1. TFT Training
    if not args.skip_tft:
        print("\n🧠 Phase 1: Deep Learning (TFT)")
        tft_success = run_script("train_tft.py")
        if not tft_success:
            print("⚠️  TFT training failed or skipped (may need DB data)")
    
    # 2. Ensemble Training
    if not args.skip_ensemble:
        print("\n🌲 Phase 2: Classical ML Ensemble")
        ensemble_success = run_script("train_ensemble.py")
        if not ensemble_success:
            print("❌ Ensemble training failed")
            success = False
    
    # 3. Load and compare results
    results = load_results()
    print_summary(results)
    
    if success:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n⚠️  Pipeline completed with warnings")


if __name__ == "__main__":
    asyncio.run(main())
