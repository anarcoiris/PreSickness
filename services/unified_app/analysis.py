from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
import json
from pathlib import Path
from dependencies import get_current_patient
import logging

# Fallback logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Determine project root based on file location
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"

print(f"[DEBUG] analysis.py starting. PROJECT_ROOT={PROJECT_ROOT}")

def get_patient_data_path(patient_id: str) -> Path:
    print(f"[DEBUG] Resolving data path for patient_id: {patient_id}")
    # Try the specific patient folder first
    target_path = DATA_PATH / patient_id
    
    if target_path.exists():
        return target_path
    
    # Fallback to demo 'paciente1' if it exists
    demo_path = DATA_PATH / "paciente1"
    if demo_path.exists():
        print(f"[DEBUG] Fallback to demo path: {demo_path}")
        return demo_path
        
    print(f"[ERROR] No data path found for {patient_id} or demo 'paciente1'")
    return target_path # Will cause 404 later

@router.get("/training", response_model=Dict[str, Any])
async def get_training_results(patient: dict = Depends(get_current_patient)):
    path = get_patient_data_path(patient["user_id_hash"]) / "training_results.json"
    print(f"[DEBUG] Accessing training results at: {path}")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Training results not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

@router.get("/ensemble", response_model=Dict[str, Any])
async def get_ensemble_results(patient: dict = Depends(get_current_patient)):
    path = get_patient_data_path(patient["user_id_hash"]) / "ensemble_results.json"
    print(f"[DEBUG] Accessing ensemble results at: {path}")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Ensemble results not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

@router.get("/optuna", response_model=Dict[str, Any])
async def get_optuna_results(patient: dict = Depends(get_current_patient)):
    path = get_patient_data_path(patient["user_id_hash"]) / "optuna_results.json"
    print(f"[DEBUG] Accessing optuna results at: {path}")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Optuna results not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

@router.get("/features", response_model=List[Dict[str, Any]])
async def get_feature_importance(patient: dict = Depends(get_current_patient)):
    path = get_patient_data_path(patient["user_id_hash"]) / "training_results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Training results not found at {path}")
    with open(path, "r") as f:
        data = json.load(f)
        return data.get("feature_importance", [])
