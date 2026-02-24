import asyncio
import logging
import sys
from pathlib import Path
from uuid import UUID

# Adjust import path if needed (though running from root usually works for scripts)
# Assuming run from services/unified_app or root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)

async def run_retraining_pipeline(patient_id: str, job_id: UUID):
    """
    Background Task:
    1. Regenerate labels (ETL)
    2. Train TFT model
    3. Update system status
    """
    logger.info(f"[JOB {job_id}] Starting retraining pipeline for {patient_id}")
    
    try:
        # 1. Train Model (TFT)
        # Note: regenerate_labels.py is skipped as train_tft.py loads from DB directly
        train_script = PROJECT_ROOT / "train_tft.py"
        if not train_script.exists():
            logger.error(f"[JOB {job_id}] Training script not found: {train_script}")
            return

        cmd_train = [sys.executable, str(train_script)]
        logger.info(f"[JOB {job_id}] Running: {' '.join(cmd_train)}")
        
        proc_train = await asyncio.create_subprocess_exec(
            *cmd_train,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc_train.communicate()
        
        if proc_train.returncode != 0:
             logger.error(f"[JOB {job_id}] Training failed: {stderr.decode()}")
             # Log stdout too for debugging
             logger.error(f"[JOB {job_id}] Training stdout: {stdout.decode()}")
             return
             
        logger.info(f"[JOB {job_id}] Training pipeline completed successfully.")
        
        # 3. Reload Model in Inference Service
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post("http://localhost:8001/v1/reload")
                if resp.status_code == 200:
                    logger.info(f"[JOB {job_id}] ML Inference reloaded: {resp.json()}")
                else:
                    logger.warning(f"[JOB {job_id}] ML Inference reload failed: {resp.status_code}")
        except Exception as e:
            logger.error(f"[JOB {job_id}] ML Inference connection failed: {e}")
        
    except Exception as e:
        logger.error(f"[JOB {job_id}] Pipeline exception: {e}")
