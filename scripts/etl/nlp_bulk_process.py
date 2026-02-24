import asyncio
import sys
import os
import logging

# Path setup to import unified_app modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, 'services', 'unified_app'))

import db
import processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nlp-bulk")

async def run_batch_upgrade(patient_id: str, total_limit: int = 5000):
    """Run NLP upgrade in batches until limit or no more messages."""
    processed_total = 0
    batch_size = 500
    
    logger.info(f"Starting NLP bulk upgrade for {patient_id}, max {total_limit}")
    
    while processed_total < total_limit:
        limit = min(batch_size, total_limit - processed_total)
        res = await processor.upgrade_nlp_for_patient(patient_id, limit=limit)
        upgraded = res.get("upgraded", 0)
        
        if upgraded == 0:
            logger.info(f"No more messages needing NLP for {patient_id}")
            break
            
        processed_total += upgraded
        logger.info(f"Progress: {processed_total}/{total_limit} upgraded for {patient_id}")
        
    return processed_total

async def main():
    await db.get_pool() # Init DB pool
    
    try:
        patients = await db.fetch_all("SELECT DISTINCT patient_id FROM raw_messages")
        logger.info(f"Found {len(patients)} patients to process")
        
        for row in patients:
            patient_id = row["patient_id"]
            count = await run_batch_upgrade(patient_id, total_limit=20000)
            logger.info(f"Finished bulk upgrade for {patient_id}: {count} messages upgraded")
            
    finally:
        await db.close_pool()

if __name__ == "__main__":
    asyncio.run(main())
