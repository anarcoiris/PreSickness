import asyncio
import sys
import os
import logging
from datetime import datetime

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Path setup to import feature-extractor modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, 'services', 'feature-extractor'))

import worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("feature-backfill")

async def main():
    # Use local settings for DB/Redis
    settings = worker.Settings(
        db_host="localhost",
        redis_url="redis://:changeme@localhost:6379/0",
        nlp_agent_url="http://localhost:8002/v1/process"
    )
    
    extractor = worker.FeatureExtractor(settings)
    await extractor.initialize()
    
    try:
        # Get date range and patient IDs
        async with extractor.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT user_id_hash, MIN(time) as start_time, MAX(time) as end_time FROM datapoints GROUP BY user_id_hash")
                patients = await cur.fetchall()
        
        logger.info(f"Found {len(patients)} patients to process features")
        
        from datetime import timedelta
        
        for row in patients:
            user_id = row["user_id_hash"]
            dt_start = row["start_time"]
            dt_end = row["end_time"]
            
            logger.info(f"Generating historical windows for patient {user_id} from {dt_start} to {dt_end}")
            
            current_date = dt_start
            while current_date <= dt_end:
                # Generate snapshots for all window sizes at this 'current_date'
                try:
                    features = await extractor.compute_windowed_features(user_id, reference_date=current_date)
                    await extractor.store_windowed_features(user_id, features)
                except Exception as e:
                    logger.error(f"Error at {current_date} for {user_id}: {e}")
                
                current_date += timedelta(days=1)
                
            logger.info(f"Successfully generated historical features for {user_id}")
                
    finally:
        await extractor.close()

if __name__ == "__main__":
    asyncio.run(main())
