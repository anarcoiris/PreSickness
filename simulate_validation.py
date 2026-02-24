import asyncio
import os
import sys
from pathlib import Path
from uuid import UUID

# Add unified_app to path
ROOT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(ROOT_DIR, 'services', 'unified_app'))

import db
from events import validate_cluster, ValidateClusterRequest
from datetime import datetime, timezone

async def simulate_validation():
    await db.get_pool()
    try:
        # 1. Find a pending cluster
        cluster = await db.fetch_one("SELECT id FROM auto_clusters WHERE status = 'pending' LIMIT 1")
        if not cluster:
            print("No pending clusters found to validate.")
            return
            
        cluster_id = cluster['id']
        print(f"Attempting to validate cluster {cluster_id}")
        
        # 2. Prepare request data
        data = ValidateClusterRequest(
            event_date=datetime.now(timezone.utc),
            event_type="confirmed_relapse",
            notes="Simulated validation"
        )
        
        # 3. Call validate_cluster
        # We need to mock the patient dict returned by get_current_patient
        patient = {"user_id_hash": "paciente1"}
        
        try:
            result = await validate_cluster(cluster_id, data, patient)
            print("Validation successful!")
            print(f"Created event: {result['id']}")
            
            # 4. Check if cluster status was updated
            updated = await db.fetch_one("SELECT status FROM auto_clusters WHERE id = %s", cluster_id)
            print(f"Cluster status in DB: {updated['status']}")
            
        except Exception as e:
            print(f"Validation failed with error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            
    finally:
        await db.close_pool()

if __name__ == "__main__":
    asyncio.run(simulate_validation())
