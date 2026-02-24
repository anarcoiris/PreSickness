import asyncio
import os
import sys
from pathlib import Path

# Add unified_app to path
ROOT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(ROOT_DIR, 'services', 'unified_app'))

import db

async def check_clusters():
    await db.get_pool()
    try:
        print("Checking clusters...")
        clusters = await db.fetch_all("SELECT id, patient_id, status FROM auto_clusters")
        if not clusters:
            print("No clusters found in auto_clusters table.")
            return
            
        print(f"Found {len(clusters)} clusters:")
        for c in clusters:
            pid = c['patient_id']
            is_match = (pid == "paciente1")
            print(f"ID: {c['id']}, Patient: {repr(pid)}, Match 'paciente1': {is_match}, Status: {c['status']}")
            
        events = await db.fetch_all("SELECT id, patient_id, event_type FROM clinical_events LIMIT 5")
        print(f"\nSample events (first 5):")
        for e in events:
             pid = e['patient_id']
             is_match = (pid == "paciente1")
             print(f"ID: {e['id']}, Patient: {repr(pid)}, Match 'paciente1': {is_match}, Type: {e['event_type']}")
             
    finally:
        await db.close_pool()

if __name__ == "__main__":
    asyncio.run(check_clusters())
