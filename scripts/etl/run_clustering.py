import asyncio
import sys
import os
import logging
from datetime import datetime
import pandas as pd
from uuid import UUID

# Path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, 'services', 'unified_app'))
sys.path.append(os.path.join(ROOT_DIR, 'scripts', 'etl'))

import db
import cluster_signals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clustering")

async def run_clustering_for_patient(patient_id: str):
    logger.info(f"Running clustering for {patient_id}")
    
    # 1. Fetch events from DB
    events = await db.fetch_all(
        "SELECT id, event_date as date, event_type, severity, notes FROM clinical_events WHERE patient_id = %s",
        patient_id
    )
    
    if not events:
        logger.warning(f"No events found for {patient_id}")
        return
        
    df = pd.DataFrame(events)
    df["date"] = pd.to_datetime(df["date"])
    
    # 2. Compute daily scores
    daily_scores = cluster_signals.calculate_daily_signal_score(df)
    
    # 3. Find clusters
    clusters = cluster_signals.find_clusters(daily_scores)
    logger.info(f"Detected {len(clusters)} clusters for {patient_id}")
    
    # 4. Store clusters in DB
    for c in clusters:
        # Insert into auto_clusters
        res = await db.fetch_one(
            """
            INSERT INTO auto_clusters (
                patient_id, start_date, end_date, peak_date, total_signals, 
                unique_types, max_severity, severity_score, density, is_probable_relapse,
                confidence, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            patient_id, c.start_date.date(), c.end_date.date(), c.peak_date.date(),
            c.total_signals, c.unique_event_types, c.max_severity,
            float(c.severity_score), float(c.density), c.is_probable_relapse,
            0.8, 'pending'
        )
        
        if res:
            cluster_id = res['id']
            # 5. Link events to this cluster
            # Simple heuristic: events within cluster dates
            await db.execute_query(
                "UPDATE clinical_events SET cluster_id = %s WHERE patient_id = %s AND event_date::date >= %s AND event_date::date <= %s",
                cluster_id, patient_id, c.start_date.date(), c.end_date.date()
            )
            
    logger.info(f"Clustering complete for {patient_id}")

async def main():
    await db.get_pool()
    try:
        patients = await db.fetch_all("SELECT DISTINCT patient_id FROM clinical_events")
        for row in patients:
            await run_clustering_for_patient(row["patient_id"])
    finally:
        await db.close_pool()

if __name__ == "__main__":
    asyncio.run(main())
