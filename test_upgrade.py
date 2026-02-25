import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath('services/unified_app'))

from processor import upgrade_nlp_for_patient
import db

async def test_upgrade():
    print('Connecting to DB...')
    await db.get_pool()
    
    print('Testing NLP Upgrade Pipeline directly...')
    result = await upgrade_nlp_for_patient('paciente2', limit=5)
    print('Result:', result)
    
    # Verify DB rows
    pool = await db.get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT numeric_features FROM datapoints WHERE nlp_level=1 AND user_id_hash='paciente2' LIMIT 2;")
            rows = await cur.fetchall()
            print("Extracted Features in DB:")
            for row in rows:
                print(row['numeric_features'])
    
    print('Done!')

asyncio.run(test_upgrade())
