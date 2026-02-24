import asyncio
import sys
import os

# Add current dir to path for db import
sys.path.append(os.getcwd())

import db
from events import load_initial_data_async

async def main():
    try:
        print("Starting manual data ingestion...")
        # Force load by bypassing checks if needed? 
        # Actually load_initial_data_async checks total_events == 0.
        # Let's check it.
        await load_initial_data_async()
        
        # Verify
        m_count = await db.count_raw_messages("paciente1")
        e_stats = await db.get_event_stats("paciente1")
        print(f"Post-ingestion messages: {m_count}")
        print(f"Post-ingestion events: {e_stats['total_events']}")
    except Exception as e:
        print(f"Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await db.close_pool()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
