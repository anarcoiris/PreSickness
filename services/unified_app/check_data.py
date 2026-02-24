import asyncio
import sys
import os

# Add current dir to path for db import
sys.path.append(os.getcwd())

import db

async def main():
    try:
        print("---DATA_CHECK_START---")
        m_count = await db.count_raw_messages("paciente1")
        e_stats = await db.get_event_stats("paciente1")
        import json
        result = {
            "messages": m_count,
            "events": {k: str(v) if hasattr(v, 'isoformat') else v for k, v in e_stats.items()}
        }
        print(json.dumps(result, indent=2))
        print("---DATA_CHECK_END---")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await db.close_pool()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
