import asyncio
import asyncpg
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def test():
    dsn = "postgresql://emuser:changeme@127.0.0.1:5432/empredictor"
    print(f"Connecting to {dsn}")
    conn = await asyncpg.connect(dsn)
    try:
        val = await conn.fetchval("SELECT COUNT(*) FROM public.users")
        print(f"USER COUNT: {val}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(test())
