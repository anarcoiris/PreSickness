import asyncio
import os

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "empredictor"
DB_USER = "emuser"
# DB_PASSWORD = "changeme" # Intentar sin pass

# DSN sin pass
DSN = f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

async def test():
    try:
        from psycopg_pool import AsyncConnectionPool
        print("Driver loaded.")
        pool = AsyncConnectionPool(DSN, open=False)
        await pool.open()
        print("Connected OK!")
        async with pool.connection() as conn:
            res = await conn.execute("SELECT 1")
            print(f"Query Result: {await res.fetchone()}")
        await pool.close()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
