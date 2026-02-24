import asyncio
import os
import sys

# Fixed for Windows ProactorEventLoop issues with Psycopg
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Simplified db test
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "empredictor"
DB_USER = "emuser"
DB_PASSWORD = "changeme"
DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

async def test_conn():
    try:
        import psycopg
        from psycopg.rows import dict_row
        async with await psycopg.AsyncConnection.connect(DSN, row_factory=dict_row) as conn:
            print("Connected!")
            # Check users table
            async with conn.cursor() as cur:
                await cur.execute("SELECT * FROM users LIMIT 1")
                row = await cur.fetchone()
                print(f"First user: {row}")
                
                # Create test user if not exists
                import hashlib
                from passlib.context import CryptContext
                pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                
                email = "test@example.com"
                password_hash = pwd_context.hash("password123")
                user_id_hash = hashlib.sha256(email.encode()).hexdigest()[:16]
                
                await cur.execute("SELECT * FROM users WHERE email = %s", (email,))
                if not await cur.fetchone():
                    await cur.execute(
                        "INSERT INTO users (user_id_hash, email, password_hash, name, consent_given_at) VALUES (%s, %s, %s, %s, NOW())",
                        (user_id_hash, email, password_hash, "Test User")
                    )
                    await conn.commit()
                    print("Created test user!")
                else:
                    print("Test user already exists.")
                    
    except ImportError:
        print("psycopg not installed")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_conn())
