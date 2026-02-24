import asyncio
import os
import sys
import hashlib
from datetime import datetime

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

async def create_admin():
    try:
        import psycopg
        from psycopg.rows import dict_row
        from passlib.context import CryptContext
        
        # Use same scheme as main.py
        pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

        async with await psycopg.AsyncConnection.connect(DSN, row_factory=dict_row) as conn:
            print("Connected to DB.")
            async with conn.cursor() as cur:
                email = "admin@example.com"
                password = "adminpassword123"
                password_hash = pwd_context.hash(password)
                user_id_hash = hashlib.sha256(email.encode()).hexdigest()[:16]
                
                await cur.execute("SELECT * FROM users WHERE email = %s", (email,))
                if not await cur.fetchone():
                    await cur.execute(
                        "INSERT INTO users (user_id_hash, email, password_hash, name, role, consent_given_at) VALUES (%s, %s, %s, %s, %s, NOW())",
                        (user_id_hash, email, password_hash, "Administrator", "admin")
                    )
                    await conn.commit()
                    print(f"Created Admin user: {email}")
                else:
                    await cur.execute(
                        "UPDATE users SET password_hash = %s, role = 'admin' WHERE email = %s",
                        (password_hash, email)
                    )
                    await conn.commit()
                    print(f"Updated Admin password and role for: {email}")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(create_admin())
