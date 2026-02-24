import asyncio
import os
import sys

# Añadir el path al venv o dependencias si es necesario
# Para este script rápido usamos el PYTHONPATH adecuadamente
sys.path.append(os.path.join(os.getcwd(), "services", "unified_app"))

try:
    import db
    import main
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)

async def check_db():
    try:
        pool = await db.get_pool()
        async with pool.connection() as conn:
            # Check if users table exists
            cursor = await conn.execute("SELECT table_name FROM information_schema.tables WHERE table_name = 'users'")
            res = await cursor.fetchone()
            print(f"Users table exists: {res is not None}")
            
            # Check users
            cursor = await conn.execute("SELECT email FROM users")
            users = await cursor.fetchall()
            print(f"Registered users: {users}")
            
            # Check recent register attempt error? No, just try to register manually
            data = {
                "user_id_hash": "test_hash",
                "email": "test@example.com",
                "password_hash": "dummy_hash",
                "name": "Test User"
            }
            try:
                # user = await db.create_user(data)
                # print(f"Created user: {user}")
                pass
            except Exception as e:
                print(f"Error creating user: {e}")
                
    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        await db.close_pool()

if __name__ == "__main__":
    asyncio.run(check_db())
