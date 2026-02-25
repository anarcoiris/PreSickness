import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "services", "unified_app"))
from db import get_pool

async def main():
    pool = await get_pool()
    async with pool.connection() as conn:
        # insert user
        await conn.execute("INSERT INTO users (user_id_hash, email, name, password_hash) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING", ("2"*64, "mock@user.com", "Mock User", "hash"))
        # insert device
        await conn.execute("INSERT INTO devices (device_id_hash, secret, user_id_hash) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING", ("1"*64, "test_device_secret", "2"*64))

        # add redis device token
        import redis.asyncio as redis
        from dependencies import settings
        r = redis.from_url(settings.redis_url, password=settings.redis_password)
        await r.set(f"{settings.device_token_prefix}:dummy_token", "1"*64)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
