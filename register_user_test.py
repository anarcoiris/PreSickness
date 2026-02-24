import httpx
import asyncio

async def register():
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                "http://localhost:8080/api/auth/register",
                json={
                    "email": "test@example.com",
                    "password": "password123",
                    "name": "Test User"
                }
            )
            print(f"Status: {resp.status_code}")
            print(f"Body: {resp.text}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(register())
