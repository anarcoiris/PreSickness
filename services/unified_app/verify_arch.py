import asyncio
import httpx
import sys

BASE_URL = "http://localhost:8085"

async def test_auth():
    print("\n--- Testing Password Auth ---")
    async with httpx.AsyncClient() as client:
        # 1. Register
        reg_payload = {
            "email": "test_audit@example.com",
            "password": "securepassword123",
            "name": "Audit User"
        }
        res = await client.post(f"{BASE_URL}/api/auth/register", json=reg_payload)
        if res.status_code == 201:
            print("✓ User registration successful")
        elif res.status_code == 400 and "ya registrado" in res.text:
            print("! User already exists (OK)")
        else:
            print(f"✗ Registration failed: {res.status_code} {res.text}")
            return

        # 2. Login
        login_data = {
            "username": "test_audit@example.com",
            "password": "securepassword123"
        }
        res = await client.post(f"{BASE_URL}/api/auth/login", data=login_data)
        if res.status_code == 200:
            token = res.json()["access_token"]
            print("✓ Login successful")
            return token
        else:
            print(f"✗ Login failed: {res.status_code} {res.text}")
            return None

async def test_google_login():
    print("\n--- Testing Google Login (Mock) ---")
    async with httpx.AsyncClient() as client:
        payload = {"id_token": "demo_token_google_tester@example.com"}
        res = await client.post(f"{BASE_URL}/api/auth/google", json=payload)
        if res.status_code == 200:
            print("✓ Google Mock login successful")
            return res.json()["access_token"]
        else:
            print(f"✗ Google Mock login failed: {res.status_code} {res.text}")
            return None

async def verify_stats(token):
    print("\n--- Testing Data Integrity (Stats) ---")
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{BASE_URL}/api/events/stats", headers=headers)
        if res.status_code == 200:
            data = res.json()
            print(f"✓ Stats retrieved. Total events: {data.get('total_events')}")
            print(f"✓ Pending retraining: {data.get('pending_retraining')}")
        else:
            print(f"✗ Stats retrieval failed: {res.status_code} {res.text}")

async def main():
    print("Starting Architecture Verification...")
    
    # Wait for server if needed
    try:
        token = await test_auth()
        if token:
            await verify_stats(token)
        
        google_token = await test_google_login()
        if google_token:
            print("Verified Google upsertion.")
            
    except Exception as e:
        print(f"FATAL: Connection error. Is the server running on 8080? {e}")

if __name__ == "__main__":
    asyncio.run(main())
