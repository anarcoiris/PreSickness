import requests
import asyncio
import hmac
import hashlib
import json
from datetime import datetime, timezone
import uuid

# Configuration
API_URL = "http://localhost:8010/v1/ingest"
DEVICE_SECRET = "test_device_secret"

# Create a deterministic mock device for testing 
device_id_hash = "1" * 64
user_id_hash = "2" * 64

def generate_signature(user_hash: str, device_hash: str, timestamp_str: str, salt: str, secret: str) -> str:
    canonical = f"{user_hash}|{device_hash}|{timestamp_str}|{salt}"
    return hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()

def send_mock_datapoint():
    now_utc = datetime.now(timezone.utc)
    ts_str = now_utc.isoformat()
    salt = "s" * 32
    
    payload = {
        "user_id_hash": user_id_hash,
        "device_id_hash": device_id_hash,
        "timestamp": ts_str,
        "embedding": {
            "embedding_encrypted": "encrypted_dummy_data",
            "embedding_dim": 768,
            "salt": salt
        },
        "numeric_features": {
            "sentiment_score": 0.5,
            "avg_sentence_len": 10.5,
            "type_token_ratio": 0.7,
            "num_messages": 5,
            "steps": 1000,
            "sleep_hours": 7.5
        }
    }
    
    # Calculate signature
    payload["signature"] = generate_signature(user_id_hash, device_id_hash, ts_str, salt, DEVICE_SECRET)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer dummy_token"
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    send_mock_datapoint()
