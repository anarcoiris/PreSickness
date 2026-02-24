import requests

BASE_URL = "http://localhost:8080/api/analysis/training"
HEADERS = {
    # We need a token. Let's see if we can use a dummy one if auth is mocked or get one.
    "Authorization": "Bearer demo_token_patient1@em.com" # Assuming demo auth works
}

def test_endpoint():
    try:
        r = requests.get(BASE_URL, headers=HEADERS)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_endpoint()
