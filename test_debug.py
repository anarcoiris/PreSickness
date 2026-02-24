import requests
import time

BASE_URL = "http://localhost:8080/api/events"
RELOAD_URL = "http://localhost:8001/v1/reload"
TRAIN_URL = "http://localhost:8080/api/events/trigger-retraining"

def test_retraining_trigger():
    print("Testing Retraining Trigger...")
    try:
        resp = requests.post(TRAIN_URL)
        if resp.status_code == 200:
            print(f"Trigger OK: {resp.json()}")
        else:
            print(f"Trigger FAILED: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Connection Error: {e}")

def test_inference_reload():
    print("\nTesting Inference Reload Endpoint...")
    try:
        resp = requests.post(RELOAD_URL)
        if resp.status_code == 200:
            print(f"Reload OK: {resp.json()}")
        else:
            print(f"Reload FAILED: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    # Wait for services to be up
    time.sleep(5)
    test_retraining_trigger()
    test_inference_reload()
