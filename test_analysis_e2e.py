import requests

LOGIN_URL = "http://localhost:8080/api/auth/login"
ANALYSIS_URL = "http://localhost:8080/api/analysis/training"

def test_full_analysis_access():
    # 1. Login
    login_data = {
        "username": "patient1@em.com",
        "password": "test"
    }
    print(f"Logging in as {login_data['username']}...")
    try:
        r_login = requests.post(LOGIN_URL, data=login_data)
        if r_login.status_code != 200:
            print(f"Login FAILED: {r_login.status_code} - {r_login.text}")
            return
        
        token = r_login.json()["access_token"]
        print("Login successful.")
        
        # 2. Call Analysis
        headers = {"Authorization": f"Bearer {token}"}
        
        for endpoint in ["training", "ensemble", "optuna"]:
            url = f"http://localhost:8080/api/analysis/{endpoint}"
            print(f"Calling {url}...")
            r = requests.get(url, headers=headers)
            print(f"[{endpoint}] Status: {r.status_code}")
            if r.status_code != 200:
                print(f"[{endpoint}] FAILED: {r.text}")
            else:
                print(f"[{endpoint}] OK.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_full_analysis_access()
