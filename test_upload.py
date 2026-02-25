import requests
import os

url = "http://127.0.0.1:8010/api/events/import/confirm"
file_path = r"c:\Users\soyko\Documents\PreSickness\datos\paciente2_whatsapp.txt"

print(f"Uploading {file_path} to {url}...")

with open(file_path, "rb") as f:
    files = {"file": ("paciente2_whatsapp.txt", f, "text/plain")}
    data = {"user_id_hash": "tester123", "source": "whatsapp"}
    
    try:
        response = requests.post(url, files=files, data=data)
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
    except Exception as e:
        print("Request failed:", e)
