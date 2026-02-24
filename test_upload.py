import requests
import time

BASE = 'http://localhost:8080'

# Login as patient
print("Logging in...")
r = requests.post(f'{BASE}/api/auth/login', data={'username':'patient1@em.com','password':'test1234'}, timeout=30)
if r.status_code != 200:
    print(f"LOGIN ERROR: {r.status_code} {r.text}")
    exit(1)

token = r.json()['access_token']
headers = {'Authorization': f'Bearer {token}'}

# Get pre-upload stats
r = requests.get(f'{BASE}/api/events/messages/stats', headers=headers, timeout=30)
print(f"PRE-UPLOAD STATS: {r.json()}")

# Upload WhatsApp file
print("Uploading 4MB WhatsApp file...")
start = time.time()
with open(r'c:\Users\soyko\Documents\PreSickness\datos\paciente1_whatsapp.txt', 'rb') as f:
    files = {'file': ('paciente1_whatsapp.txt', f, 'text/plain')}
    r = requests.post(f'{BASE}/api/patients/upload', files=files, headers=headers, timeout=600)
elapsed = time.time() - start

if r.status_code in [200, 201]:
    print(f"UPLOAD SUCCESS in {elapsed:.1f}s: {r.json()}")
else:
    print(f"UPLOAD ERROR {r.status_code}: {r.text[:500]}")

# Get post-upload stats
r = requests.get(f'{BASE}/api/events/messages/stats', headers=headers, timeout=30)
print(f"POST-UPLOAD STATS: {r.json()}")
