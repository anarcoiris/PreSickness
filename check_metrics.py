import requests
BASE = 'http://localhost:8080'

# Login 
r = requests.post(f'{BASE}/api/auth/login', data={'username':'patient1@em.com','password':'test1234'})
token = r.json()["access_token"]
headers = {'Authorization': f'Bearer {token}'}

# Get stats
print('=== MESSAGE STATS ===')
r = requests.get(f'{BASE}/api/events/messages/stats', headers=headers)
print(r.json())

# Get uploads
print('\n=== UPLOADS ===')
r = requests.get(f'{BASE}/api/patients/data', headers=headers)
uploads = r.json()
print(f'Total uploads: {len(uploads)}')
for u in uploads[-3:]:
    print(f'  - {u["filename"]}: processed={u["processed"]}')

# Get events
print('\n=== EVENTS ===')
r = requests.get(f'{BASE}/api/events/', headers=headers)
events = r.json()
print(f'Total events: {len(events)}')
if events:
    for e in events[:5]:
        print(f'  - {e["event_date"][:10]} | {e["event_type"]} | severity={e.get("severity","n/a")}')

# Get clusters
print('\n=== CLUSTERS ===')
r = requests.get(f'{BASE}/api/events/clusters', headers=headers)
clusters = r.json()
print(f'Total clusters: {len(clusters)}')
for c in clusters[:3]:
    print(f'  - {c["start_date"][:10]} to {c["end_date"][:10]} | signals={c["total_signals"]} | is_relapse={c["is_probable_relapse"]}')

# Get prediction
print('\n=== PREDICTION ===')
r = requests.post(f'{BASE}/api/predict', json={'horizon_days': 14}, headers=headers)
if r.status_code == 200:
    p = r.json()
    print(f'Risk: {p["risk_level"]} | Probability: {p["probability"]:.2%}')
else:
    print(f'Error: {r.status_code} {r.text[:200]}')
