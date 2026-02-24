"""Full System Test - PreSickness"""
import requests
import json

BASE = 'http://localhost:8080'

# Login as patient
print("Logging in...")
r = requests.post(f'{BASE}/api/auth/login', data={'username':'patient1@em.com','password':'test1234'})
token = r.json()['access_token']
headers = {'Authorization': f'Bearer {token}'}

# Test predict
print('\n=== PREDICTION TEST ===')
r = requests.post(f'{BASE}/api/predict', json={'horizon_days': 14}, headers=headers)
print(f'Status: {r.status_code}')
print(json.dumps(r.json(), indent=2))

# Get message stats
print('\n=== MESSAGE STATS ===')
r = requests.get(f'{BASE}/api/events/messages/stats', headers=headers)
print(json.dumps(r.json(), indent=2))

# Get events list
print('\n=== RECENT EVENTS ===')
r = requests.get(f'{BASE}/api/events/', headers=headers)
events = r.json()
print(f'Total: {len(events)} events')
for e in events[:5]:
    print(f"  {e['event_date'][:10]} | {e['event_type']} | {e.get('severity', 'n/a')}")

# Test NLP agent directly
print('\n=== NLP AGENT TEST ===')
try:
    r = requests.post('http://localhost:8002/v1/process', json={
        'text': 'Hoy me siento muy cansada y con dolor de cabeza',
        'timestamp': '2026-02-07T12:00:00Z',
        'language': 'es'
    }, timeout=10)
    print(f'Status: {r.status_code}')
    resp = r.json()
    print(f"Embeddings dim: {len(resp.get('embeddings', []))}")
    print(f"Symptom scores: {resp.get('symptom_scores', {})}")
except Exception as e:
    print(f"Error: {e}")

# Service status summary
print('\n=== SERVICES STATUS ===')
services = [
    ('unified_app', 'http://localhost:8080/health'),
    ('nlp-agent', 'http://localhost:8002/health'),
    ('ml-inference', 'http://localhost:8001/'),
    ('mlflow', 'http://localhost:5000/health'),
    ('postgres', 'http://localhost:5432'),
    ('redis', 'http://localhost:6379'),
]
for name, url in services:
    try:
        if 'postgres' in name or 'redis' in name:
            import socket
            port = int(url.split(':')[-1])
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            result = s.connect_ex(('localhost', port))
            s.close()
            print(f'{name}: {"OK" if result == 0 else "UNREACHABLE"}')
        else:
            r = requests.get(url, timeout=5)
            print(f'{name}: OK ({r.status_code})')
    except Exception as e:
        print(f'{name}: ERROR - {str(e)[:40]}')

print('\n=== TEST COMPLETE ===')
