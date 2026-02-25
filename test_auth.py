import urllib.request
import json
import sys
import httpx

url = "http://127.0.0.1:8010/api/auth/google"
req = urllib.request.Request(
    url,
    data=json.dumps({'code': 'dummy'}).encode(),
    headers={'Content-Type': 'application/json'}
)

try:
    urllib.request.urlopen(req)
except Exception as e:
    if hasattr(e, 'read'):
        print(e.read().decode())
    else:
        print(e)
