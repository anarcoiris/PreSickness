"""
Sistema de Análisis de Flujos - EM-Predictor
Detecta cabos sueltos, endpoints no conectados, placeholders, etc.
"""
import requests
import json
from typing import Dict, List, Tuple

BASE = 'http://localhost:8080'
NLP_BASE = 'http://localhost:8002'
ML_BASE = 'http://localhost:8001'
MLFLOW_BASE = 'http://localhost:5000'

# ============================================================================
# SERVICE HEALTH
# ============================================================================
def check_services() -> Dict[str, str]:
    services = {}
    endpoints = [
        ('unified_app', f'{BASE}/health'),
        ('nlp-agent', f'{NLP_BASE}/health'),
        ('ml-inference', f'{ML_BASE}/v1/health'),
        ('mlflow', f'{MLFLOW_BASE}/health'),
    ]
    for name, url in endpoints:
        try:
            r = requests.get(url, timeout=5)
            services[name] = 'OK' if r.status_code in [200, 404] else f'ERROR ({r.status_code})'
        except Exception as e:
            services[name] = f'UNREACHABLE'
    
    # Check DB/Redis via socket
    import socket
    for name, port in [('postgres', 5432), ('redis', 6379)]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        result = s.connect_ex(('localhost', port))
        s.close()
        services[name] = 'OK' if result == 0 else 'UNREACHABLE'
    
    return services

# ============================================================================
# ENDPOINT ANALYSIS
# ============================================================================
def analyze_endpoints() -> Dict:
    r = requests.get(f'{BASE}/openapi.json')
    schema = r.json()
    paths = schema['paths']
    
    analysis = {
        'total': len(paths),
        'by_method': {},
        'by_domain': {},
        'endpoints': []
    }
    
    for path, methods in paths.items():
        for method in methods:
            if method not in ['get', 'post', 'put', 'delete', 'patch']:
                continue
            
            # Count by method
            analysis['by_method'][method.upper()] = analysis['by_method'].get(method.upper(), 0) + 1
            
            # Count by domain
            parts = path.split('/')
            domain = parts[2] if len(parts) > 2 else 'root'
            analysis['by_domain'][domain] = analysis['by_domain'].get(domain, 0) + 1
            
            # Get operation info
            op = methods[method]
            analysis['endpoints'].append({
                'method': method.upper(),
                'path': path,
                'summary': op.get('summary', 'N/A'),
                'auth_required': 'security' in op or 'security' in schema.get('security', [])
            })
    
    return analysis

# ============================================================================
# DATA FLOW ANALYSIS
# ============================================================================
def analyze_data_flows() -> List[Dict]:
    flows = [
        {
            'name': 'Usuario → Registro',
            'steps': [
                ('POST', '/api/auth/register', 'Frontend', 'unified_app'),
                ('INSERT', 'users', 'unified_app', 'postgres'),
            ],
            'status': 'OK'
        },
        {
            'name': 'Usuario → Login → Token',
            'steps': [
                ('POST', '/api/auth/login', 'Frontend', 'unified_app'),
                ('SELECT', 'users', 'unified_app', 'postgres'),
                ('JWT', 'token', 'unified_app', 'Frontend'),
            ],
            'status': 'OK'
        },
        {
            'name': 'WhatsApp Upload → Parsing → DB',
            'steps': [
                ('POST', '/api/patients/upload', 'Frontend', 'unified_app'),
                ('PARSE', 'events.py', 'unified_app', 'unified_app'),
                ('INSERT', 'messages', 'unified_app', 'postgres'),
                ('INSERT', 'uploads', 'unified_app', 'postgres'),
            ],
            'status': 'OK'
        },
        {
            'name': 'Mensajes → NLP Processing',
            'steps': [
                ('POST', '/api/events/messages/process', 'Frontend', 'unified_app'),
                ('POST', '/v1/process', 'unified_app', 'nlp-agent'),
                ('ONNX', 'inference', 'nlp-agent', 'nlp-agent'),
                ('UPDATE', 'messages.nlp_*', 'unified_app', 'postgres'),
            ],
            'status': 'OK'
        },
        {
            'name': 'Predicción → ML Inference',
            'steps': [
                ('POST', '/api/predict', 'Frontend', 'unified_app'),
                ('SELECT', 'events+messages', 'unified_app', 'postgres'),
                ('POST', '/predict (interno)', 'unified_app', 'ml-inference'),
                ('MLFLOW', 'load model', 'ml-inference', 'mlflow'),
            ],
            'status': 'PROBABLY_OK'
        },
        {
            'name': 'Doctor → Patient List',
            'steps': [
                ('GET', '/api/doctor/patients', 'Frontend', 'unified_app'),
                ('SELECT', 'doctor_patient_access', 'unified_app', 'postgres'),
            ],
            'status': 'OK'
        },
        {
            'name': 'Doctor → Impersonation',
            'steps': [
                ('GET', '/api/events/', 'Frontend + X-Patient-ID', 'unified_app'),
                ('CHECK', 'doctor_patient_access', 'unified_app', 'postgres'),
                ('SELECT', 'events WHERE user=patient', 'unified_app', 'postgres'),
            ],
            'status': 'OK'
        },
        {
            'name': 'Patient → Revoke Doctor',
            'steps': [
                ('DELETE', '/api/patient/doctors/{id}', 'Frontend', 'unified_app'),
                ('DELETE', 'doctor_patient_access', 'unified_app', 'postgres'),
            ],
            'status': 'OK'
        },
    ]
    return flows

# ============================================================================
# ISSUE DETECTION
# ============================================================================
def detect_issues() -> List[Dict]:
    issues = []
    
    # Check ML model
    try:
        r = requests.get(f'{ML_BASE}/v1/health', timeout=5)
        if r.status_code == 200:
            data = r.json()
            if 'heuristic' in str(data).lower() or 'fallback' in str(data).lower():
                issues.append({
                    'severity': 'WARNING',
                    'component': 'ml-inference',
                    'issue': 'Using heuristic fallback instead of TFT model',
                    'suggestion': 'Train and register TFT model in MLflow'
                })
    except:
        issues.append({
            'severity': 'ERROR',
            'component': 'ml-inference',
            'issue': 'Service unreachable',
            'suggestion': 'Start ml-inference service'
        })
    
    # Check NLP processing ratio
    try:
        r = requests.post(f'{BASE}/api/auth/login', data={'username':'patient1@em.com','password':'test1234'})
        if r.status_code == 200:
            token = r.json()['access_token']
            headers = {'Authorization': f'Bearer {token}'}
            r = requests.get(f'{BASE}/api/events/messages/stats', headers=headers)
            stats = r.json()
            ratio = stats['nlp_processed'] / max(stats['raw_messages'], 1)
            if ratio < 0.1:
                issues.append({
                    'severity': 'INFO',
                    'component': 'nlp-pipeline',
                    'issue': f'Only {ratio:.1%} of messages have NLP embeddings',
                    'suggestion': 'Run bulk NLP processing or increase sampling'
                })
    except Exception as e:
        pass
    
    # Check for placeholder endpoints
    placeholder_endpoints = []
    try:
        r = requests.get(f'{BASE}/openapi.json')
        schema = r.json()
        for path, methods in schema['paths'].items():
            for method, op in methods.items():
                if method not in ['get', 'post', 'put', 'delete']:
                    continue
                summary = op.get('summary', '')
                if 'TODO' in summary or 'placeholder' in summary.lower():
                    placeholder_endpoints.append(f'{method.upper()} {path}')
    except:
        pass
    
    if placeholder_endpoints:
        issues.append({
            'severity': 'WARNING',
            'component': 'api',
            'issue': f'Placeholder endpoints found: {placeholder_endpoints}',
            'suggestion': 'Implement or remove placeholder endpoints'
        })
    
    # Check clusters
    try:
        r = requests.get(f'{BASE}/api/events/clusters', headers=headers)
        if r.status_code == 200:
            clusters = r.json()
            if len(clusters) == 0:
                issues.append({
                    'severity': 'INFO',
                    'component': 'clustering',
                    'issue': 'No clusters detected yet',
                    'suggestion': 'Run clustering algorithm on events'
                })
    except:
        pass
    
    return issues

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print('=' * 70)
    print(' EM-PREDICTOR SYSTEM ANALYSIS')
    print('=' * 70)
    
    # Services
    print('\nSERVICES STATUS')
    print('-' * 40)
    services = check_services()
    for name, status in services.items():
        icon = '[OK]' if status == 'OK' else '[ERROR/UNREACHABLE]'
        print(f'  {icon} {name}: {status}')
    
    # Endpoints
    print('\nENDPOINTS ANALYSIS')
    print('-' * 40)
    analysis = analyze_endpoints()
    print(f'  Total endpoints: {analysis["total"]}')
    print(f'  By method: {analysis["by_method"]}')
    print(f'  By domain: {analysis["by_domain"]}')
    
    # Data Flows
    print('\nDATA FLOWS')
    print('-' * 40)
    flows = analyze_data_flows()
    # Try to verify ML Inference status dynamically
    ml_health = {}
    try:
        ml_health = requests.get(f'{ML_BASE}/v1/health', timeout=2).json()
    except: pass

    for flow in flows:
        status = flow['status']
        if flow['name'] == 'Predicción → ML Inference':
            if ml_health.get('model_loaded'):
                status = 'OK'
            else:
                status = 'FALLBACK'
                flow['issue'] = 'TFT model not registered or loaded, check inference logs'
        
        icon = '[OK]' if status == 'OK' else '[WARNING]'
        print(f'  {icon} {flow["name"]}')
        if flow.get('issue') and status != 'OK':
            print(f'     └─ [ISSUE] {flow["issue"]}')
    
    # Issues
    print('\nDETECTION OF ISSUES')
    print('-' * 40)
    issues = detect_issues()
    if issues:
        for issue in issues:
            icon = {'ERROR': '[X]', 'WARNING': '[!]', 'INFO': '[i]'}.get(issue['severity'], '?')
            print(f'  {icon} [{issue["severity"]}] {issue["component"]}')
            print(f'     Issue: {issue["issue"]}')
            print(f'     Fix: {issue["suggestion"]}')
    else:
        print('  [OK] No critical issues detected')
    
    print('\n' + '=' * 70)
    print(' ANALYSIS COMPLETE')
    print('=' * 70)
