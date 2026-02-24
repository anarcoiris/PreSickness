# Guía de Inicio Rápido

> **Referencia**: [ROADMAP.md](ROADMAP.md) | [ARCHITECTURE.md](ARCHITECTURE.md)

## Requisitos

- Python 3.10+
- Node.js 18+ (para frontend)
- Docker Desktop
- 8GB RAM mínimo

---

## 🚀 Inicio con Script (Recomendado)

```bash
# Desde la raíz del proyecto
start_all.bat
```

Esto iniciará automáticamente:
1. Contenedores Docker (Postgres, Redis, MLflow)
2. Backend API (puerto 8080)
3. NLP Agent (puerto 8002)
4. ML Inference (puerto 8001)
5. Frontend (puerto 5173)

**Parar todo:**
```bash
stop_all.bat
```

---

## 🛠️ Instalación Manual

### 1. Instalar dependencias

```bash
# Backend
pip install -r requirements.txt

# Frontend
cd services/webapp
npm install
```

### 2. Iniciar Docker

```bash
# Levantar containers
docker start em_postgres em_redis

# O crear desde cero
docker-compose up -d
```

### 3. Iniciar servicios

```bash
# Terminal 1: NLP Agent
cd services/nlp-agent
python main.py

# Terminal 2: ML Inference
cd services/ml-inference
python main.py

# Terminal 3: Backend API
cd services/unified_app
python main.py --port 8080

# Terminal 4: Frontend
cd services/webapp
npm run dev
```

---

## 📍 URLs de Servicios

| Servicio | URL |
|----------|-----|
| **Frontend** | http://localhost:5173 |
| **API Docs** | http://localhost:8080/docs |
| **NLP Health** | http://localhost:8002/health |
| **MLflow** | http://localhost:5000 |

---

## 📤 Procesar Datos de Paciente

### 1. Subir archivo WhatsApp

Coloca el export en `datos/`:
```
datos/paciente1_whatsapp.txt
```

### 2. Via API (recomendado)

```bash
# Login
curl -X POST http://localhost:8080/api/auth/login \
  -d "username=tu@email.com&password=password"

# Subir archivo
curl -X POST http://localhost:8080/api/patients/upload \
  -H "Authorization: Bearer TOKEN" \
  -F "file=@datos/paciente1_whatsapp.txt"
```

### 3. Via Frontend

1. Ir a http://localhost:5173
2. Login como paciente
3. Configuración → Subir datos

---

## 🔮 Obtener Predicción

### Via API

```bash
curl -X POST http://localhost:8080/api/predict \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"horizon_days": 14}'
```

**Respuesta:**
```json
{
  "probability": 0.439,
  "risk_level": "warning",
  "horizon_days": 14,
  "generated_at": "2026-02-07T20:41:57Z"
}
```

### Via Frontend

Dashboard → Botón "Calcular Predicción"

---

## 🔧 Troubleshooting

### Error: "Connection refused" en puerto 8080
```bash
# Verificar que el backend está corriendo
netstat -ano | findstr ":8080"

# Reiniciar backend
cd services/unified_app
python main.py --port 8080
```

### Error: "MLflow connection failed"
```bash
# Iniciar MLflow
docker start mlflow

# O crear container
docker run -d -p 5000:5000 --name mlflow python:3.10-slim \
  bash -c "pip install mlflow && mlflow server --host 0.0.0.0"
```

### Error: "No module found"
```bash
pip install -r services/unified_app/requirements.txt
pip install -r services/nlp-agent/requirements.txt
```

---

## 📊 Verificar Sistema

```bash
python test_system.py
```

**Salida esperada:**
```
=== PREDICTION TEST ===
Status: 200
probability: 0.43, risk_level: warning

=== SERVICES STATUS ===
unified_app: OK (200)
nlp-agent: OK (200)
ml-inference: OK (404)
mlflow: OK (200)
postgres: OK
redis: OK
```
