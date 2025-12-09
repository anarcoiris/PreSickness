---
description: Levantar todos los servicios del stack EM-Predictor
---

# Workflow: Levantar Servicios

Este workflow inicia todo el stack de microservicios para desarrollo y testing.

## Pasos

// turbo
### 1. Levantar infraestructura base

```powershell
cd c:\Users\aladin\Documents\Presickness
docker-compose up -d postgres redis minio redpanda mlflow
```

// turbo
### 2. Verificar salud de servicios

```powershell
docker-compose ps
docker-compose logs postgres --tail=20
```

### 3. Levantar servicios de aplicación

```powershell
docker-compose up -d api_gateway feature_extractor ml_inference alert_manager dashboard
```

### 4. Verificar endpoints

| Servicio | URL | Verificación |
|----------|-----|--------------|
| API Gateway | http://localhost:8000/docs | Swagger UI |
| ML Inference | http://localhost:8001/docs | Swagger UI |
| Dashboard | http://localhost:8501 | Streamlit |
| MLflow | http://localhost:5000 | Experiments |
| Grafana | http://localhost:3000 | Monitoring |

// turbo
### 5. Test de health check

```powershell
curl http://localhost:8000/v1/health
```

## Para Desarrollo Local (sin Docker)

```powershell
# Activar venv
.venv\Scripts\activate

# API Gateway
cd services/api-gateway
uvicorn main:app --reload --port 8000

# Dashboard
cd services/dashboard
streamlit run app.py
```

## Troubleshooting

- Si Postgres no inicia: `docker-compose down -v && docker-compose up -d postgres`
- Si hay conflictos de puertos: verificar con `netstat -an | findstr :8000`
