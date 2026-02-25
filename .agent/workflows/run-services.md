---
description: Levantar todos los servicios del stack EM-Predictor
---

# Workflow: Levantar Servicios

Este workflow inicia todo el stack de microservicios para desarrollo y testing, apoyándose en Docker para infraestructura y procesos nativos para la lógica de negocio.

## Pasos

// turbo
### 1. Levantar infraestructura base

```powershell
docker-compose up -d postgres redis minio redpanda mlflow
```

// turbo
### 2. Verificar salud de infraestructura

```powershell
docker-compose ps
docker-compose logs postgres --tail=20
```

### 3. Levantar servicios nativos (Aplicación)

Se recomienda encarecidamente usar el script unificado que inicializa todos los entornos virtuales y procesos automáticamente:

```powershell
.\start_all.bat
```

### 4. Verificar endpoints

| Servicio | URL | Verificación |
|----------|-----|--------------|
| Unified App | http://localhost:8010/docs | Swagger UI |
| ML Inference | http://localhost:8001/docs | Swagger UI |
| NLP Agent | http://localhost:8002/health | Health check |
| Webapp | http://localhost:5173 | Interfaz Web |
| MLflow | http://localhost:5000 | Experiments |

// turbo
### 5. Test de health check unificado

```powershell
curl http://localhost:8010/api/metrics
```

## Troubleshooting

- Si Postgres o MinIO no inician: `docker-compose down -v` (CUIDADO: Borra datos) y luego `docker-compose up -d`
- Si hay conflictos de puertos nativos: verificar con `netstat -an | findstr :8010` o matando los procesos de python desde el Administrador de Tareas.
