---
description: Desplegar servicios EM-Predictor con ngrok para acceso remoto
---

# Workflow: Despliegue con ngrok

Expone los servicios del stack EM-Predictor a través de túneles ngrok para acceso remoto seguro.

## Requisitos Previos

- ngrok instalado y autenticado (`ngrok authtoken <token>`)
- Docker Desktop corriendo
- Puerto 8080 libre (unified gateway)

---

## Pasos

// turbo
### 1. Verificar ngrok instalado

```powershell
ngrok version
```

// turbo
### 2. Levantar infraestructura base

```powershell
cd c:\Users\aladin\Documents\Presickness
docker-compose up -d postgres redis
```

// turbo
### 3. Verificar servicios base

```powershell
docker-compose ps
```

### 4. Levantar aplicación unificada

```powershell
cd c:\Users\aladin\Documents\Presickness\services\unified_app
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```


### 5. Exponer con ngrok (en terminal separada)

```powershell
ngrok http 8080 --domain=em-predictor.ngrok.io
```

> **Nota**: Sin dominio personalizado usar solo `ngrok http 8080`

---

## Verificación de Endpoints

| Servicio | Ruta Local | Descripción |
|----------|------------|-------------|
| Web App | http://localhost:8080 | Dashboard principal |
| API Docs | http://localhost:8080/docs | Swagger UI |
| Health | http://localhost:8080/health | Estado del sistema |
| Registro | http://localhost:8080/patients | Gestión pacientes |
| Inferencia | http://localhost:8080/api/v1/predict | Predicciones ML |
| Datos | http://localhost:8080/api/v1/ingest | Ingesta de datos |

---

## Modo Docker Completo

Para despliegue con todos los servicios dockerizados:

```powershell
# Levantar todo el stack
docker-compose --profile ngrok up -d

# Logs del servicio unificado
docker-compose logs -f unified_app
```

---

## Troubleshooting

- **ngrok no conecta**: Verificar `ngrok authtoken` configurado
- **Puerto ocupado**: `netstat -an | findstr :8080`
- **DB no disponible**: `docker-compose logs postgres`
- **Reiniciar todo**: `docker-compose down && docker-compose up -d`
