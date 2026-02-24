# EM-Predictor

**Predicción de brotes de Esclerosis Múltiple usando análisis lingüístico y ML**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Objetivo

Predecir recaídas de Esclerosis Múltiple con **7-30 días de antelación** usando:
- Análisis lingüístico de comunicaciones (WhatsApp, Telegram)
- Patrones de actividad temporal
- Features clínicos y de comportamiento

---

## 📊 Estado Actual

| Métrica | Valor |
|---------|-------|
| **Mensajes procesados** | 48,010 |
| **NLP Embeddings** | 1,940 |
| **Eventos clínicos** | 30 |
| **AUROC (holdout)** | 0.7026 ✅ |
| **Predicción actual** | 43.9% (warning) |

---

## 🚀 Inicio Rápido

### Opción 1: Script Automatizado (Recomendado)

```bash
# Iniciar todo el sistema
start_all.bat

# Parar todos los servicios
stop_all.bat
```

### Opción 2: Manual

```bash
# 1. Iniciar containers Docker
docker start em_postgres em_redis mlflow

# 2. Iniciar servicios
cd services/nlp-agent && python main.py
cd services/ml-inference && python main.py
cd services/unified_app && python main.py --port 8080
cd services/webapp && npm run dev
```

---

## 🌐 Servicios

| Servicio | Puerto | URL | Descripción |
|----------|--------|-----|-------------|
| Frontend | 5173 | http://localhost:5173 | App React |
| Backend API | 8080 | http://localhost:8080/docs | FastAPI + Swagger |
| NLP Agent | 8002 | http://localhost:8002/health | Embeddings + Síntomas |
| ML Inference | 8001 | http://localhost:8001 | Modelo TFT |
| MLflow | 5000 | http://localhost:5000 | Tracking ML |

---

## 📚 Documentación

| Documento | Descripción |
|-----------|-------------|
| [docs/ROADMAP.md](docs/ROADMAP.md) | **Estado del proyecto**, fases y métricas |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Stack técnico y decisiones |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Guía de instalación paso a paso |
| [docs/ANALYSIS.md](docs/ANALYSIS.md) | Análisis de resultados ML |
| [docs/MINILLM_INTEGRATION_PLAN.md](docs/MINILLM_INTEGRATION_PLAN.md) | Plan de integración NLP |

### Workflows Disponibles

```bash
/etl-patient-data      # Procesar datos de paciente
/train-model           # Entrenar modelo TFT
/run-services          # Levantar servicios Docker
/deploy-ngrok          # Deploy con acceso remoto
```

---

## 📁 Estructura del Proyecto

```
PreSickness/
├── datos/                  # Datos crudos de pacientes
├── data/processed/         # Features procesados
├── services/
│   ├── unified_app/       # Backend API principal (8080)
│   ├── nlp-agent/         # Microservicio NLP (8002)
│   ├── ml-inference/      # Microservicio ML (8001)
│   └── webapp/            # Frontend React (5173)
├── scripts/
│   ├── etl/               # Pipeline de extracción
│   └── ml/                # Entrenamiento y evaluación
├── docs/                  # Documentación técnica
├── .agent/workflows/      # Workflows automatizados
├── start_all.bat          # Script de inicio
└── stop_all.bat           # Script de parada
```

---

## 🔧 Stack Tecnológico

- **ML**: scikit-learn, pytorch-forecasting, optuna
- **NLP**: sentence-transformers (MiniLM), ONNX
- **Data**: pandas, parquet, TimescaleDB
- **Backend**: FastAPI, pydantic, psycopg3
- **Frontend**: React, Vite, TanStack Query
- **Infra**: Docker Compose, Redis, MLflow

---

## 📜 Licencia

MIT License - Ver [LICENSE](LICENSE)
