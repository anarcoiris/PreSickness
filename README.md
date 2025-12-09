# EM-Predictor

**Predicción de brotes de Esclerosis Múltiple usando análisis lingüístico y ML**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Objetivo

Predecir recaídas de Esclerosis Múltiple con **7-30 días de antelación** usando:
- Análisis lingüístico de comunicaciones (WhatsApp, Telegram)
- Patrones de actividad temporal
- Features clínicos y de comportamiento

**Métrica objetivo**: AUROC > 0.65 | **Resultado actual**: AUROC 0.6851 ✅

---

## 📚 Documentación

| Documento | Descripción |
|-----------|-------------|
| [ROADMAP.md](docs/ROADMAP.md) | Estado del proyecto, fases y próximos pasos |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Decisiones técnicas, stack y diseño |
| [QUICKSTART.md](docs/QUICKSTART.md) | Guía rápida de instalación y uso |
| [PLATFORM.md](docs/PLATFORM.md) | **Nuevo**: Plataforma Web, CLI y Backend Unificado |
| [LEGAL.md](LEGAL/LEGAL.md) | GDPR, HIPAA, consentimiento informado |

### Workflows Disponibles

```bash
/etl-patient-data      # Procesar datos de paciente
/train-model           # Entrenar modelo TFT
/run-services          # Levantar servicios Docker
/research-predictive-medicine  # Investigar ML en salud
/research-gpts-nlp     # Investigar modelos de lenguaje
```

---

## 🚀 Quickstart

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
pip install sentence-transformers optuna
```

### 2. Procesar datos de paciente
```bash
# Extraer eventos clínicos
python scripts/etl/extract_events.py datos/paciente1_whatsapp.txt --output datos/events.csv

# Ejecutar ETL completo
python -m scripts.etl.pipeline --input datos/paciente1_whatsapp.txt --events datos/events.csv --patient-id paciente1 --output data/processed/

# Regenerar labels con clusters
python scripts/etl/regenerate_labels.py --data-path data/processed/paciente1 --clusters datos/paciente1_events_auto_clusters.csv
```

### 3. Entrenar modelo
```bash
# Pipeline completo (feature eng + embeddings + ensemble)
python scripts/ml/run_full_pipeline.py --data-path data/processed/paciente1

# O paso a paso:
python scripts/ml/feature_engineering.py --data-path data/processed/paciente1
python scripts/ml/optuna_simple.py --data-path data/processed/paciente1 --n-trials 30
python scripts/ml/ensemble_model.py --data-path data/processed/paciente1
```

### 4. Visualizar resultados
```bash
python scripts/ml/plot_features_timeseries.py --data-path data/processed/paciente1
python scripts/ml/walk_forward_validation.py --data-path data/processed/paciente1
```

---

## 📊 Resultados Actuales (paciente1)

| Modelo | AUROC | AUPRC |
|--------|-------|-------|
| GBM (Optuna) | **0.6851** | 0.3557 |
| RF (Optuna) | 0.6791 | 0.3506 |
| RF+GBM Average | 0.6611 | 0.3485 |

---

## 📁 Estructura del Proyecto

```
em-predictor/
├── datos/                  # Datos crudos de pacientes
├── data/processed/         # Features procesados
├── scripts/
│   ├── etl/               # Pipeline de extracción
│   │   ├── pipeline.py
│   │   ├── extract_events.py
│   │   ├── cluster_signals.py
│   │   └── embeddings.py
│   └── ml/                # Entrenamiento y evaluación
│       ├── run_full_pipeline.py
│       ├── feature_engineering.py
│       ├── ensemble_model.py
│       └── optuna_simple.py
├── services/              # Microservicios (Docker)
├── docs/                  # Documentación técnica
├── .agent/workflows/      # Workflows automatizados
└── tinyllm/              # Modelo de lenguaje local
```

---

## 🔧 Stack Tecnológico

- **ML**: scikit-learn, pytorch-forecasting, optuna
- **NLP**: sentence-transformers, spacy
- **Data**: pandas, parquet, TimescaleDB
- **Infra**: Docker Compose, FastAPI, Redis
- **MLOps**: MLflow, Prometheus, Grafana

---

## 📜 Licencia

MIT License - Ver [LICENSE](LICENSE)
