# Mapa de Sistema EM-Predictor
> Generado: 2026-02-07

## 📊 Resumen de Análisis

### Servicios (6/6 OK)
| Servicio | Puerto | Estado |
|----------|--------|--------|
| unified_app | 8080 | ✅ OK |
| nlp-agent | 8002 | ✅ OK |
| ml-inference | 8001 | ✅ OK |
| mlflow | 5000 | ✅ OK |
| postgres | 5432 | ✅ OK |
| redis | 6379 | ✅ OK |

### Endpoints (32 total)
| Dominio | Cantidad |
|---------|----------|
| events | 14 |
| auth | 3 |
| patients | 3 |
| doctor | 3 |
| patient | 2 |
| predict | 1 |
| alerts | 1 |
| metrics | 1 |
| health | 1 |

---

## 🔄 Mapa de Flujos de Datos

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    FRONTEND                                         │
│                               (React + Vite :5173)                                  │
└───────────────────────────────────────┬─────────────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                              unified_app (:8080)                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │
│  │    AUTH     │  │   EVENTS    │  │  PATIENTS   │  │   PREDICT   │                   │
│  │  /register  │  │  /events/   │  │ /patients/* │  │  /predict   │                   │
│  │  /login     │  │  /clusters  │  │ /doctor/*   │  │             │                   │
│  │  /google    │  │  /settings  │  │ /patient/*  │  │             │                   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                   │
└─────────┼────────────────┼────────────────┼────────────────┼──────────────────────────┘
          │                │                │                │
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PostgreSQL/TimescaleDB (:5432)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  ┌───────────────────────┐   │
│  │   users     │  │   events    │  │    messages     │  │ doctor_patient_access │   │
│  │ ─────────── │  │ ─────────── │  │ ─────────────── │  │ ─────────────────────  │   │
│  │ id          │  │ id          │  │ id              │  │ doctor_id             │   │
│  │ email       │  │ user_id     │  │ user_id         │  │ patient_id            │   │
│  │ name        │  │ event_type  │  │ content         │  │ status                │   │
│  │ role        │  │ severity    │  │ nlp_embedding   │  │ granted_at            │   │
│  │ password    │  │ event_date  │  │ nlp_symptoms    │  └───────────────────────┘   │
│  └─────────────┘  └─────────────┘  └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
          │
          │  (NLP Processing)
          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              nlp-agent (:8002)                                      │
│  ┌───────────────────────────────────────────────────────────────┐                  │
│  │                     /v1/process                               │                  │
│  │  Input: {message_id, user_id, text}                           │                  │
│  │  Output: {embeddings[384], symptom_scores, linguistic_meta}   │                  │
│  │                                                               │                  │
│  │  Model: all-MiniLM-L6-v2 (ONNX)                               │                  │
│  │  Heads: fatigue, pain, cognitive, mood, sleep                 │                  │
│  └───────────────────────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
          │
          │  (Prediction Request)
          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              ml-inference (:8001)                                   │
│  ┌───────────────────────────────────────────────────────────────┐                  │
│  │                       /predict                                │                  │
│  │  Input: {user_id_hash, window_days, horizon_days}             │                  │
│  │  Output: {probability, risk_level, model_uri}                 │                  │
│  │                                                               │                  │
│  │  Model: TFT via MLflow (⚠️ FALLBACK: Heuristic)               │                  │
│  └───────────────────────────────────────────────────────────────┘                  │
│                               │                                                     │
└───────────────────────────────┼─────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              MLflow (:5000)                                         │
│  ┌─────────────────────────────────────────────────────────┐                        │
│  │  Experiments     │  Model Registry  │  Artifacts        │                        │
│  │  ─────────────   │  ─────────────── │  ─────────────    │                        │
│  │  (empty)         │  ⚠️ No TFT model  │  (empty)          │                        │
│  └─────────────────────────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

```

---

## ⚠️ Issues Detectados

### 1. TFT Model Not Registered (WARNING)
- **Componente**: ml-inference → MLflow
- **Estado**: Usando heurística de fallback
- **Impacto**: Predicciones basadas en reglas, no en modelo entrenado
- **Fix**: Entrenar TFT y registrar en MLflow con `mlflow.pytorch.log_model()`

### 2. NLP Processing Ratio Bajo (INFO)
- **Componente**: nlp-pipeline
- **Estado**: Solo 4% de mensajes procesados (1,940 / 48,010)
- **Impacto**: Features de NLP limitados para entrenamiento
- **Fix**: Ejecutar procesamiento bulk:
  ```bash
  python scripts/etl/process_nlp_bulk.py
  ```

### 3. No Clusters Detectados (INFO)
- **Componente**: clustering
- **Estado**: 0 clusters en tabla
- **Impacto**: No hay agrupación de señales para labels
- **Fix**: Ejecutar clustering:
  ```bash
  python scripts/etl/cluster_signals.py
  ```

---

## ✅ Flujos Verificados OK

| Flow | Estado |
|------|--------|
| Registro de Usuario | ✅ |
| Login → JWT Token | ✅ |
| Upload WhatsApp → Parse → DB | ✅ |
| NLP Processing (individual) | ✅ |
| Predicción (con fallback) | ✅ |
| Doctor → Añadir Paciente | ✅ |
| Doctor → Impersonación | ✅ |
| Paciente → Ver Doctores | ✅ |
| Paciente → Revocar Acceso | ✅ |
| Creación de Eventos | ✅ |
| Edición de Eventos | ✅ |
| Eliminación de Eventos | ✅ |

---

## 📌 Pendientes Principales

1. **Entrenar y registrar modelo TFT** en MLflow
2. **Procesar NLP bulk** para más mensajes
3. **Ejecutar clustering** sobre eventos
4. **Deploy con ngrok** para acceso externo
