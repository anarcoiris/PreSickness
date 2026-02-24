# Arquitectura EM-Predictor

> **Referencia**: [ROADMAP.md](ROADMAP.md) | [QUICKSTART.md](QUICKSTART.md)

## Visión General

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTA DE DATOS                                  │
├──────────────┬──────────────┬──────────────┬───────────────┬───────────────┤
│   WhatsApp   │   Telegram   │     CSV      │   API REST    │   WhatsApp    │
│   (export)   │   (export)   │   (manual)   │   (upload)    │   (live)      │
└──────┬───────┴──────┬───────┴──────┬───────┴───────┬───────┴───────┬───────┘
       │              │              │               │               │
       └──────────────┴──────────────┴───────────────┴───────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         unified_app (8080)                                  │
│  FastAPI Backend │ Auth │ Upload │ Events │ Predictions │ Doctor/Patient   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│   nlp-agent (8002)  │ │ ml-inference    │ │   TimescaleDB       │
│   ─────────────────  │ │    (8001)       │ │     (5432)          │
│   • MiniLM ONNX     │ │ ───────────────  │ │ ─────────────────── │
│   • Symptom Heads   │ │ • TFT Model     │ │ • users             │
│   • Embeddings 384d │ │ • Heuristic FB  │ │ • events            │
│   • Linguistic Meta │ │ • MLflow Client │ │ • messages          │
└─────────────────────┘ └────────┬────────┘ │ • predictions       │
                                 │          └─────────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │  MLflow (5000)  │
                        │ ───────────────  │
                        │ • Experiments   │
                        │ • Model Registry│
                        │ • Artifacts     │
                        └─────────────────┘
```

---

## Microservicios

### unified_app (Puerto 8080)
**Backend API Principal**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/auth/register` | POST | Registro usuario |
| `/api/auth/login` | POST | Login JWT |
| `/api/patients/me` | GET | Perfil usuario |
| `/api/patients/upload` | POST | Subir WhatsApp |
| `/api/events/` | GET | Lista eventos |
| `/api/predict` | POST | Predicción ML |
| `/api/doctor/patients` | GET/POST | Gestión pacientes |
| `/api/patient/doctors` | GET/DELETE | Gestión permisos |

**Archivos clave:**
- `services/unified_app/main.py` - FastAPI app
- `services/unified_app/db.py` - Capa de datos async
- `services/unified_app/events.py` - Parser WhatsApp

---

### nlp-agent (Puerto 8002)
**Servicio de Procesamiento NLP**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Estado del servicio |
| `/v1/process` | POST | Procesar mensaje |

**Stack:**
- Modelo: `sentence-transformers/all-MiniLM-L6-v2`
- Inference: ONNX Runtime
- Heads: Clasificación multihead de síntomas

**Archivos clave:**
- `services/nlp-agent/main.py` - FastAPI endpoints
- `services/nlp-agent/model.py` - Motor ONNX
- `services/nlp-agent/heads_v1.onnx` - Modelo entrenado

---

### ml-inference (Puerto 8001)
**Servicio de Predicción ML**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/predict` | POST | Predicción TFT |

**Stack:**
- Modelo: TFT (Temporal Fusion Transformer)
- Fallback: Heurística basada en eventos recientes
- Registry: MLflow

**Archivos clave:**
- `services/ml-inference/main.py` - FastAPI + MLflow client

---

## Stack Tecnológico

### Core ML
| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| Baseline | scikit-learn | Modelos rápidos, probados |
| Temporal | pytorch-forecasting | TFT para series temporales |
| HPO | Optuna | Pruning eficiente |
| Embeddings | sentence-transformers | Multilingüe, local |
| Inference | ONNX Runtime | Latencia optimizada |

### Data
| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| Storage | Parquet | Columnar, compresión |
| DB | TimescaleDB | Time-series PostgreSQL |
| Cache | Redis | Feature store en memoria |

### Infraestructura
| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| Containers | Docker Compose | Dev local simple |
| API | FastAPI | Async, autodocs |
| MLOps | MLflow | Tracking experimentos |
| Frontend | React + Vite | SPA moderna |

---

## Decisiones Arquitectónicas (ADR)

### ADR-001: Modelos Locales vs Cloud
**Decisión**: Priorizar modelos locales (MiniLM, ONNX)

**Razón**: Datos de salud sensibles, GDPR compliance

**Consecuencias**:
- ✅ Control total de datos
- ✅ Sin costes de API
- ⚠️ Menor capacidad que GPT-4

---

### ADR-002: Microservicios vs Monolito
**Decisión**: Arquitectura de microservicios ligeros

**Razón**: Escalabilidad independiente, despliegue parcial

**Consecuencias**:
- ✅ NLP puede escalar separado de ML
- ✅ Fallback si un servicio falla
- ⚠️ Complejidad de red

---

### ADR-003: Multi-Tenant con Impersonación
**Decisión**: Header `X-Patient-ID` para delegación

**Razón**: Médico necesita ver datos de paciente sin duplicar endpoints

**Consecuencias**:
- ✅ Endpoints reutilizados
- ✅ Auditoría clara
- ⚠️ Requiere validación de permisos

---

## Seguridad

### Autenticación
- JWT con expiración 24h
- Hash de contraseñas con bcrypt
- Roles: `patient`, `doctor`

### Autorización
- Middleware de verificación de permisos
- Tabla `doctor_patient_access`
- Revocación bilateral

### Datos
- Encriptación AES-256 en reposo
- HTTPS obligatorio
- PII pseudonimizado
