# EM-Predictor Roadmap

## Estado Actual: Fase 4 Completada

```
██████████████████████████░░ 90% completado
```

---

## Índice de Documentación

| Documento | Descripción |
|-----------|-------------|
| [README.md](../README.md) | Visión general y quickstart |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Stack técnico y decisiones |
| [QUICKSTART.md](QUICKSTART.md) | Guía de instalación |
| [ANALYSIS.md](ANALYSIS.md) | Análisis de métricas |
| [MINILLM_INTEGRATION_PLAN.md](MINILLM_INTEGRATION_PLAN.md) | Plan de integración NLP |

---

## Fases del Proyecto

### ✅ Fase 1: Preparación de Datos
**Estado: COMPLETADO**

- [x] Setup de infraestructura Docker (Postgres, Redis, MLflow)
- [x] Pipeline ETL para WhatsApp/Telegram
- [x] Extracción automática de eventos clínicos
- [x] Clustering temporal de señales
- [x] Generación de labels con horizontes 7/14/30 días

**Resultados:**
- 157,045 mensajes procesados
- 168+ días de datos
- 30 eventos clínicos confirmados

---

### ✅ Fase 2: Modelado Baseline
**Estado: COMPLETADO**

- [x] Feature engineering (lags, rolling, interactions)
- [x] Implementación de embeddings (Sentence Transformers)
- [x] Optimización con Optuna
- [x] Ensemble models (RF + GBM)
- [x] Walk-forward validation

**Resultados:**
- AUROC (holdout): 0.7026 ✅
- AUROC (walk-forward): 0.49 ± 0.21 ⚠️
- 88 features engineered

---

### ✅ Fase 3: Pipeline NLP
**Estado: COMPLETADO**

- [x] Microservicio `nlp-agent` con ONNX
- [x] Embeddings con MiniLM (384d)
- [x] Clasificación de síntomas multihead
- [x] Integración con feature extractor
- [x] 7,788 mensajes procesados con NLP

**Resultados:**
- Modelo: `st-all-MiniLM-L6-v2+onnx-heads-v1`
- Latencia: ~1.4s por mensaje
- Puerto: 8002

---

### ✅ Fase 4: Productización
**Estado: COMPLETADO**

- [x] API REST unificada (`unified_app` en puerto 8080)
- [x] Sistema de predicción con fallback heurístico
- [x] Frontend React + Vite (puerto 5173)
- [x] Sistema multi-tenant (Doctor/Paciente)
- [x] Impersonación segura para médicos
- [x] Scripts de automatización (`start_all.bat`)

**Servicios Activos:**
| Servicio | Puerto | Estado |
|----------|--------|--------|
| unified_app | 8080 | ✅ OK |
| nlp-agent | 8002 | ✅ OK |
| ml-inference | 8001 | ✅ OK |
| MLflow | 5000 | ✅ OK |
| PostgreSQL | 5432 | ✅ OK |
| Redis | 6379 | ✅ OK |

---

### 🔄 Fase 5: Piloto Clínico
**Estado: EN PROGRESO**

- [ ] Deploy en staging con ngrok
- [ ] Validación con equipo médico
- [ ] Ajustes basados en feedback
- [ ] Documentación clínica
- [x] Entrenamiento modelo TFT real

---

## Métricas de Éxito

| Métrica | Target | Actual | Estado |
|---------|--------|--------|--------|
| AUROC (14d, holdout) | > 0.65 | 0.7026 | ✅ |
| AUROC (walk-forward) | > 0.60 | 0.49 | ⚠️ |
| Mensajes procesados | > 10K | 157,045 | ✅ |
| NLP Embeddings | > 1K | 7,788 | ✅ |
| Predicción activa | Funcional | 43.9% | ✅ |
| Latencia predicción | < 500ms | ~200ms | ✅ |

---

## Changelog

### v0.5.0 (2026-02-07)
- Sistema multi-tenant Doctor/Paciente
- Impersonación segura para médicos
- Gestión de permisos bidireccional
- Scripts de automatización (`start_all.bat`, `stop_all.bat`)
- 6 servicios en producción local

### v0.4.0 (2026-02-06)
- Microservicio `nlp-agent` con ONNX
- Integración MLflow para tracking
- Frontend React completo
- API unificada en `unified_app`

### v0.3.0 (2024-12-09)
- Feature engineering con lags y rolling stats
- Optuna hyperparameter tuning
- Ensemble models
- AUROC 0.7026 alcanzado

### v0.2.0 (2024-12-08)
- Pipeline ETL completo
- Extracción de eventos clínicos
- Clustering temporal

### v0.1.0 (2024-12-01)
- Setup inicial del proyecto
- Infraestructura Docker
