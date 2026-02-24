# Índice de Documentación

> Última actualización: 2026-02-07

## 📑 Documentos Principales

| # | Documento | Descripción | Estado |
|---|-----------|-------------|--------|
| 1 | [README.md](../README.md) | Visión general, quickstart, estructura | ✅ Actualizado |
| 2 | [ROADMAP.md](ROADMAP.md) | Fases, métricas, changelog | ✅ Actualizado |
| 3 | [ARCHITECTURE.md](ARCHITECTURE.md) | Microservicios, ADRs, stack | ✅ Actualizado |
| 4 | [QUICKSTART.md](QUICKSTART.md) | Instalación, scripts, troubleshooting | ✅ Actualizado |

## 📊 Análisis y Planes

| Documento | Descripción |
|-----------|-------------|
| [ANALYSIS.md](ANALYSIS.md) | Análisis crítico de métricas ML |
| [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md) | ADRs detallados |
| [MINILLM_INTEGRATION_PLAN.md](MINILLM_INTEGRATION_PLAN.md) | Plan de integración NLP |
| [PLATFORM.md](PLATFORM.md) | Plataforma Web y CLI |
| [full_loop_planning.md](full_loop_planning.md) | Planificación del loop completo |

## 📄 Reportes

| Documento | Descripción |
|-----------|-------------|
| [em_predictor_technical_report.pdf](em_predictor_technical_report.pdf) | Reporte técnico LaTeX |
| [nlp_critique.md](nlp_critique.md) | Crítica del pipeline NLP |

## 🔗 Referencias Cruzadas

```
README.md ─────────────────────────────────────────────────────────────┐
    │                                                                  │
    ├─→ docs/ROADMAP.md (estado actual, fases)                         │
    │       └─→ docs/ANALYSIS.md (métricas detalladas)                 │
    │                                                                  │
    ├─→ docs/ARCHITECTURE.md (diseño técnico)                          │
    │       └─→ docs/ARCHITECTURE_DECISIONS.md (ADRs)                  │
    │       └─→ services/unified_app/ (backend)                        │
    │       └─→ services/nlp-agent/ (NLP)                              │
    │       └─→ services/ml-inference/ (ML)                            │
    │                                                                  │
    ├─→ docs/QUICKSTART.md (instalación)                               │
    │       └─→ start_all.bat (script maestro)                         │
    │       └─→ stop_all.bat (script de parada)                        │
    │                                                                  │
    └─→ .agent/workflows/ (automatizaciones)                           │
            └─→ etl-patient-data.md                                    │
            └─→ train-model.md                                         │
            └─→ run-services.md                                        │
            └─→ deploy-ngrok.md                                        │
───────────────────────────────────────────────────────────────────────┘
```

## 📦 Servicios y Puertos

| Servicio | Puerto | Documentación |
|----------|--------|---------------|
| Frontend (webapp) | 5173 | [services/webapp/](../services/webapp/) |
| Backend (unified_app) | 8080 | [services/unified_app/](../services/unified_app/) |
| NLP Agent | 8002 | [services/nlp-agent/](../services/nlp-agent/) |
| ML Inference | 8001 | [services/ml-inference/](../services/ml-inference/) |
| MLflow | 5000 | Docker container |
| PostgreSQL | 5432 | Docker container |
| Redis | 6379 | Docker container |
