# EM-Predictor Roadmap

## Estado Actual: Semana 6 de 16

```
██████████░░░░░░░░░░░░░ 37.5% completado
```

---

## Fases del Proyecto

### ✅ Fase 1: Preparación de Datos (Semanas 1-3)
**Estado: COMPLETADO**

- [x] Setup de infraestructura Docker
- [x] Pipeline ETL para WhatsApp/Telegram
- [x] Extracción automática de eventos clínicos
- [x] Clustering temporal de señales
- [x] Generación de labels con horizontes 7/14/30 días

**Resultados:**
- 50,392 mensajes procesados
- 168 días de datos
- 5 clusters de brote identificados

---

### ✅ Fase 2: Modelado Baseline (Semanas 4-6)
**Estado: COMPLETADO**

- [x] Feature engineering (lags, rolling, interactions)
- [x] Implementación de embeddings (Sentence Transformers)
- [x] Optimización con Optuna
- [x] Ensemble models (RF + GBM)
- [x] Walk-forward validation

**Resultados:**
- AUROC: 0.6851 (target: >0.65) ✅
- 88 features engineered
- Best model: GBM con parámetros optimizados

---

### 🔄 Fase 3: Modelo Temporal (Semanas 7-9)
**Estado: PENDIENTE**

- [ ] Integrar embeddings reales del paciente
- [ ] Entrenar Temporal Fusion Transformer (TFT)
- [ ] Fine-tuning por paciente
- [ ] Validación con segundo paciente

**Target:** AUROC > 0.70

---

### ⏳ Fase 4: Productización (Semanas 10-12)
**Estado: PENDIENTE**

- [ ] API REST para predicciones
- [ ] Sistema de alertas (email/SMS)
- [ ] Dashboard de monitoreo
- [ ] Tests E2E automatizados

---

### ⏳ Fase 5: Piloto Clínico (Semanas 13-16)
**Estado: PENDIENTE**

- [ ] Deploy en staging
- [ ] Validación con equipo médico
- [ ] Ajustes basados en feedback
- [ ] Documentación clínica

---

## Métricas de Éxito

| Métrica | Target | Actual | Estado |
|---------|--------|--------|--------|
| AUROC (14 días) | > 0.65 | 0.6851 | ✅ |
| AUROC (7 días) | > 0.60 | TBD | ⏳ |
| Latencia predicción | < 500ms | TBD | ⏳ |
| Falsos positivos | < 30% | TBD | ⏳ |

---

## Próximos Hitos

| Fecha | Hito |
|-------|------|
| Semana 7 | Modelo TFT entrenado |
| Semana 9 | Validación multi-paciente |
| Semana 12 | API en staging |
| Semana 16 | Piloto clínico completado |

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Dataset insuficiente | Alta | Alto | Aumentar ventana de datos, data augmentation |
| Overfitting temporal | Media | Alto | Walk-forward CV, regularización |
| Latencia en producción | Baja | Medio | Caching, modelo ligero |
| Compliance GDPR | Media | Alto | Encriptación, anonimización |

---

## Changelog

### v0.3.0 (2024-12-09)
- Feature engineering con lags y rolling stats
- Optuna hyperparameter tuning
- Ensemble models
- AUROC 0.6851 alcanzado

### v0.2.0 (2024-12-08)
- Pipeline ETL completo
- Extracción de eventos clínicos
- Clustering temporal
- Baseline RF con AUROC 0.64

### v0.1.0 (2024-12-01)
- Setup inicial del proyecto
- Infraestructura Docker
- Documentación base
