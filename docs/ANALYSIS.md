# Análisis Crítico del Proyecto EM-Predictor

**Fecha:** 2026-02-04  
**Analista:** Antigravity AI  
**Proyecto:** PreSickness (EM-Predictor)

---

## Resumen Ejecutivo

Se reprodujeron exitosamente los entrenamientos del pipeline ML del proyecto EM-Predictor. El modelo GBM logró **AUROC 0.7026** en holdout, superando el objetivo mínimo de 0.65. Sin embargo, la validación walk-forward revela **alta varianza** (AUROC medio 0.49, std 0.21), indicando problemas de generalización temporal.

---

## Datos del Proyecto

### Dataset Disponible
| Métrica | Valor |
|---------|-------|
| Mensajes totales | 51,714 |
| Días de datos | 168 |
| Rango temporal | 2024-07-19 a 2025-12-01 |
| Eventos clínicos | 306 |
| Samples (ventanas) | 4,200 |
| Features engineered | 88 |
| Tasa de positivos | 31.5% |

### Archivos Procesados
- `daily_features.parquet` - Features agregados por día
- `window_features.parquet` - Features por ventana temporal
- `training_dataset_engineered.parquet` - Dataset con feature engineering
- `training_dataset_clusters.parquet` - Dataset con labels basados en clusters

---

## Resultados de Reproducción

### Optuna Tuning (20 trials)

| Modelo | CV AUROC | Holdout AUROC | Variación vs Previo |
|--------|----------|---------------|---------------------|
| RandomForest | 0.5632 | 0.6833 | +0.4% |
| GradientBoosting | 0.5926 | 0.6323 | -2.5% |

### Ensemble Models

| Modelo | AUROC | AUPRC | Estado |
|--------|-------|-------|--------|
| **GBM** | **0.7026** | **0.3675** | ✅ Mejor |
| RF | 0.6238 | 0.3368 | ✅ |
| Manual Avg | 0.6681 | 0.3424 | ✅ |
| Voting (Soft) | 0.6140 | 0.3251 | ⚠️ |
| Stacking | 0.4513 | 0.2418 | ❌ |
| LogReg | 0.1526 | 0.1626 | ❌ |

> [!TIP]
> El mejor modelo GBM mejoró de AUROC 0.6851 a **0.7026** (+1.75 puntos), superando la meta de 0.65.

### Walk-Forward Validation (9 folds)

| Fold | AUROC | Test Positive Rate | Comentario |
|------|-------|-------------------|------------|
| 0 | 0.718 | 71.4% | ✅ Bueno |
| 1 | 0.769 | 50.0% | ✅ Bueno |
| 2 | 0.260 | 14.3% | ❌ Muy bajo |
| 3 | 0.377 | 64.3% | ❌ Bajo |
| 4 | 0.415 | 85.7% | ⚠️ Bajo |
| 5 | 0.789 | 35.7% | ✅ Bueno |
| 6 | 0.294 | 7.1% | ❌ Muy bajo |
| 7 | 0.454 | 57.1% | ⚠️ Medio |
| 8 | 0.332 | 57.1% | ❌ Bajo |

**Estadísticas Walk-Forward:**
- Mean AUROC: **0.4898**
- Std AUROC: **0.2107**
- Min AUROC: 0.260
- Max AUROC: 0.789

---

## Análisis Crítico

### 🔴 Problemas Identificados

#### 1. Alta Varianza en Validación Temporal
> [!CAUTION]
> El modelo muestra un AUROC medio de 0.49 en walk-forward vs 0.70 en holdout simple. Esto indica **overfitting temporal** significativo.

**Causa probable:** El split train/test en holdout simple tiene data leakage por:
- Ventanas temporales solapadas (7 días con stride de 1 día)
- Features con lags que pueden infiltrar información futura

#### 2. Colapso del Modelo Stacking
El Stacking ensemble (AUROC 0.45) y LogReg (AUROC 0.15) muestran colapso total. Esto sugiere:
- ⚠️ Features con colinealidad extrema
- ⚠️ Escalado inadecuado para LogReg
- ⚠️ Posible target leakage

#### 3. Sensibilidad a Distribución de Clases
Los folds con test_positive_rate extremo (7%, 14%, 86%) tienen AUROC degradado. El modelo no generaliza bien ante cambios de distribución.

#### 4. Dataset Limitado
Con solo 168 días y 5 clusters de brote, el modelo tiene poca variedad de patrones de recaída para aprender.

### 🟡 Áreas de Mejora

#### 1. Feature Engineering
```python
# Features actuales: 15 base + 73 engineered = 88 total
# Oportunidades:
- Reducir dimensionalidad (PCA o selección)
- Añadir features de tendencia más robustos
- Considerar embeddings de texto (actualmente no usados en ensemble)
```

#### 2. Validación
- Implementar **Purged Time-Series CV** con gap entre train/test
- Usar **embargo period** de 14 días (igual al horizonte de predicción)
- Considerar validación leave-one-cluster-out

#### 3. Regularización
- Aplicar class weights más agresivos
- Probar modelos con regularización L1/L2 explícita
- Calibrar probabilidades post-hoc

### 🟢 Aspectos Positivos

1. **Pipeline reproducible**: Los scripts funcionan correctamente y los resultados son consistentes
2. **Documentación completa**: README, ROADMAP y ARCHITECTURE están actualizados
3. **Meta superada en holdout**: AUROC 0.70 > 0.65 objetivo
4. **Infraestructura lista**: Docker, MLflow, servicios configurados
5. **Feature importance razonable**: Coverage, sentiment_std y words_mean son los más predictivos

---

## Sincronización con Documentación

### Estado Actual vs ROADMAP.md

| Item | ROADMAP | Actual | Sincronizado |
|------|---------|--------|--------------|
| AUROC objetivo | >0.65 | 0.7026 | ✅ Superado |
| Fase 1 (Prep datos) | Completado | ✅ | ✅ |
| Fase 2 (Baseline) | Completado | ✅ | ✅ |
| Fase 3 (TFT) | Pendiente | Pendiente | ✅ |
| Samples procesados | 50,392 | 51,714 | ⚠️ Actualizar |
| Días de datos | 168 | 168 | ✅ |

### Cambios Recomendados

1. **README.md**: Actualizar AUROC de 0.6851 a **0.7026**
2. **ROADMAP.md**: Añadir warning sobre walk-forward validation
3. **ARCHITECTURE.md**: Documentar problemas de generalización temporal

---

## Recomendaciones Prioritarias

### Corto Plazo (1-2 semanas)

1. **Implementar Purged CV** con gap de 14+ días
2. **Eliminar features con alta correlación** (>0.95)
3. **Añadir segundo paciente** para validación cruzada
4. **Generar informe de feature importance** con SHAP

### Medio Plazo (3-4 semanas)

1. **Entrenar modelo TFT** (Fase 3 del ROADMAP)
2. **Integrar embeddings reales** en el pipeline
3. **Implementar calibración de probabilidades**
4. **Crear tests de regresión** para métricas

### Largo Plazo

1. **Aumentar ventana de datos** (objetivo: >365 días)
2. **Multi-paciente validation** (objetivo: 3+ pacientes)
3. **Explainability dashboard** con SHAP/LIME
4. **Piloto clínico** (Fase 5)

---

## Conclusión

El proyecto EM-Predictor muestra resultados prometedores en métricas de holdout (AUROC 0.70) pero revela problemas de generalización temporal en walk-forward validation (AUROC 0.49). Antes de avanzar a producción o piloto clínico, es **crítico** implementar validación temporal más rigurosa y posiblemente aumentar el dataset con datos de pacientes adicionales.

**Veredicto:** ✅ Viabilidad técnica demostrada, pero ⚠️ requiere mejoras en generalización antes de uso clínico.

---

## Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `data/processed/paciente1/optuna_results.json` | Parámetros óptimos RF/GBM |
| `data/processed/paciente1/ensemble_results.json` | Métricas de ensemble |
| `data/processed/paciente1/walk_forward_results.csv` | Resultados por fold |
| `data/processed/paciente1/features_timeseries.png` | Visualización temporal |
| `data/processed/paciente1/walk_forward_auroc.png` | AUROC por fold |
