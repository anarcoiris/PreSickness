# Plan de Implementación: Pipeline Predictivo PreSickness

## 1. Visión General
Este documento detalla las fases y pasos prácticos para implementar el pipeline de Machine Learning sobre los datos de mensajería de los usuarios en PreSickness, garantizando una evolución segura y medible.

## 2. Fases de Implementación

### Fase 1: Exploración y Baseline (Semanas 1-3)
**Objetivo:** Crear un MVP (Minimum Viable Product) funcional para validar la ingesta y extracción básica.
1. **Script de Preprocesado asíncrono:** Desarrollar proceso que lea `unprocessed_messages`, elimine PII y extraiga conteos básicos (`word_count`, keywords, `nlp_level=1`).
2. **Actualización de Datapoints:** Insertar/Actualizar la tabla `datapoints` con los `numeric_features` calculados para ventanas temporales (ej. diario).
3. **Modelado Baseline:** Entrenar un modelo de suavizado exponencial (EWMA) combinado con un XGBoost básico, buscando predecir eventos en un horizonte de 7 días.
4. **Métricas a evaluar:** Recall general, Lead time preliminar.

### Fase 2: Features Avanzados y A/B Testing (Semanas 4-5)
**Objetivo:** Enriquecer las señales y medir el valor añadido de la semántica compleja.
1. **Integración de NLP:** Incorporar `sentence-transformers` en la ingesta para agregar `embedding_encrypted` en los `datapoints` con `nlp_level=2+`.
2. **Features Temporales y Comportamentales:** Medir tiempos entre mensajes, ratio de typos, y anomalías en frecuencia base de la persona.
3. **Experimentos A/B:** Comparar el modelo Baseline (Fase 1) vs. el nuevo modelo usando métricas de *Delta Recall* y *Delta Lead Time*.

### Fase 3: Validación de Robustez y "Holdout" (Semanas 6-7)
**Objetivo:** Evitar el *data leakage* (fuga de información temporal).
1. **Backtesting Temporal Estricto:** Dividir los datos donde el conjunto de train sea cronológicamente estricto al de test. Usar cortes `-3` o `-7` días del `event_date` clínico.
2. **Pruebas de Generalización:** Realizar *Cross-region holdout* (entrenar en región/grupo A, validar en región/grupo B).
3. **Calibración de Alertas:** Determinar umbrales probabilísticos para mantener la métrica *False alarms/week* por debajo de un nivel clínico aceptable (ej. ≤1 por semana/región).

### Fase 4: Producción y Feedback Loop
1. **Generación de Alertas:** Conectar inferencias diarias con la tabla `alerts`.
2. **Rechazo y Validación:** Habilitar interfaz en frontend para que el doctor marque alertas como falsos positivos. Retroalimentación hacia pesos del modelo (Human-in-the-loop).
