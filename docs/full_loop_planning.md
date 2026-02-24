# Planificación: Loop Completo de MS-Predictor

Este documento define las tareas necesarias para conectar el frontend de gestión de etiquetas con el pipeline de ML real, cerrando confirmar el ciclo de mejora continua del modelo.

## 1. Persistencia de Datos
Actualmente el backend usa memoria (`dict`). Se necesita persistencia real para no perder etiquetas entre reinicios.

- [ ] **Migración a TimescaleDB/Postgres**:
    - Implementar conexión SQLAlchemy/AsyncPG en `unified_app`.
    - Migrar endpoints de `events.py` para usar queries SQL en lugar de dicts.
    - Asegurar que `paciente1_events.csv` se importa a la DB solo si está vacío.

## 2. Pipeline de Regeneración de Labels (ETL)
Cuando el usuario cambia settings o añade eventos, los labels en el dataset de entrenamiento deben actualizarse.

- [ ] **Script `regenerate_labels.py`**:
    - Input: `patient_id`.
    - Acción: Leer configuración de `label_settings` y eventos de `clinical_events`.
    - Lógica: Re-ejecutar `LabelGenerator` (existente en `scripts/etl/pipeline.py`) usando los nuevos parámetros.
    - Output: Actualizar `training_dataset_engineered.parquet` con nuevas columnas target.

## 3. Disparador de Re-entrenamiento (ML)
Conectar el botón "Reentrenar Ahora" con el proceso de entrenamiento.

- [ ] **Job Queue (Celery/Redis)**:
    - Implementar worker asíncrono en `unified_app` para no bloquear la API.
    - Tarea `train_model_task(patient_id)`:
        1. Ejecutar `regenerate_labels.py`.
        2. Ejecutar script de entrenamiento (actualmente `scripts/ml/ensemble_model.py` o similar).
        3. Guardar métricas y artefactos.

## 4. Visualización de Resultados
El usuario necesita ver si sus cambios mejoraron el modelo.

- [ ] **Almacenamiento de Métricas**:
    - Crear tabla `training_runs` (id, timestamp, status, auroc, auprc, validation_loss).
    - Guardar resultados al finalizar el entrenamiento.

- [ ] **Frontend: Resultados**:
    - Nuevo endpoint `GET /api/training/history`.
    - Actualizar `SettingsPage` o crear `ResultsPage` para mostrar:
        - Gráfico de mejora de AUROC/Loss a lo largo del tiempo.
        - Comparativa "Modelo Anterior" vs "Modelo Nuevo".
        - Log de entrenamiento en tiempo real (opcional, vía WebSocket o polling).

## 5. Inferencia Actualizada
El sistema de inferencia debe cargar automáticamente el mejor modelo nuevo.

- [ ] **Hot-swap de Modelos**:
    - El servicio de inferencia debe detectar nuevos modelos (watchdog o endpoint de notificación).
    - Cargar los nuevos pesos sin downtime.

## Estimación
- **Fase 1 (Persistencia + ETL Trigger)**: 1 sprint
- **Fase 2 (Training Worker + Resultados)**: 1-2 sprints
