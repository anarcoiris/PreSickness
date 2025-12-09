---
description: Procesar datos de paciente desde WhatsApp hasta features para ML
---

# Workflow: ETL de Datos de Paciente

Este workflow procesa un export de WhatsApp y lo prepara para entrenamiento ML.

## Requisitos Previos
- Python 3.11+ con venv activado
- Archivo WhatsApp en `datos/pacienteX_whatsapp.txt`
- Archivo de eventos clínicos en `datos/pacienteX.json`

## Pasos

// turbo
### 0. Extraer eventos clínicos automáticamente (NUEVO)

```powershell
cd c:\Users\aladin\Documents\Presickness
python scripts/etl/extract_events.py datos/paciente1_whatsapp.txt --output datos/paciente1_events_auto.csv --detailed-report
```

Esto detecta automáticamente:
- `relapse`: brotes, recaídas, crisis
- `fatigue`: cansancio, agotamiento
- `pain`: dolor, hormigueo, punzadas
- `cognitive`: niebla mental, olvidos
- `vision`: visión borrosa
- `mobility`: dificultad para caminar
- `hospitalization`: menciones de hospital
- `medication`: tratamientos, medicamentos

Revisar el reporte JSON generado para validar los eventos detectados.

### 1. Crear archivo de metadatos del paciente

Crear `datos/pacienteX.json` con la siguiente estructura:

```json
{
  "patient_id": "paciente1",
  "patient_marker": "<?>",
  "events": [
    {"date": "2024-12-21", "event_type": "relapse", "severity": "moderate", "notes": "brote mencionado en chat"}
  ],
  "data_start": "2024-07-19",
  "data_end": "2025-xx-xx",
  "notes": "Paciente con EM, datos de chat con contraparte"
}
```

### 2. Limpiar mensajes de sistema del WhatsApp

El pipeline ETL filtra automáticamente:
- `<Multimedia omitido>`
- Mensajes de cifrado extremo a extremo
- `Se editó este mensaje`
- `Eliminaste este mensaje`

// turbo
### 3. Ejecutar ETL pipeline

```powershell
cd c:\Users\aladin\Documents\Presickness
python -m scripts.etl.pipeline --input datos/paciente1_whatsapp.txt --events datos/paciente1_events.csv --patient-id paciente1 --output data/processed/ --target-sender "<?"
```

### 4. Verificar outputs

Los outputs estarán en `data/processed/paciente1/`:
- `daily_features.parquet` - Features agregados por día
- `window_features.parquet` - Features por ventana temporal (1,3,7,14,30 días)
- `labels.parquet` - Labels de brotes por horizonte
- `training_dataset.parquet` - Dataset combinado para ML
- `metadata.json` - Estadísticas y configuración

// turbo
### 5. Inspeccionar dataset

```powershell
python -c "import pandas as pd; df = pd.read_parquet('data/processed/paciente1/training_dataset.parquet'); print(f'Shape: {df.shape}'); print(df.head())"
```

## Notas

- El paciente target está marcado como `<?>` en el export de WhatsApp
- Los mensajes de contrapartes también se procesan pero con peso diferente
- Asegurar que el archivo de eventos contiene al menos 1 brote documentado
