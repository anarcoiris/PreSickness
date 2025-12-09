---
description: Entrenar modelo TFT con datos procesados de paciente
---

# Workflow: Entrenamiento de Modelo

Este workflow entrena un modelo TFT usando datos procesados del pipeline ETL.

## Requisitos Previos
- ETL completado con outputs en `data/processed/pacienteX/`
- MLflow running (o usar tracking local)
- GPU recomendada pero no requerida

## Pasos

### 1. Verificar que MLflow está corriendo

```powershell
docker-compose up -d mlflow
# Verificar en http://localhost:5000
```

// turbo
### 2. Ejecutar entrenamiento TFT

```powershell
cd c:\Users\aladin\Documents\Presickness
python train_tft.py --data-path data/processed/paciente1/training_dataset.parquet --experiment-name "paciente1_baseline"
```

### 3. Evaluar métricas

Revisar en MLflow UI (http://localhost:5000):
- AUROC (target: >0.65 en validación)
- AUPRC
- Calibration Brier score (<0.25)

### 4. Exportar modelo

```powershell
# El mejor modelo se guarda automáticamente en mlruns/ o en MinIO
# Para exportar a TorchScript:
python -c "import torch; from train_tft import load_model; model = load_model('latest'); torch.jit.save(torch.jit.script(model), 'models/tft_paciente1.pt')"
```

## Siguientes Pasos

- Fine-tuning específico del paciente
- Comparar con baseline (LSTM, Prophet)
- Validar con holdout temporal
