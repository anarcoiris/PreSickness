# Guía de Inicio Rápido

## Requisitos

- Python 3.10+
- Docker Desktop (opcional, para servicios)
- 8GB RAM mínimo

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/your-org/em-predictor.git
cd em-predictor

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
pip install sentence-transformers optuna
```

## Procesar Primer Paciente

### 1. Preparar datos
Coloca el export de WhatsApp en `datos/`:
```
datos/paciente1_whatsapp.txt
```

### 2. Extraer eventos clínicos
```bash
python scripts/etl/extract_events.py datos/paciente1_whatsapp.txt --output datos/paciente1_events.csv
```

### 3. Ejecutar ETL
```bash
python -m scripts.etl.pipeline \
  --input datos/paciente1_whatsapp.txt \
  --events datos/paciente1_events.csv \
  --patient-id paciente1 \
  --output data/processed/ \
  --target-sender "<?>"
```

### 4. Generar clusters de brote
```bash
python scripts/etl/cluster_signals.py \
  --input datos/paciente1_events.csv \
  --output datos/paciente1_clusters.csv
```

### 5. Regenerar labels
```bash
python scripts/etl/regenerate_labels.py \
  --data-path data/processed/paciente1 \
  --clusters datos/paciente1_clusters.csv
```

## Entrenar Modelo

### Opción A: Pipeline completo
```bash
python scripts/ml/run_full_pipeline.py --data-path data/processed/paciente1
```

### Opción B: Paso a paso
```bash
# Feature engineering
python scripts/ml/feature_engineering.py --data-path data/processed/paciente1

# Optimización Optuna
python scripts/ml/optuna_simple.py --data-path data/processed/paciente1 --n-trials 30

# Ensemble
python scripts/ml/ensemble_model.py --data-path data/processed/paciente1
```

## Visualizar Resultados

```bash
# Evolution temporal
python scripts/ml/plot_features_timeseries.py --data-path data/processed/paciente1

# Walk-forward validation
python scripts/ml/walk_forward_validation.py --data-path data/processed/paciente1
```

## Servicios Docker (Opcional)

```bash
# Levantar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar
docker-compose down
```

### URLs de Servicios

| Servicio | URL |
|----------|-----|
| API Gateway | http://localhost:8000 |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 |
| MinIO | http://localhost:9001 |

## Estructura de Salida

```
data/processed/paciente1/
├── daily_features.parquet          # Features por día
├── window_features.parquet         # Features por ventana
├── training_dataset.parquet        # Dataset original
├── training_dataset_clusters.parquet   # Con labels de clusters
├── training_dataset_engineered.parquet # Con feature engineering
├── walk_forward_results.csv        # Resultados CV temporal
├── optuna_results.json             # Mejores hiperparámetros
├── ensemble_results.json           # Comparativa de modelos
└── *.png                           # Plots generados
```

## Troubleshooting

### Error: "No se encontraron mensajes del paciente"
- Verifica que `--target-sender` coincide con el nombre en el export
- En español, el sender del usuario suele ser `<?>`

### Error: "Dataset vacío"
- Verifica formato del archivo WhatsApp (debe ser `DD/MM/YYYY, HH:MM - Nombre: Mensaje`)
- Revisa que hay mensajes en el rango de fechas

### Error: "AUROC bajo (< 0.5)"
- Verifica balance de labels (ideal 15-40% positivos)
- Usa `regenerate_labels.py` con clusters en lugar de eventos individuales
