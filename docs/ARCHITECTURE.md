# Arquitectura EM-Predictor

## Visión General

```
┌─────────────────────────────────────────────────────────────┐
│                      INGESTA DE DATOS                       │
├──────────────┬──────────────┬──────────────┬───────────────┤
│   WhatsApp   │   Telegram   │     CSV      │    API Rest   │
└──────┬───────┴──────┬───────┴──────┬───────┴───────┬───────┘
       │              │              │               │
       ▼              ▼              ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                     ETL PIPELINE                            │
│  extract_events.py → cluster_signals.py → pipeline.py      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE STORE                             │
│         data/processed/{patient_id}/*.parquet               │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML PIPELINE                              │
│  feature_engineering → embeddings → ensemble → optuna       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE API                             │
│              FastAPI + Redis Cache + Alerts                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Stack Tecnológico

### Core ML
| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| Training | scikit-learn | Baselines rápidos, bien probados |
| Temporal | pytorch-forecasting | TFT para series temporales |
| HPO | Optuna | Pruning eficiente, fácil de usar |
| Embeddings | sentence-transformers | Multilingüe, local, gratuito |

### Data
| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| Storage | Parquet | Columnar, compresión, fast I/O |
| DB | TimescaleDB | Time-series optimized PostgreSQL |
| Cache | Redis | Feature store en memoria |
| Queue | Redpanda | Kafka-compatible, más simple |

### Infraestructura
| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| Containers | Docker Compose | Desarrollo local simple |
| API | FastAPI | Async, autodocs, validación |
| MLOps | MLflow | Tracking experimentos |
| Monitoring | Prometheus + Grafana | Estándar industria |

---

## Decisiones Clave

### ADR-001: Modelos Locales vs Cloud
**Decisión**: Priorizar modelos locales (sentence-transformers, MiniLLM)

**Contexto**: Datos de salud sensibles, GDPR/HIPAA

**Alternativas**:
- OpenAI API: Mejor calidad pero riesgo de privacidad
- Cloud privado: Costoso para MVP

**Consecuencias**:
- (+) Control total de datos
- (+) Sin costes de API
- (-) Menor capacidad que GPT-4

---

### ADR-002: Ventanas Temporales
**Decisión**: Usar ventanas de 7 días con stride de 1 día

**Contexto**: Balancear granularidad vs ruido

**Alternativas**:
- Ventanas diarias: Muy ruidosas
- Ventanas de 14 días: Pierde granularidad

**Consecuencias**:
- (+) Captura patrones semanales
- (+) Suficientes muestras para training
- (-) Data leakage entre ventanas solapadas

---

### ADR-003: Labels de Brote
**Decisión**: Usar clusters de señales en lugar de eventos individuales

**Contexto**: Eventos individuales generan 95%+ positivos

**Alternativas**:
- Cada mención = evento: Demasiados positivos
- Solo eventos severos: Muy pocos datos

**Consecuencias**:
- (+) Balance 15-30% positivos
- (+) Representa períodos reales de riesgo
- (-) Requiere validación clínica

---

## Features Pipeline

### Extracción (ETL)
```python
# Entrada: WhatsApp export
# Salida: DataFrame con mensajes normalizados

MessageExtractor → filtro sistema → normalización
EventsExtractor → keywords → severidad → clustering
TextProcessor → tokenización → métricas lingüísticas
```

### Ingeniería
```python
# Entrada: Features diarios
# Salida: Features enriquecidos

Lag Features: valores t-1, t-3, t-7
Rolling Stats: mean, std, trend en ventanas 3, 7 días
Change Features: diff, pct_change
Interactions: sentiment × volume, complexity score
```

### Embeddings
```python
# Modelos soportados
PRESETS = {
    "fast": "distiluse-base-multilingual-cased-v1",     # 512d
    "balanced": "paraphrase-multilingual-mpnet-base-v2", # 768d
    "quality": "intfloat/multilingual-e5-large",        # 1024d
}
```

---

## Modelos

### Baseline (Actual)
- **RandomForest**: AUROC 0.6791 (optimizado)
- **GradientBoosting**: AUROC 0.6851 (mejor)
- **Ensemble**: Voting/Stacking de ambos

### Temporal (Próximo)
- **TFT (Temporal Fusion Transformer)**
  - Attention sobre secuencias
  - Interpretabilidad de features
  - Multi-horizon forecasting

---

## Seguridad y Compliance

### Datos en Reposo
- Encriptación AES-256 para datos de pacientes
- Pseudonimización de IDs
- Logs sin PII

### Datos en Tránsito
- HTTPS obligatorio
- JWT para autenticación
- Rate limiting

### GDPR
- Derecho al olvido implementado
- Consentimiento informado requerido
- DPIA documentado en `/LEGAL/`
