# Decisiones de Arquitectura - EM Predictor

## Resumen Ejecutivo

Este documento registra las decisiones arquitectónicas clave y su justificación.

---

## 1. Microservicios vs Monolito

### Decisión: Microservicios con posible consolidación para MVP

**Estado actual:** 5 servicios separados
- `api-gateway`: Ingesta y autenticación
- `feature-extractor`: Procesamiento de features
- `ml-inference`: Predicciones
- `alert-manager`: Notificaciones
- `dashboard`: UI clínica

**Justificación para microservicios:**
- Escalabilidad independiente (inference puede necesitar GPU)
- Despliegue independiente (actualizar modelo sin tocar API)
- Aislamiento de fallos
- Equipos paralelos (si se escala el equipo)

**Recomendación para MVP:**
```
Fase 1 (MVP): Consolidar a 2-3 contenedores
  - backend (api + inference + alerts)
  - frontend (dashboard)
  - workers (feature-extractor)

Fase 2 (Producción): Separar según necesidad de escala
```

**Cuándo separar:**
- Cuando inference necesite GPU dedicada
- Cuando el feature-extractor sea bottleneck
- Cuando haya >1000 usuarios activos

---

## 2. SQL (PostgreSQL/TimescaleDB) vs Alternativas

### Decisión: PostgreSQL + TimescaleDB + Redis

**Justificación:**

| Requisito | Por qué SQL |
|-----------|-------------|
| **Compliance (GDPR/HIPAA)** | Transacciones ACID, audit logs, borrado verificable |
| **Series temporales** | TimescaleDB: compresión, particionado automático, consultas temporales optimizadas |
| **Consultas complejas** | JOINs entre pacientes, eventos, predicciones |
| **Integridad referencial** | FK entre users, datapoints, predictions |

**Alternativas consideradas:**

| Opción | Pros | Contras | Veredicto |
|--------|------|---------|-----------|
| **Parquet/Delta Lake** | Excelente para ML batch | Sin transacciones, sin queries ad-hoc | ❌ Solo para export |
| **MongoDB** | Esquema flexible | Sin series temporales optimizadas | ❌ |
| **InfluxDB** | Optimizado time-series | Sin JOINs, compliance limitado | ❌ |
| **Redis solo** | Ultra rápido | Sin persistencia garantizada | ❌ Solo para caché |

**Arquitectura híbrida actual:**
```
PostgreSQL/TimescaleDB → Almacenamiento principal (compliance, queries)
Redis                  → Caché de features para serving (latencia)
Parquet (export)       → Training de modelos (ML pipelines)
```

---

## 3. Tokenización de Oraciones

### Decisión: Tokenización híbrida con soporte español

**Problema:** `text.split('.')` no maneja:
- Abreviaturas: "Dr. García dijo..."
- Signos españoles: "¿Cómo estás? ¡Genial!"
- Puntos suspensivos: "No sé..."

**Solución implementada:**

```python
def _tokenize_sentences(self, text: str) -> List[str]:
    # 1. Proteger abreviaturas
    abbreviations = {"Dr.", "Dra.", "Sr.", "Sra.", "etc.", ...}
    
    # 2. Proteger puntos suspensivos
    text = text.replace("...", "§§§")
    
    # 3. Split por puntuación final
    sentences = re.split(r'[.!?¡¿]+', text)
    
    # 4. Restaurar
    ...
```

**Alternativa para producción:** spaCy con modelo español
```python
import spacy
nlp = spacy.load("es_core_news_sm")
sentences = [sent.text for sent in nlp(text).sents]
```

**Dependencias añadidas:** `spacy`, `es_core_news_sm` (opcional)

---

## 4. Features de Actividad como Predictores

### Decisión: Incluir métricas de frecuencia comunicativa

**Hipótesis clínica:**
Antes de un brote de EM, los pacientes frecuentemente experimentan:
1. **Fatiga prodrómica** → Menos mensajes
2. **Cambios de sueño** → Más actividad nocturna
3. **Dificultad cognitiva** → Respuestas más lentas
4. **Aislamiento** → Menos horas activas

**Features implementadas:**

| Feature | Descripción | Señal esperada pre-brote |
|---------|-------------|--------------------------|
| `messages_per_day_mean` | Promedio mensajes/día | ↓ Disminuye |
| `messages_per_day_std` | Variabilidad | ↑ Aumenta |
| `messages_trend` | Slope de actividad | ↓ Negativo |
| `active_hours_mean` | Horas distintas con actividad | ↓ Disminuye |
| `night_ratio` | % mensajes 23:00-06:00 | ↑ Aumenta |
| `gap_max_hours` | Máximo gap sin actividad | ↑ Aumenta |
| `coverage` | % días con actividad | ↓ Disminuye |

**Validación necesaria:**
- Correlación con labels de brotes en dataset real
- Feature importance en modelo entrenado
- Análisis de falsos positivos/negativos

---

## 5. Fine-tuning por Paciente

### Decisión: Modelo base + fine-tuning personalizado

**Problema:** Cada paciente tiene:
- Baseline de comunicación diferente (introvertido vs extrovertido)
- Patrones de horario únicos (trabajador nocturno vs diurno)
- Vocabulario y estilo personal

**Arquitectura:**

```
┌─────────────────────────────────────────────────────────────┐
│                    MODELO BASE (TFT)                        │
│  Entrenado con datos agregados o sintéticos                 │
│  Captura patrones generales de deterioro                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               FINE-TUNING POR PACIENTE                      │
│  - Congela embeddings (patrones generales)                  │
│  - Ajusta capas finales (baseline individual)               │
│  - Learning rate bajo (1e-4 vs 1e-3)                        │
│  - Early stopping agresivo (patience=3)                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              MODELO PERSONALIZADO (paciente_001.pt)         │
│  - Normalización específica (z-score del paciente)          │
│  - Threshold calibrado (Platt scaling)                      │
│  - Metadata: fechas entrenamiento, métricas                 │
└─────────────────────────────────────────────────────────────┘
```

**Flujo de datos:**

```
1. ETL Pipeline
   WhatsApp export → Parseo → Features diarios → Ventanas temporales → Parquet

2. Fine-tuning
   Parquet + Labels → PatientDataset → LSTM + Attention → best_model.pt

3. Inferencia
   Nuevos datos → Normalización (params del paciente) → Modelo → P(brote)
```

**Requisitos mínimos para fine-tuning:**
- 30+ días de datos
- Al menos 1 evento positivo (brote) para supervisión
- Sin eventos: usar modelo base con threshold conservador

---

## 6. ETL Pipeline

### Decisión: Pipeline modular con soporte multi-formato

**Formatos soportados:**
- WhatsApp (.txt export)
- ChatGPT conversations (.json)
- CSV genérico
- JSONL

**Estructura de salida:**

```
data/processed/
└── patient_001/
    ├── daily_features.parquet    # Features agregados por día
    ├── window_features.parquet   # Features por ventana (1,3,7,14,30 días)
    ├── labels.parquet            # Labels de brotes por horizonte
    ├── training_dataset.parquet  # Combinado para ML
    └── metadata.json             # Config y estadísticas
```

**Uso:**
```bash
python -m scripts.etl.pipeline \
    --input data/raw/whatsapp_export.txt \
    --events data/raw/brotes.csv \
    --patient-id paciente_001 \
    --output data/processed/
```

---

## 7. Próximos Pasos Arquitectónicos

### Prioridad Alta
1. **Integración MiniLLM** para embeddings personalizados
2. **Tests E2E** del flujo completo
3. **Validación con dataset real** (cuando esté disponible)

### Prioridad Media
4. Consolidar Dockerfiles para MVP
5. Añadir spaCy como dependencia opcional
6. Implementar Platt scaling para calibración de probabilidades

### Prioridad Baja
7. Feature store dedicado (Feast/Featureform)
8. Kubernetes manifests para producción
9. A/B testing de modelos

---

*Documento mantenido por: Agent Architect (Archie) & Agent ML (Brain)*
*Última actualización: 2025-12-02*

