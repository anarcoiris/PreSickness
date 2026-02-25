# Plan de Investigación: Modelos Predictivos basados en Mensajería (PreSickness)

## 1. Objetivos de Investigación
El objetivo principal de esta investigación es determinar la viabilidad y eficacia de utilizar datos de mensajería (ej. WhatsApp) procesados y anonimizados para la predicción temprana de brotes o recaídas en el contexto clínico de PreSickness.

## 2. Líneas de Investigación Principales

### 2.1. Extracción de Señales y NLP
- **Identificación de Síntomas:** Evaluar diccionarios clínicos vs. modelos de extracción de entidades (NER) médicos aplicados a lenguaje coloquial.
- **Embeddings Semánticos:** Investigar modelos ligeros (ej. `sentence-transformers`) adecuados para textos cortos e informales.
- **Señales de Comportamiento (Digital Exhaust):** Estudiar la correlación entre la dinámica de tecleo (longitud de mensaje, tiempo de respuesta, *typos*) y el estrés cognitivo/físico o deterioro de la salud.

### 2.2. Modelado y Algoritmia
- **Baselines Estadísticos:** Análisis del rendimiento de Control Charts, EWMA y ARIMA en la detección de anomalías en conteos temporales.
- **Modelos de Machine Learning (Supervisados):** Comparativa de XGBoost, LightGBM y Random Forest usando *features* agregados.
- **Modelos Secuenciales:** Evaluación de arquitecturas como Temporal Convolutional Networks (TCN) y Transformers genéricos para series temporales (ej. Temporal Fusion Transformer - TFT).

### 2.3. Criterios Clínicos y Epidemiológicos
- **Validación del "Lag" Clínico:** Investigar la diferencia temporal entre el inicio biológico de la recaída (señales sutiles en texto) y la fecha de confirmación clínica (`event_date`).
- **Fatiga de Alarma:** Referenciar estudios sobre la tolerancia de los profesionales médicos a los falsos positivos en sistemas de alerta temprana.

## 3. Riesgos, Sesgos y Ética
- **Privacidad:** Investigar el impacto legal y las soluciones criptográficas (Hashing irreversible, Differential Privacy) necesarias bajo la normativa GDPR/HIPAA.
- **Sesgos Poblacionales:** Analizar cómo la demografía de los usuarios de smartphones/WhatsApp puede introducir sesgos de representatividad en el modelo.
- **Robusted ante Anomalías Sociales:** Estudiar el impacto del ruido externo (noticias en medios, campañas de salud que disparan el uso de palabras clave sin haber enfermedad).
