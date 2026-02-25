# Arquitectura y Diseño Consolidado para Modelado Predictivo (PreSickness)

## 1. Principios de Arquitectura
- **Privacy-by-Design:** Desde el origen hasta inferencia.
- **Desacoplamiento:** Componentes de ML separados del backend transaccional mediante colas asíncronas y tablas de estado.
- **Resiliencia Evolutiva (Consolidable):** Diseñado para empezar de forma centralizada y migrar pacíficamente hacia Federated Learning o *On-device ML* en fases futuras.

## 2. Flujo de Datos Arquitectónico (Estado Actual Consolidado)

```mermaid
graph TD
    A[Dispositivo Paciente / WhatsApp] -->|Mensaje Crudo| B[API de Ingesta]
    B -->|Seudonimización & Hashing| C[(Tabla: raw_messages)]
    
    C -->|Worker Asíncrono| D{Extractor NLP / Features}
    D -->|Análisis Básico - Counts, Typos| E[(Tabla: datapoints)]
    D -->|Inferencia Semántica - Embeddings| E
    
    E -->|Batch Diario / Ventanas| F[Motor de ML]
    G[(Tabla: clinical_events)] -->|Target / Etiquetas| F
    
    F -->|Baseline EWMA| H[Generador de Alertas]
    F -->|Predicciones XGBoost/TFT| H
    H -->|Probabilidad > Umbral| I[(Tabla: alerts / predictions)]
    I --> J[Dashboard Médico]
    
    J -->|Feedback (False Positive)| G
```

## 3. Componentes Detallados

### 3.1. Capa de Ingesta y Almacenamiento Seguro
- Las fuentes externas interactúan con endpoints estrictos.
- La tabla `raw_messages` ofusca todo contenido: `content_encrypted`, `content_hash`. Se extrae el `patient_id` subyacente de la autenticación.

### 3.2. Worker NLP (Pipeline de `datapoints`)
- **Proceso:** Script en segundo plano escanea mensajes no procesados (`nlp_level=0`).
- **Acción:** Transforma el contenido en métricas agregables (ej. JSON `numeric_features`: `word_count`, `symptom_rate`, `typo_rate`) y genera embeddings vectoriales.
- **Salida:** Borrado o consolidación del crudo, persistencia pura en `datapoints`.

### 3.3. Motor de Inferencia y Alertas
- Consultas periódicas agregan `datapoints` por paciente (Día o 3-Días).
- Un modelo *ensemble* ejecuta predicciones:
  - Nivel 1: Alertas tempranas estadísticas (robustas, bajo nivel).
  - Nivel 2: Modelos supervisados para clasificación de riesgo alto.
- Resultados guardados en tabla `predictions` y se derivan en la tabla `alerts` con niveles escalados.

## 4. Evolución de la Arquitectura (Diseño Consolidable a Futuro)

Para mitigar los riesgos éticos de centralizar bases de datos con comunicaciones de usuarios (aun estando ofuscadas), la arquitectura debe poder evolucionar hacia:

### 4.1. Edge Feature Extraction (Federated Learning Inicial)
- Mover el bloque *Extractor NLP / Features* al dispositivo móvil del paciente (App local).
- La nube PreSickness ya no recibe `raw_messages`, recibe directamente los vectores empaquetados (`datapoints`) a través del endpoint existente. 
- *Beneficio:* Privacidad máxima (cero textos en servidor), descarga de coste computacional. Reúso de la tabla `datapoints`.

### 4.2. Secure Aggregation
- Los modelos no actualizan sus pesos observando historiales individuales, sino combinando gradientes ofuscados (con Differential Privacy) de múltiples terminales (Federated Learning pesado).
- *Beneficio:* Escalabilidad poblacional segura frente a normativas hiper-restrictivas.
