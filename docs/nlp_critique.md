# Análisis Crítico de Implementación NLP / Proxy de Sentimiento

## Resumen del Estado Actual

La implementación actual en `PreSickness` presenta una arquitectura fragmentada en lo que respecta al procesamiento de lenguaje natural (NLP):

1.  **Feature Extractor (`services/feature-extractor/worker.py`)**:
    *   **Motor de Sentimiento**: Utiliza `TextBlob`, una librería basada en reglas y léxicos muy básicos.
    *   **Lógica**: `sentiment = blob.sentiment.polarity` devuelve un valor escalar entre -1.0 y 1.0.
    *   **Limitaciones Críticas**:
        *   **Ignorancia de Contexto**: No entiende sarcasmo, negaciones complejas, ni matices médicos (e.g., "Estoy cansado" puede ser negativo en general, pero crítico en fatiga crónica).
        *   **Reglas Rígidas**: La extracción de rasgos adicionales (orientación temporal, pronombres) se basa en listas "hardcoded" de palabras clave (e.g., "tomorrow", "ayer"). Esto es frágil y difícil de escalar.
        *   **Multilingüismo Débil**: Aunque intenta manejar ES/EN con listas separadas, TextBlob por defecto suele funcionar mejor en inglés, y su precisión en español es inferior sin modelos específicos.

2.  **ML Inference (`services/ml-inference/main.py`)**:
    *   Utiliza el `sentiment_mean` calculado por el extractor como un feature numérico más.
    *   **Riesgo en Heurística**: En el fallback (cuando falla el modelo TFT), se aplica una penalización directa: `score += -0.4 * sentiment`. Esto otorga un peso desproporcionado a una métrica de baja confianza. Un falso negativo en sentimiento dispara artificialmente el riesgo de recaída.

3.  **TinyLLM (`tinyllm/`)**:
    *   Es una implementación "scratch" de un GPT (v2/v3-style) con características modernas (RoPE, RMSNorm-ish, etc.).
    *   **Desconectado**: Actualmente **no** está integrado en el pipeline de inferencia ni en el extractor de features. Es un recurso latente (código muerto desde la perspectiva del servicio activo).

## Crítica en Profundidad

La arquitectura actual sufre de una **disonancia tecnológica**: por un lado, se tiene una implementación sofisticada de un LLM (`tinyllm`), y por otro, el sistema de producción depende de una librería de 2013 (`TextBlob`) para una métrica crítica de salud.

### 1. Calidad de la Señal ("Garbage In, Garbage Out")
El "proxy" de sentimiento actual es ruidoso. En un contexto de predicción de recaídas (PreSickness), la sutileza es clave. TextBlob fallará en distinguir entre:
*   *"Tengo mucho dolor hoy"* (Negativo, síntoma físico, alto riesgo)
*   *"Odio que cancelaran mi serie favorita"* (Negativo, irrelevante clínicamente, bajo riesgo)
Ambos recibirán un score negativo, confundiendo al modelo de predicción (TFT).

### 2. Desaprovechamiento de Recursos
La presencia de `tinyllm` y `SentenceTransformer` (usado solo para embeddings genéricos) indica que la capacidad para hacer un análisis semántico real ya existe en el código, pero no se está orquestando.

### 3. Latencia vs. Precisión
El diseño actual prioriza la velocidad extrema (operaciones de string + diccionario) sobre la precisión. Integrar un LLM (incluso `tinyllm` pequeño) introducirá latencia. El diseño futuro debe equilibrar esto: ¿necesitamos inferencia en tiempo real (<200ms) o asíncrona para el feature store?

## Cuestiones a Esclarecer para el Nuevo Diseño

Para proponer una integración efectiva que reemplace este proxy deficiente, necesito clarificar los siguientes puntos:

1.  **¿Cuál es el rol deseado para el "Language Model"?**
    *   *Opción A (Clasificador):* Fine-tuning de un modelo pequeño (BERT/TinyLLM) para output de clasificación directa (e.g., Probabilidad de Recaída, Nivel de Dolor, Ánimo).
    *   *Opción B (Embeddings Ricos):* Reemplazar el escalar `sentiment` por un vector de embeddings denso que el TFT aprenda a interpretar.
    *   *Opción C (Generador/Analista):* Que el LLM genere un resumen textual clínico o una explicación del estado.

2.  **Restricciones de Hardware / Despliegue**
    *   ¿Disponemos de GPU en inferencia? Correr `tinyllm` o incluso `BERT` por cada mensaje en CPU puede saturar el worker si el volumen de mensajes es alto.
    *   ¿El despliegue es local (on-premise/edge) o nube?

3.  **Integración de TinyLLM**
    *   ¿El objetivo es usar **específicamente** el código de `tinyllm` (por razones pedagógicas o de control total) o preferimos usar librerías estándar optimizadas (HuggingFace/ONNX) con un modelo pre-entrenado robusto?

4.  **Definición de "Mejora"**
    *   ¿Buscamos solo mejor "sentimiento" (Pos/Neg) o métricas clínicas reales (Ansiedad, Fatiga, Confusión)?

## Propuesta Preliminar (Hipótesis)
Recomiendo reemplazar el "proxy" de TextBlob con un **modelo de NLP especializado pequeño (DistilBERT multilingüe o similar)** corriendo en ONNX para velocidad, o integrar una versión cuantizada de **TinyLLM** si se desea capacidad generativa, exponiéndolo como un microservicio interno al que `feature-extractor` consulta asíncronamente.
