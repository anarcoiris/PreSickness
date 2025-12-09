---
description: Investigar modelos GPT, Transformers y técnicas de NLP
---

# Workflow: Investigación en GPTs, Transformers y NLP

Guía para investigar arquitecturas de lenguaje, embeddings y técnicas de procesamiento
de texto aplicables a EM-Predictor.

## Fuentes Primarias

### Repositorios y Papers
1. **arXiv (cs.CL)**: https://arxiv.org/list/cs.CL/recent
   - Computation and Language
   - Filtrar por: transformers, embeddings, multilingual

2. **ACL Anthology**: https://aclanthology.org/
   - Papers de ACL, EMNLP, NAACL
   - Buscar: sentiment analysis, clinical NLP

3. **Hugging Face Papers**: https://huggingface.co/papers
   - Papers con modelos disponibles
   - Trending en NLP

4. **Papers With Code**: https://paperswithcode.com/
   - Benchmarks y leaderboards
   - Implementaciones open source

### Modelos y Código
- **Hugging Face Hub**: https://huggingface.co/models
- **GitHub Trending (Python/ML)**: github.com/trending/python
- **Sentence Transformers**: https://www.sbert.net/

## Áreas de Investigación

### 1. Modelos de Lenguaje para Español
```
Keywords: Spanish BERT, BETO, RoBERTa-BNE, multilingual models
Estado actual:
- BETO: 110M params, baseline español
- RoBERTa-BNE: 355M params, alta calidad
- mBERT/XLM-R: Multilingües
```

**Modelos a evaluar:**
- [ ] `dccuchile/bert-base-spanish-wwm-cased` (BETO)
- [ ] `PlanTL-GOB-ES/roberta-large-bne`
- [ ] `bertin-project/bertin-roberta-base-spanish`
- [ ] `intfloat/multilingual-e5-large`

### 2. Sentence Embeddings
```
Keywords: sentence transformers, contrastive learning, semantic similarity
Aplicación: Representación de mensajes para clustering y predicción
```

**Benchmarks:**
- STS Benchmark (Semantic Textual Similarity)
- MTEB (Massive Text Embedding Benchmark)

### 3. Análisis de Sentimiento y Emoción
```
Keywords: sentiment analysis, emotion detection, affect recognition
Aplicación: Detectar cambios emocionales pre-brote
```

**Datasets español:**
- TASS (Taller de Análisis de Sentimientos)
- SemEval español
- EmoEvent

### 4. Perplexity y Coherencia
```
Keywords: language model perplexity, text coherence, linguistic entropy
Hipótesis: Perplexity aumenta con deterioro cognitivo
```

### 5. Transformers Temporales
```
Keywords: temporal attention, time-aware transformers, longitudinal NLP
Modelos: TFT, Informer, Autoformer
```

## Papers Clave

### Arquitecturas Fundamentales
- [ ] "Attention Is All You Need" (Vaswani et al., 2017)
- [ ] "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
- [ ] "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (Liu et al., 2019)

### Sentence Embeddings
- [ ] "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (Reimers & Gurevych, 2019)
- [ ] "Text and Code Embeddings by Contrastive Pre-Training" (OpenAI, 2022)
- [ ] "E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training" (Wang et al., 2022)

### Modelos Español
- [ ] "Spanish Pre-trained BERT Model and Evaluation Data" (Cañete et al., 2020) - BETO
- [ ] "MarIA: Spanish Language Models" (Gutiérrez-Fandiño et al., 2021)
- [ ] "ALBETO and DistilBETO" (Cañete et al., 2022)

### NLP Clínico
- [ ] "Clinical BERT: Modeling Clinical Notes and Predicting Readmission" (Huang et al.)
- [ ] "BioBERT: Pre-trained Biomedical Language Representation Model" (Lee et al.)

## Checklist de Investigación

### Fase 1: Benchmark de Modelos (2-4 horas)
- [ ] Descargar 3-5 modelos candidatos
- [ ] Evaluar en textos de ejemplo del paciente
- [ ] Medir latencia y uso de memoria
- [ ] Comparar calidad de embeddings (similaridad coseno)

### Fase 2: Experimentos (4-8 horas)
- [ ] Fine-tuning en dominio (opcional)
- [ ] Evaluar perplexity como feature
- [ ] Probar clustering de mensajes por embedding

### Fase 3: Integración
- [ ] Seleccionar modelo final
- [ ] Optimizar para producción (quantization, ONNX)
- [ ] Documentar decisión

## Comparativa de Modelos

| Modelo | Params | Dim | Español | Velocidad | Calidad |
|--------|--------|-----|---------|-----------|---------|
| distiluse-multilingual | 135M | 512 | ✓ | ⚡⚡⚡ | ⭐⭐ |
| paraphrase-mpnet | 278M | 768 | ✓ | ⚡⚡ | ⭐⭐⭐ |
| multilingual-e5-large | 560M | 1024 | ✓ | ⚡ | ⭐⭐⭐⭐ |
| BETO | 110M | 768 | ✓✓ | ⚡⚡ | ⭐⭐⭐ |
| RoBERTa-BNE | 355M | 1024 | ✓✓✓ | ⚡ | ⭐⭐⭐⭐ |

## Código de Evaluación Rápida

```python
# Comparar embeddings de modelos
from sentence_transformers import SentenceTransformer
import numpy as np

models = [
    "distiluse-base-multilingual-cased-v1",
    "paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
]

test_texts = [
    "Estoy muy cansado hoy",
    "Me siento agotado",  # Similar
    "El cielo está azul",  # Diferente
]

for model_name in models:
    model = SentenceTransformer(model_name)
    embs = model.encode(test_texts)
    
    # Similaridad entre texto 0 y 1 (deberían ser similares)
    sim_01 = np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]))
    
    # Similaridad entre texto 0 y 2 (deberían ser diferentes)
    sim_02 = np.dot(embs[0], embs[2]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[2]))
    
    print(f"{model_name}:")
    print(f"  Similar (0↔1): {sim_01:.3f}")
    print(f"  Diferente (0↔2): {sim_02:.3f}")
    print(f"  Discriminación: {sim_01 - sim_02:.3f}")
```

## APIs de LLM Online

### Para Análisis Avanzado (Fallback)

| API | Modelo | Precio/1M tokens | Uso |
|-----|--------|------------------|-----|
| OpenAI | text-embedding-3-small | $0.02 | Embeddings |
| OpenAI | gpt-4o-mini | $0.15 | Análisis |
| Anthropic | claude-3-haiku | $0.25 | Análisis |
| Cohere | embed-multilingual-v3 | $0.10 | Embeddings |
| Jina AI | jina-embeddings-v2-es | $0.02 | Español |

### Consideraciones de Privacidad
> [!WARNING]
> Para datos de pacientes, preferir modelos locales.
> APIs solo para datos anonimizados o sintéticos.

## Output Esperado

Crear documento en `docs/research/llm_evaluation.md`:
```markdown
# Evaluación de Modelos LLM para EM-Predictor

## Modelos Evaluados
[Tabla con resultados]

## Recomendación
[Justificación]

## Configuración Óptima
[Parámetros]
```

## Recursos Adicionales

- **Transformers Course**: https://huggingface.co/learn/nlp-course
- **LLM Course**: https://github.com/mlabonne/llm-course
- **Sentence Transformers Docs**: https://www.sbert.net/docs/
- **Spanish NLP Resources**: https://github.com/dccuchile/spanish-word-embeddings
