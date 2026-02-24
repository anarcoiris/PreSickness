# Plan de Integración MiniLLM para EM Predictor

**Responsables:** Agent ML (Brain), Agent Backend (Backus)  
**Estado:** 🟢 Integrado como backend opcional en ETL (`scripts/etl/embeddings.py`)  
**Última actualización:** 06/02/2026

---

## 📋 Resumen

Integrar el modelo **MiniLLM (TinyGPTv2)** existente como extractor de features lingüísticas para el pipeline de predicción de brotes de Esclerosis Múltiple. El objetivo es capturar patrones sutiles en el lenguaje que puedan anticipar episodios clínicos.

---

## 🎯 Objetivos

1. **Extraer embeddings contextuales** de los mensajes diarios usando las capas internas de MiniLLM.
2. **Fine-tune opcional** del modelo en el corpus específico del paciente para mejorar la representación.
3. **Calcular features derivadas** (perplexity, entropía, coherencia) que complementen sentiment y métricas léxicas.
4. **Integrar con el pipeline TFT** como features adicionales de entrada.

---

## 🏗️ Arquitectura Propuesta

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FLUJO DE DATOS                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Mensajes crudos]                                                  │
│        │                                                            │
│        ▼                                                            │
│  ┌─────────────────┐                                                │
│  │ parse_chat_     │  ← Limpieza, normalización, timestamps         │
│  │ export.py       │                                                │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐     ┌─────────────────┐                        │
│  │ Feature         │────▶│ MiniLLM         │                        │
│  │ Extractor       │     │ (TinyGPTv2)     │                        │
│  │ (existente)     │     │                 │                        │
│  └────────┬────────┘     └────────┬────────┘                        │
│           │                       │                                 │
│           │  ┌────────────────────┘                                 │
│           │  │ embeddings (n_embd=256/384)                          │
│           │  │ perplexity por mensaje                               │
│           │  │ attention entropy                                    │
│           ▼  ▼                                                      │
│  ┌─────────────────┐                                                │
│  │ Feature Store   │  ← Postgres/Redis                              │
│  │ (feature_       │                                                │
│  │  windows)       │                                                │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │ TFT Training /  │  ← Modelo de series temporales                 │
│  │ Inference       │                                                │
│  └─────────────────┘                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Features a Extraer con MiniLLM

### Nivel 1: Embeddings directos
| Feature | Dimensión | Descripción |
|---------|-----------|-------------|
| `emb_mean` | n_embd | Media de embeddings de tokens del mensaje |
| `emb_cls` | n_embd | Embedding del primer token (si se usa CLS) |
| `emb_last` | n_embd | Embedding del último token |

### Nivel 2: Métricas derivadas
| Feature | Tipo | Descripción |
|---------|------|-------------|
| `perplexity` | float | Qué tan "sorprendente" es el texto para el modelo |
| `attention_entropy` | float | Dispersión de la atención (foco vs difuso) |
| `token_repetition` | float | Ratio de tokens repetidos |
| `vocab_coverage` | float | % del vocabulario usado vs disponible |

### Nivel 3: Patrones temporales
| Feature | Tipo | Descripción |
|---------|------|-------------|
| `perplexity_trend_7d` | float | Tendencia de perplexity últimos 7 días |
| `emb_drift_7d` | float | Distancia coseno entre embeddings actuales y hace 7 días |
| `coherence_drop` | float | Cambio en coherencia semántica |

---

## 🔧 Implementación Técnica

### Paso 1: Wrapper de MiniLLM para embeddings

```python
# services/feature-extractor/minillm_embeddings.py

import torch
from MiniLLM.model import TinyGPTv2

class MiniLLMEmbedder:
    """Extrae embeddings usando MiniLLM."""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.model, self.tokenizer = self._load_model(checkpoint_path)
        self.model.eval()
    
    def _load_model(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = ckpt.get("model_config", {})
        
        model = TinyGPTv2(
            vocab_size=config.get("vocab_size", 32000),
            block_size=config.get("block_size", 256),
            n_embd=config.get("n_embd", 384),
            n_layer=config.get("n_layer", 8),
            n_head=config.get("n_head", 6),
            use_rope=config.get("use_rope", True),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        
        # Cargar tokenizer
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file("MiniLLM/tokenizer.json")
        
        return model, tokenizer
    
    def get_embeddings(self, text: str) -> dict:
        """Extrae embeddings y métricas de un texto."""
        ids = self.tokenizer.encode(text).ids
        if len(ids) == 0:
            return self._empty_embeddings()
        
        # Truncar a block_size
        ids = ids[:self.model.block_size]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # Forward pass - obtener embeddings internos
            tok_emb = self.model.token_emb(x)
            
            # Pasar por bloques transformer
            if self.model.use_rope:
                cos, sin = self.model.rotary_emb(x.size(1), device=x.device)
                rope_cos_sin = (cos, sin)
            else:
                rope_cos_sin = None
            
            hidden = self.model.drop(tok_emb)
            for block in self.model.blocks:
                hidden = block(hidden, rope_cos_sin=rope_cos_sin)
            
            hidden = self.model.ln_f(hidden)  # (1, seq_len, n_embd)
            
            # Calcular perplexity
            logits = self.model.head(hidden)
            perplexity = self._compute_perplexity(logits, x)
        
        # Extraer features
        hidden_np = hidden[0].cpu().numpy()
        
        return {
            "emb_mean": hidden_np.mean(axis=0).tolist(),
            "emb_last": hidden_np[-1].tolist(),
            "perplexity": perplexity,
            "seq_len": len(ids),
        }
    
    def _compute_perplexity(self, logits, targets):
        """Calcula perplexity del texto."""
        # Shift para autoregressive
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction="mean"
        )
        return float(torch.exp(loss).item())
    
    def _empty_embeddings(self):
        return {
            "emb_mean": [0.0] * self.model.n_embd,
            "emb_last": [0.0] * self.model.n_embd,
            "perplexity": 0.0,
            "seq_len": 0,
        }
```

### Paso 2: Integración con Feature Extractor existente

Añadir al `services/feature-extractor/worker.py`:

```python
# En FeatureExtractor.__init__
self.minillm_embedder = MiniLLMEmbedder(
    checkpoint_path="MiniLLM/runs/colmena/ckpt_best.pt",
    device=self.device
)

# Nuevo método
def extract_llm_features(self, text: str) -> dict:
    """Extrae features usando MiniLLM."""
    return self.minillm_embedder.get_embeddings(text)
```

### Paso 3: Actualizar schema de feature_windows

```sql
-- Añadir columnas para embeddings LLM
ALTER TABLE feature_windows 
ADD COLUMN IF NOT EXISTS llm_embedding FLOAT[] DEFAULT NULL,
ADD COLUMN IF NOT EXISTS llm_perplexity FLOAT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS llm_perplexity_trend FLOAT DEFAULT NULL;
```

---

## 📁 Preparación del Dataset

### Formato esperado de entrada

```
dataset/
├── messages/
│   ├── 2024-01-01.txt   # Mensajes del día
│   ├── 2024-01-02.txt
│   └── ...
├── events.csv           # Eventos clínicos
│   # date,event_type,severity,notes
│   # 2024-03-15,relapse,moderate,fatiga severa
│   # 2024-06-20,relapse,mild,hormigueo piernas
└── metadata.json        # Info del paciente (anonimizada)
```

### Script de preparación (a ejecutar cuando tengas el dataset)

```bash
# 1. Parsear mensajes (si vienen de WhatsApp/Telegram)
python MiniLLM/parse_chat_export.py export.json --out dataset/messages/

# 2. Crear dataset supervisado
python scripts/prepare_ms_dataset.py \
    --messages dataset/messages/ \
    --events dataset/events.csv \
    --output dataset/prepared/ \
    --horizons 7,14,30

# 3. Extraer features con MiniLLM
python scripts/extract_llm_features.py \
    --input dataset/prepared/features.parquet \
    --checkpoint MiniLLM/runs/colmena/ckpt_best.pt \
    --output dataset/prepared/features_with_llm.parquet
```

---

## 🧪 Validación

### Métricas objetivo
- **Correlación perplexity-brote**: Esperamos que perplexity aumente ~7-14 días antes de un brote
- **Embedding drift**: Cambio significativo en representación semántica pre-brote
- **Feature importance**: LLM features deberían aparecer en top-10 de SHAP values

### Experimentos planificados
1. **Baseline sin LLM**: Solo features léxicas + sentiment
2. **Con embeddings MiniLLM**: Añadir emb_mean como feature
3. **Con perplexity**: Añadir perplexity y su tendencia
4. **Full**: Todas las features LLM

---

## ⏱️ Timeline estimado

| Fase | Duración | Dependencias |
|------|----------|--------------|
| Recibir dataset etiquetado | — | Usuario |
| Parsear y limpiar mensajes | 1 día | Dataset |
| Extraer features LLM | 1 día | Paso anterior |
| Entrenar TFT con LLM features | 2-3 días | Features |
| Evaluar y comparar | 1 día | Modelo |

---

## 📝 Notas

- El modelo `colmena` ya entrenado puede usarse directamente o hacer fine-tune en el corpus específico
- Perplexity es especialmente interesante: un modelo "sorprendido" por el texto puede indicar patrones anómalos
- Los embeddings de 384 dims pueden reducirse con PCA si es necesario para el TFT

---

## 🔗 Referencias

- `MiniLLM/README.md` - Documentación del modelo
- `MiniLLM/model.py` - Arquitectura TinyGPTv2
- `MiniLLM/generation.py` - Funciones de perplexity
- `train_tft.py` - Pipeline de entrenamiento actual

