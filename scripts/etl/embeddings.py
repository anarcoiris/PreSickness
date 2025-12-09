#!/usr/bin/env python3
"""
Extractores de Embeddings para EM-Predictor.

Soporta múltiples backends:
1. Sentence Transformers (local, gratuito)
2. MiniLLM/TinyGPT (local, con perplexity)
3. OpenAI API (cloud, de pago)

Uso:
    from embeddings import EmbeddingExtractor
    extractor = EmbeddingExtractor(backend="sentence-transformers")
    embeddings = extractor.encode(["Hola mundo", "Otro mensaje"])
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class EmbeddingConfig:
    """Configuración para extracción de embeddings."""
    
    backend: str = "sentence-transformers"  # sentence-transformers, minillm, openai
    
    # Sentence Transformers
    st_model: str = "intfloat/multilingual-e5-large"
    st_batch_size: int = 32
    st_normalize: bool = True
    
    # MiniLLM
    minillm_checkpoint: Optional[Path] = None
    minillm_tokenizer: Optional[Path] = None
    
    # OpenAI
    openai_model: str = "text-embedding-3-small"
    openai_batch_size: int = 100
    
    # Cache
    cache_dir: Path = Path("data/embeddings_cache")
    use_cache: bool = True
    
    # Modelo presets
    PRESETS: Dict[str, Dict] = field(default_factory=lambda: {
        "fast": {
            "st_model": "distiluse-base-multilingual-cased-v1",
            "dim": 512,
        },
        "balanced": {
            "st_model": "paraphrase-multilingual-mpnet-base-v2",
            "dim": 768,
        },
        "quality": {
            "st_model": "intfloat/multilingual-e5-large",
            "dim": 1024,
        },
        "spanish": {
            "st_model": "hiiamsid/sentence_similarity_spanish_es",
            "dim": 768,
        },
    })
    
    @classmethod
    def from_preset(cls, preset: str) -> "EmbeddingConfig":
        """Crea configuración desde un preset."""
        config = cls()
        if preset in config.PRESETS:
            config.st_model = config.PRESETS[preset]["st_model"]
        return config


# ══════════════════════════════════════════════════════════════════════════════
# BASE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════


class BaseEmbeddingExtractor(ABC):
    """Interfaz base para extractores de embeddings."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._cache: Dict[str, np.ndarray] = {}
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Codifica textos a embeddings."""
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensión de los embeddings."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nombre del modelo usado."""
        pass
    
    def encode_with_cache(self, texts: List[str]) -> np.ndarray:
        """Codifica con cache opcional."""
        if not self.config.use_cache:
            return self.encode(texts)
        
        # Separar textos cacheados y nuevos
        cached_embeddings = {}
        texts_to_encode = []
        indices_to_encode = []
        
        for i, text in enumerate(texts):
            cache_key = hash(text)
            if cache_key in self._cache:
                cached_embeddings[i] = self._cache[cache_key]
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Codificar textos nuevos
        if texts_to_encode:
            new_embeddings = self.encode(texts_to_encode)
            for idx, text, emb in zip(indices_to_encode, texts_to_encode, new_embeddings):
                cache_key = hash(text)
                self._cache[cache_key] = emb
                cached_embeddings[idx] = emb
        
        # Reconstruir array en orden
        result = np.zeros((len(texts), self.embedding_dim))
        for i in range(len(texts)):
            result[i] = cached_embeddings[i]
        
        return result


# ══════════════════════════════════════════════════════════════════════════════
# SENTENCE TRANSFORMERS
# ══════════════════════════════════════════════════════════════════════════════


class SentenceTransformerExtractor(BaseEmbeddingExtractor):
    """Extractor usando Sentence Transformers (local, gratuito)."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._model = None
        self._dim = None
    
    @property
    def model(self):
        """Carga modelo de forma lazy."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Cargando modelo: {self.config.st_model}")
                self._model = SentenceTransformer(self.config.st_model)
                self._dim = self._model.get_sentence_embedding_dimension()
                logger.info(f"Modelo cargado. Dimensión: {self._dim}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers no instalado. "
                    "Ejecuta: pip install sentence-transformers"
                )
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            _ = self.model  # Trigger lazy load
        return self._dim
    
    @property
    def model_name(self) -> str:
        return self.config.st_model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Codifica textos usando Sentence Transformers."""
        if not texts:
            return np.array([])
        
        # Para modelos E5, añadir prefijo
        if "e5" in self.config.st_model.lower():
            texts = [f"query: {t}" for t in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.st_batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.config.st_normalize,
        )
        
        return np.array(embeddings)


# ══════════════════════════════════════════════════════════════════════════════
# MINILLM PERPLEXITY
# ══════════════════════════════════════════════════════════════════════════════


class MiniLLMExtractor(BaseEmbeddingExtractor):
    """Extractor usando MiniLLM local para perplexity como feature."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Carga modelo MiniLLM."""
        import sys
        import torch
        
        # Añadir path de tinyllm
        tinyllm_path = Path(__file__).parent.parent.parent / "tinyllm"
        if str(tinyllm_path) not in sys.path:
            sys.path.insert(0, str(tinyllm_path))
        
        from tokenizers import Tokenizer
        from model import TinyGPTv2
        
        # Cargar tokenizer
        tokenizer_path = self.config.minillm_tokenizer or tinyllm_path / "tokenizer.json"
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        # Cargar modelo
        ckpt_path = self.config.minillm_checkpoint
        if ckpt_path and ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model_args = checkpoint.get("model_args", {})
            self._model = TinyGPTv2(
                vocab_size=self._tokenizer.get_vocab_size(),
                **model_args
            )
            self._model.load_state_dict(checkpoint["model"])
            self._model.eval()
        else:
            logger.warning("No se encontró checkpoint de MiniLLM")
            self._model = None
    
    @property
    def embedding_dim(self) -> int:
        return 1  # Solo perplexity como feature escalar
    
    @property
    def model_name(self) -> str:
        return "MiniLLM-Perplexity"
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Calcula perplexity para cada texto."""
        if self._model is None:
            self._load_model()
        
        if self._model is None:
            logger.warning("MiniLLM no disponible, retornando zeros")
            return np.zeros((len(texts), 1))
        
        import torch
        
        perplexities = []
        
        with torch.no_grad():
            for text in texts:
                tokens = self._tokenizer.encode(text).ids
                if len(tokens) < 2:
                    perplexities.append(0.0)
                    continue
                
                tokens_tensor = torch.tensor([tokens])
                logits, _ = self._model(tokens_tensor)
                
                # Calcular cross-entropy
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = tokens_tensor[:, 1:].contiguous()
                
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                
                perplexity = torch.exp(loss).item()
                perplexities.append(min(perplexity, 1000))  # Cap para evitar infinitos
        
        return np.array(perplexities).reshape(-1, 1)


# ══════════════════════════════════════════════════════════════════════════════
# OPENAI API
# ══════════════════════════════════════════════════════════════════════════════


class OpenAIExtractor(BaseEmbeddingExtractor):
    """Extractor usando OpenAI API (requiere API key)."""
    
    EMBEDDING_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._client = None
    
    @property
    def client(self):
        """Inicializa cliente OpenAI."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                raise ImportError("openai no instalado. Ejecuta: pip install openai")
        return self._client
    
    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIMS.get(self.config.openai_model, 1536)
    
    @property
    def model_name(self) -> str:
        return f"OpenAI/{self.config.openai_model}"
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Codifica usando OpenAI API."""
        if not texts:
            return np.array([])
        
        embeddings = []
        
        # Procesar en batches
        for i in range(0, len(texts), self.config.openai_batch_size):
            batch = texts[i:i + self.config.openai_batch_size]
            
            response = self.client.embeddings.create(
                model=self.config.openai_model,
                input=batch,
            )
            
            for item in response.data:
                embeddings.append(item.embedding)
        
        return np.array(embeddings)


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════


class EmbeddingExtractor:
    """Factory para crear extractores de embeddings."""
    
    BACKENDS = {
        "sentence-transformers": SentenceTransformerExtractor,
        "st": SentenceTransformerExtractor,
        "minillm": MiniLLMExtractor,
        "openai": OpenAIExtractor,
    }
    
    def __init__(
        self,
        backend: str = "sentence-transformers",
        preset: str = "balanced",
        **kwargs
    ):
        """
        Crea extractor de embeddings.
        
        Args:
            backend: "sentence-transformers", "minillm", o "openai"
            preset: "fast", "balanced", "quality", o "spanish"
            **kwargs: Configuración adicional
        """
        config = EmbeddingConfig.from_preset(preset)
        config.backend = backend
        
        # Aplicar kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        if backend not in self.BACKENDS:
            raise ValueError(f"Backend no soportado: {backend}")
        
        self._extractor = self.BACKENDS[backend](config)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Codifica textos a embeddings."""
        return self._extractor.encode(texts)
    
    def encode_with_cache(self, texts: List[str]) -> np.ndarray:
        """Codifica con cache."""
        return self._extractor.encode_with_cache(texts)
    
    @property
    def embedding_dim(self) -> int:
        return self._extractor.embedding_dim
    
    @property
    def model_name(self) -> str:
        return self._extractor.model_name


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def main():
    """CLI para test de embeddings."""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Test de extractores de embeddings")
    parser.add_argument("--backend", "-b", default="sentence-transformers", 
                       choices=["sentence-transformers", "minillm", "openai"])
    parser.add_argument("--preset", "-p", default="balanced",
                       choices=["fast", "balanced", "quality", "spanish"])
    parser.add_argument("--texts", "-t", nargs="+", 
                       default=["Hola mundo", "Estoy muy cansado hoy", "Me duele la cabeza"])
    
    args = parser.parse_args()
    
    print(f"\n🔧 Backend: {args.backend}")
    print(f"📦 Preset: {args.preset}")
    
    extractor = EmbeddingExtractor(backend=args.backend, preset=args.preset)
    
    print(f"📐 Dimensión: {extractor.embedding_dim}")
    print(f"🏷️ Modelo: {extractor.model_name}")
    
    print(f"\n📝 Codificando {len(args.texts)} textos...")
    start = time.time()
    embeddings = extractor.encode(args.texts)
    elapsed = time.time() - start
    
    print(f"✅ Shape: {embeddings.shape}")
    print(f"⏱️ Tiempo: {elapsed:.2f}s ({elapsed/len(args.texts)*1000:.1f}ms/texto)")
    
    # Similaridad entre textos
    if len(args.texts) >= 2:
        from numpy.linalg import norm
        sim = np.dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        print(f"\n📊 Similaridad '{args.texts[0]}' ↔ '{args.texts[1]}': {sim:.3f}")


if __name__ == "__main__":
    main()
