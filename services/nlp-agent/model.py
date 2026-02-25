import logging
import hashlib
import time
import numpy as np
import os
import httpx
import json
import re
from typing import Dict, List, Tuple

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("nlp-agent")

class NlpEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", heads_onnx: str = "heads_v1.onnx"):
        self.encoder_name = model_name
        self.heads_path = heads_onnx
        self.model_version = "mvp-v1-senttrans"
        
        # Load Encoder
        logger.info(f"Loading SentenceTransformer: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        # Load Heads (ONNX)
        self.heads_session = None
        if HAS_ORT and os.path.exists(self.heads_path):
            try:
                logger.info(f"Loading ONNX heads from {self.heads_path}")
                self.heads_session = ort.InferenceSession(self.heads_path)
                self.model_version = f"st-{model_name}+onnx-heads-v1"
            except Exception as e:
                logger.error(f"Failed to load ONNX heads: {e}")
        else:
            logger.warning("ONNX heads not found or ORT missing. Using dummy heads.")

    def compute_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> Tuple[List[float], int]:
        if not text:
            dim = 384
            return [0.0] * dim, dim
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        return embedding.tolist(), len(embedding)

    def predict_heads(self, embedding: List[float], text: str) -> Dict[str, Dict]:
        """
        Predicts symptom scores using local Ollama LLM if available, otherwise dummy.
        """
        symptoms = ["pain", "fatigue", "anxiety", "mood", "sleep"]
        
        # OLLAMA INTEGRATION
        ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        ollama_model = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")
        
        prompt = f"""Analyze the following patient message and determine the probability (0.0 to 1.0) of the patient experiencing each of these 5 symptoms: pain, fatigue, anxiety, mood, sleep.
Return ONLY a valid JSON object with the symptom names as keys and the probabilities as float values. No explanations.
Message: "{text}"
JSON:"""

        try:
            # We use format: "json" to coerce ollama models (especially llama3/mistral) to output valid JSON
            logger.info(f"Querying Ollama ({ollama_model}) for symptom extraction...")
            req = httpx.post(
                f"{ollama_url}/api/generate",
                json={"model": ollama_model, "prompt": prompt, "stream": False, "format": "json"},
                timeout=45.0
            )
            if req.status_code == 200:
                response_text = req.json().get("response", "")
                
                # Remove <think> blocks (Deepseek-R1 creates these)
                clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
                
                # Attempt to find JSON
                match = re.search(r'\{.*\}', clean_text, flags=re.DOTALL)
                if match:
                    parsed_json = json.loads(match.group(0))
                else:
                    parsed_json = json.loads(clean_text)
                
                results = {}
                for sym in symptoms:
                    prob = float(parsed_json.get(sym, 0.0))
                    prob = max(0.0001, min(0.9999, prob)) # Clamp
                    logit = float(np.log(prob / (1 - prob)))
                    results[sym] = {
                        "prob": round(prob, 4),
                        "logit": round(logit, 4),
                        "uncertainty": 0.1
                    }
                logger.info(f"Ollama extracted symptoms successfully.")
                return results
                
        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            
        logger.warning("Ollama API failed. Falling back to dummy logic.")
        
        # Fallback to Dummy logic
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(h % 2**32)
        results = {}
        for sym in symptoms:
            prob = float(np.random.beta(2, 5))
            results[sym] = {
                "prob": round(prob, 4),
                "logit": round(np.log(prob / (1 - prob + 1e-9)), 4),
                "uncertainty": 0.3
            }
        return results

    def extract_linguistic_meta(self, text: str) -> Dict:
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        token_count = len(words)
        pronouns = {"yo", "me", "mi", "conmigo", "i", "my", "mine"}
        pronoun_count = sum(1 for w in words if w in pronouns)
        pronoun_ratio = pronoun_count / token_count if token_count > 0 else 0.0
        negations = {"no", "nunca", "jamás", "tampoco", "nadie", "nada", "not", "never"}
        negation_count = sum(1 for w in words if w in negations)
        temporal_refs = {"ayer", "hoy", "mañana", "luego", "antes", "despues"}
        found_refs = [w for w in words if w in temporal_refs]
        return {
            "negation_count": negation_count,
            "pronoun_ratio": pronoun_ratio,
            "temporal_refs": found_refs,
            "tokens": token_count
        }
