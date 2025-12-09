import time
import numpy as np
from sentence_transformers import SentenceTransformer
import os

models = [
    "distiluse-base-multilingual-cased-v1",
    "paraphrase-multilingual-MiniLM-L12-v2", 
    "intfloat/multilingual-e5-small"
]

test_texts = [
    "Estoy muy cansado hoy, no tengo energía para nada.",
    "Me siento totalmente agotado y sin fuerzas.",  # Similar to 0
    "El día está soleado y los pájaros cantan.",  # Different from 0
    "Tengo ansiedad y me cuesta respirar.",       # Clinical relevance
    "Siento una opresión en el pecho y nerviosismo." # Similar to 3
]

print(f"{'Model':<40} | {'Load Time':<10} | {'Latency (ms)':<12} | {'Sim (0-1)':<10} | {'Sim (0-2)':<10} | {'Sim (3-4)':<10}")
print("-" * 110)

for model_name in models:
    try:
        # Load Time
        start_load = time.time()
        model = SentenceTransformer(model_name)
        load_time = time.time() - start_load
        
        # Warmup
        model.encode("warmup")
        
        # Latency Test (encode 1 sentence 50 times)
        start_lat = time.time()
        for _ in range(50):
            model.encode(test_texts[0])
        avg_latency = ((time.time() - start_lat) / 50) * 1000 # ms
        
        # Quality Test
        # e5 models usually need "query: " prefix for asymmetric tasks, 
        # but for similarity we can use just text or "passage: ". 
        # For simplicity and standard comparison we use raw text, 
        # but for e5 we might strictly need prefixes if performance is poor.
        # Let's try raw first as these are sentence-transformers wrappers.
        
        embs = model.encode(test_texts)
        
        def cos_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
        sim_01 = cos_sim(embs[0], embs[1]) # Fatigue similarity
        sim_02 = cos_sim(embs[0], embs[2]) # Fatigue vs Nature
        sim_34 = cos_sim(embs[3], embs[4]) # Anxiety similarity
        
        print(f"{model_name:<40} | {load_time:<10.2f} | {avg_latency:<12.2f} | {sim_01:<10.3f} | {sim_02:<10.3f} | {sim_34:<10.3f}")
        
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
