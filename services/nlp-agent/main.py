import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from schemas import ProcessingRequest, ProcessingResponse, EmbeddingData, SymptomScore, LinguisticMeta
from model import NlpEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nlp-agent")

engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Initializing NLP Engine...")
    engine = NlpEngine()
    logger.info("NLP Engine Ready.")
    yield
    # Cleanup if needed
    pass

app = FastAPI(title="EM Predictor - NLP Agent", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": engine.model_version if engine else "not_loaded"}

@app.post("/v1/process", response_model=ProcessingResponse)
async def process_text(request: ProcessingRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Compute Hash
        text_hash = engine.compute_hash(request.text)
        
        # Embed
        vector, dim = engine.embed(request.text)
        
        # Predict Heads
        scores_raw = engine.predict_heads(vector, request.text)
        
        # Linguistic Meta
        meta_raw = engine.extract_linguistic_meta(request.text)
        
        # Construct Response
        symptom_scores = {
            k: SymptomScore(**v) for k, v in scores_raw.items()
        }
        
        linguistic_meta = LinguisticMeta(**meta_raw)
        
        duration = (time.time() - start_time) * 1000
        
        return ProcessingResponse(
            message_id=request.message_id,
            user_id=request.user_id,
            timestamp=request.timestamp,
            language=request.language_hint,
            text_hash=text_hash,
            embeddings=EmbeddingData(model=engine.model_version, dim=dim, vector=vector),
            symptom_scores=symptom_scores,
            linguistic_meta=linguistic_meta,
            model_version=engine.model_version,
            processing_time_ms=duration
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
