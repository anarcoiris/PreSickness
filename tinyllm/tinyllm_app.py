from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model import TinyGPTv2
from generation import GenerationConfig, generate
from tokenizers import Tokenizer
from typing import List, Optional
import os

app = FastAPI(title="TinyLLM Service")

# Model configuration
CHECKPOINT_PATH = os.getenv("TINYLLM_CHECKPOINT", "runs/small_model/ckpt_best.pt")
TOKENIZER_PATH = os.getenv("TINYLLM_TOKENIZER", "tokenizer.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HFTokenizerWrapper:
    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.vocab_size = tok.get_vocab_size()
    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text).ids
    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids)

model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer = HFTokenizerWrapper(tok)

    print(f"Loading model from {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
        print("Checkpoint not found, creating a fresh small model for demo/placeholder.")
        from model import create_small_model
        model = create_small_model(tokenizer.vocab_size)
    else:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model_config = ckpt.get('model_config', {})
        model = TinyGPTv2(
            vocab_size=model_config.get('vocab_size', tokenizer.vocab_size),
            block_size=model_config.get('block_size', 256),
            n_embd=model_config.get('n_embd', 256),
            n_layer=model_config.get('n_layer', 6),
            n_head=model_config.get('n_head', 8),
            use_rope=model_config.get('use_rope', True)
        )
        model.load_state_dict(ckpt['model_state_dict'])
    
    model.to(DEVICE)
    model.eval()
    print("Model ready.")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9

@app.post("/generate")
async def generate_text(req: GenerateRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    config = GenerationConfig(
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p
    )
    
    generated = generate(model, tokenizer, req.prompt, config, DEVICE)
    return {"generated_text": generated}

@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
