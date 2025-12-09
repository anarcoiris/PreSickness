#!/usr/bin/env python3
# llm_chunker_nano_only.py
# Purpose: send chunks of a .txt to gpt-5-nano for cleaning/transcription,
# allow testing with a single chunk, and mark chunks that should be reviewed by gpt-5-mini.
#
# Outputs:
#  - corpus_nano_clean.jsonl  (cleaned chunks from nano)
#  - mark_for_mini.json       (list of chunks flagged for review by mini)
#  - equations.json           (extracted equations)
#  - index.json               (chunk index)
#
# Usage examples (dry-run, no API calls):
#   python llm_chunker_nano_only.py --input mycorpus.txt --outdir runs/corpus_nano --chunk-tokens 1500 --test-first-chunk --dry-run
#
# To run real calls:
#   export OPENAI_API_KEY=...
#   pip install openai tiktoken tqdm python-dotenv
#   python llm_chunker_nano_only.py --input merged.txt --outdir corpus/merged --chunk-tokens 1500 --test-first-chunk
#
import os, sys, json, time, argparse, datetime, re
from pathlib import Path
from typing import Optional
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x

# Optional tokenization library
try:
    import tiktoken
except Exception:
    tiktoken = None

# Optional OpenAI (calls will be attempted only if openai is installed and dry_run is False)
try:
    import openai
except Exception:
    openai = None

import difflib
import hashlib

# ----------------- utils -----------------
def sha1_of_text(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def load_text(path: str, encoding='utf-8') -> str:
    with open(path, 'r', encoding=encoding, errors='replace') as f:
        return f.read()

def count_tokens(text: str, encoding_name: Optional[str]=None) -> int:
    if tiktoken and encoding_name:
        try:
            enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
        except Exception:
            pass
    if tiktoken:
        try:
            enc = tiktoken.encoding_for_model("gpt-4o-mini")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)

def compute_change_ratio(original: str, cleaned: str) -> float:
    # ratio = similarity in [0,1]; change_ratio = 1 - ratio
    try:
        ratio = difflib.SequenceMatcher(None, original, cleaned).ratio()
        return 1.0 - ratio
    except Exception:
        return 1.0

# ----------------- chunking -----------------
def chunk_text_by_tokens(text: str, max_completion_tokens: int = 1500, stride: Optional[int]=None, encoding_name: Optional[str]=None):
    if stride is None:
        stride = max_completion_tokens // 2
    if tiktoken:
        try:
            enc = tiktoken.encoding_for_model("gpt-4o-mini")
            ids = enc.encode(text)
            total = len(ids)
            chunks = []
            i = 0
            cid = 0
            while i < total:
                j = min(i + max_completion_tokens, total)
                tokens = ids[i:j]
                chunks.append({
                    "chunk_id": cid,
                    "text": enc.decode(tokens),
                    "start_token": i,
                    "end_token": j,
                    "tokens": j - i
                })
                cid += 1
                i = i + max_completion_tokens - stride
            return chunks
        except Exception:
            pass
    # fallback char-based
    avg = 4
    chunks = []
    start = 0
    cid = 0
    L = len(text)
    step = max(1, max_completion_tokens * avg - (stride * avg))
    while start < L:
        end = min(L, start + max_completion_tokens * avg)
        chunk_text = text[start:end]
        chunks.append({"chunk_id": cid, "text": chunk_text, "start_char": start, "end_char": end, "tokens": max(1, len(chunk_text)//avg)})
        cid += 1
        start = start + max_completion_tokens * avg - (stride * avg)
    return chunks

# ----------------- latex extraction (local) -----------------
LATEX_PATTERNS = [
    (re.compile(r'\$\$(.+?)\$\$', re.DOTALL), 'display_dollar'),
    (re.compile(r'\\begin\{equation\*\}(.+?)\\end\{equation\*\}', re.DOTALL), 'env_equation_star'),
    (re.compile(r'\\begin\{equation\}(.+?)\\end\{equation\}', re.DOTALL), 'env_equation'),
    (re.compile(r'\\\[([^\]]+?)\\\]', re.DOTALL), 'display_bracket'),
    (re.compile(r'\$(.+?)\$', re.DOTALL), 'inline_dollar'),
    (re.compile(r'\\\((.+?)\\\)', re.DOTALL), 'inline_paren'),
    (re.compile(r'\\begin\{align\*?\}(.+?)\\end\{align\*?\}', re.DOTALL), 'env_align'),
]
def extract_latex_equations(text: str):
    eqs = []
    for pat, kind in LATEX_PATTERNS:
        for m in pat.finditer(text):
            eq_raw = m.group(1).strip()
            eqs.append({"kind": kind, "equation": '\\n'.join(line.rstrip() for line in eq_raw.splitlines()), "span": (m.start(), m.end())})
    # dedupe
    seen = set(); out = []
    for e in eqs:
        k = (e['kind'], e['equation'])
        if k in seen: continue
        seen.add(k); out.append(e)
    return out

# ----------------- OpenAI call wrapper -----------------
def call_openai_clean(prompt: str, model: str="gpt-5-nano", max_completion_tokens: int=1024, temperature: float=1.0, retries:int=4, backoff:float=2.0):
    """
    Llamada a OpenAI compatible con openai>=1.0.0 (nueva API).
    Si la librería no está instalada o falla, devuelve {"success": False, "error": "..."}.
    """
    if openai is None:
        return {"success": False, "error": "openai package not installed."}

    messages = [
        {"role":"system","content":"You are a careful editor and LaTeX-aware transcriber. Return a strict JSON object with fields: clean_text (string), equations (list), needs_review (bool), confidence (float 0-1). No extra commentary."},
        {"role":"user","content":prompt}
    ]

    attempt = 0
    last_err = None

    while attempt < retries:
        try:
            # Prefer the new client interface (openai>=1.0.0)
            OpenAIClient = getattr(openai, "OpenAI", None)
            if OpenAIClient is not None:
                client = OpenAIClient()
                # new style: client.chat.completions.create(...)
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature
                )
                # Try robust extraction of content
                try:
                    content = resp.choices[0].message.content
                except Exception:
                    try:
                        # fallback dict-like
                        content = resp["choices"][0]["message"]["content"]
                    except Exception:
                        content = str(resp)
                return {"success": True, "content": content, "raw": resp}

            # If OpenAI client class not present, give a helpful error (do NOT call legacy ChatCompletion)
            # This avoids triggering the "ChatCompletion is no longer supported" error.
            return {"success": False, "error": "openai package installed but new OpenAI client not found. Run 'pip install --upgrade openai' or pin to 0.28 if you prefer legacy API."}

        except Exception as e:
            attempt += 1
            last_err = e
            wait = backoff ** attempt
            print(f"[openai] attempt {attempt} failed: {e}. waiting {wait:.1f}s")
            time.sleep(wait)

    return {"success": False, "error": f"max_retries_exceeded, last_error: {repr(last_err)}"}

CLEANUP_PROMPT_TEMPLATE = """Eres un corrector y transcriptor experto en contenido académico.
Tareas (devuelve únicamente un objeto JSON válido):
1) Corrige ortografía/gramática y normaliza notación matemática; deja las ecuaciones en LaTeX.
2) Devuelve JSON con:
   - clean_text: string con el texto corregido (no añadas explicaciones).
   - equations: lista de ecuaciones en LaTeX (strings).
   - needs_review: true/false si crees que el chunk requiere revisión humana / por un modelo mejor.
   - confidence: número entre 0.0 y 1.0 sobre la calidad del clean_text.
Si no puedes extraer todo en JSON, intenta devolver el fragmento JSON sólo. Fragmento: 
\"\"\"{chunk_text}\"\"\"
"""

# ----------------- pipeline -----------------
def process_file(input_path: str, outdir: str, chunk_tokens: int=1500, stride_tokens: Optional[int]=None, model="gpt-5-nano", max_api_tokens=1024, dry_run: bool=True, test_first_chunk: bool=False, mark_for_mini: bool=True, change_threshold:float=0.15, confidence_threshold:float=0.6, encoding_name: Optional[str]=None):
    p = Path(input_path)
    if not p.exists():
        raise RuntimeError("Input file not found: " + str(input_path))
    os.makedirs(outdir, exist_ok=True)
    raw = load_text(input_path)
    print(f"Loaded {len(raw)} chars; approx tokens: {count_tokens(raw, encoding_name)}")
    # simple section split: for now treat whole as one section; user can split externally if desired
    sections = [{"title":"full_text","start":0,"end":len(raw)}]
    index = []; equations_master = []; flagged = []
    out_jsonl = Path(outdir)/"corpus_nano_clean.jsonl"
    eq_file = Path(outdir)/"equations.json"
    index_file = Path(outdir)/"index.json"
    flag_file = Path(outdir)/"mark_for_mini.json"
    with out_jsonl.open("w", encoding="utf-8") as jf:
        for sec_id, sec in enumerate(sections):
            sec_text = raw[sec['start']:sec['end']].strip()
            sec_sha = sha1_of_text(sec_text)
            chunks = chunk_text_by_tokens(sec_text, max_completion_tokens=chunk_tokens, stride=stride_tokens, encoding_name=encoding_name)
            if test_first_chunk:
                chunks = chunks[:1]
            for ch in tqdm(chunks, desc=f"section{sec_id}"):
                chunk_text = ch['text']
                metadata = {
                    "source_file": str(p),
                    "section_id": sec_id,
                    "section_title": sec.get('title',"full_text"),
                    "section_sha1": sec_sha,
                    "chunk_id": ch['chunk_id'],
                    "approx_tokens": ch.get('tokens', count_tokens(chunk_text, encoding_name)),
                    "sha1": sha1_of_text(chunk_text),
                    "timestamp": datetime.datetime.utcnow().isoformat()+"Z"
                }
                local_eqs = extract_latex_equations(chunk_text)
                metadata['local_equations_count'] = len(local_eqs)
                prompt = CLEANUP_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
                if dry_run:
                    # do local-only processing
                    clean_text = chunk_text
                    model_eqs = []
                    model_needs = False
                    model_conf = 1.0
                else:
                    # call nano model
                    try:
                        resp = call_openai_clean(prompt=prompt, model=model, max_completion_tokens=max_api_tokens)
                    except Exception as e:
                        resp = {"success": False, "error": str(e)}
                    if resp.get("success"):
                        content = resp["content"]
                        # try to parse JSON object
                        parsed = None
                        try:
                            parsed = json.loads(content)
                        except Exception:
                            m = re.search(r'\{.*\}', content, re.DOTALL)
                            if m:
                                try:
                                    parsed = json.loads(m.group(0))
                                except Exception:
                                    parsed = None
                        if parsed:
                            clean_text = parsed.get("clean_text", "").strip() or chunk_text
                            model_eqs = parsed.get("equations", []) or []
                            model_needs = parsed.get("needs_review", False)
                            try:
                                model_conf = float(parsed.get("confidence", 1.0))
                            except Exception:
                                model_conf = 1.0
                        else:
                            clean_text = content.strip() or chunk_text
                            model_eqs = []
                            model_needs = False
                            model_conf = 1.0
                # compute local change ratio
                change_ratio = compute_change_ratio(chunk_text, clean_text)
                # decide if this chunk should be flagged for mini review
                flagged_reason = []
                flag = False
                if mark_for_mini:
                    if model_needs:
                        flag = True; flagged_reason.append("model_marked")
                    if model_conf is not None and model_conf < confidence_threshold:
                        flag = True; flagged_reason.append(f"low_confidence_{model_conf:.2f}")
                    if change_ratio > change_threshold:
                        flag = True; flagged_reason.append(f"big_change_{change_ratio:.3f}")
                # write outputs
                entry = {"text": clean_text, "metadata": {**metadata, "model_equations_count": len(model_eqs), "model_needs_review": bool(model_needs), "model_confidence": model_conf, "change_ratio": change_ratio, "flagged": flag, "flag_reasons": flagged_reason}}
                jf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                # collect equations
                for eq in local_eqs:
                    equations_master.append({"source": metadata, "kind": eq["kind"], "equation": eq["equation"]})
                for eq in model_eqs:
                    equations_master.append({"source": metadata, "kind": "model_extracted", "equation": eq})
                # index
                index.append({"source_file": str(p), "section_id": sec_id, "chunk_id": ch['chunk_id'], "approx_tokens": metadata["approx_tokens"], "sha1": metadata["sha1"], "flagged": flag, "flag_reasons": flagged_reason})
                if flag:
                    flagged.append({"chunk": metadata, "reasons": flagged_reason})
    # write auxiliary files
    with open(eq_file, "w", encoding="utf-8") as ef:
        json.dump(equations_master, ef, ensure_ascii=False, indent=2)
    with open(index_file, "w", encoding="utf-8") as idxf:
        json.dump(index, idxf, ensure_ascii=False, indent=2)
    with open(flag_file, "w", encoding="utf-8") as ff:
        json.dump(flagged, ff, ensure_ascii=False, indent=2)
    print("Done. Outputs:")
    print(" -", out_jsonl)
    print(" -", eq_file)
    print(" -", index_file)
    print(" -", flag_file)

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(description="Send .txt chunks to gpt-5-nano and mark chunks for mini review.")
    parser.add_argument("--input", "-i", required=True, help="Input text file")
    parser.add_argument("--outdir", "-o", default="runs/corpus_nano", help="Output directory")
    parser.add_argument("--chunk-tokens", type=int, default=1500, help="Chunk target in tokens")
    parser.add_argument("--stride-tokens", type=int, default=None, help="Stride tokens (default half)")
    parser.add_argument("--model", type=str, default="gpt-5-nano", help="Model for cleaning (default gpt-5-nano)")
    parser.add_argument("--max-api-tokens", type=int, default=1024, help="Max tokens requested from model response")
    parser.add_argument("--dry-run", action="store_true", help="Do not call OpenAI API (default: True for safety).")
    parser.add_argument("--test-first-chunk", action="store_true", help="Only send the first chunk (good for testing)")
    parser.add_argument("--no-mark-for-mini", dest="mark_for_mini", action="store_false", help="Disable marking for mini review")
    parser.add_argument("--change-threshold", type=float, default=0.15, help="Local change ratio above which to flag chunk")
    parser.add_argument("--confidence-threshold", type=float, default=0.6, help="Model confidence below which to flag chunk")
    parser.add_argument("--encoding", type=str, default=None, help="tiktoken encoding name (optional)")
    args = parser.parse_args()
    process_file(args.input, args.outdir, chunk_tokens=args.chunk_tokens, stride_tokens=args.stride_tokens, model=args.model, max_api_tokens=args.max_api_tokens, dry_run=args.dry_run, test_first_chunk=args.test_first_chunk, mark_for_mini=args.mark_for_mini, change_threshold=args.change_threshold, confidence_threshold=args.confidence_threshold, encoding_name=args.encoding)

if __name__ == '__main__':
    main()
