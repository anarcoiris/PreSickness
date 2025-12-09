#!/usr/bin/env python3
# parse_chat_export.py
"""
Uso:
  python parse_chat_export.py export.json --out outdir --chunk --tokenizer gpt2

Genera:
 - outdir/<base>_clean.txt           # texto plano para TTS
 - outdir/<base>_train.jsonl         # jsonl con {doc_id, title, text, metadata,...}
 - outdir/<base>_weird_chars.json    # chars no-ascii contados
 - opcional: outdir/chunks_<base>.jsonl (si --chunk y transformers instalado)
"""

import os
import re
import json
import argparse
from collections import Counter
from typing import Dict, Any, List
import unicodedata

DOC_END_TOKEN = "<|endoc|>"

# regexs
TS_RE = re.compile(r'^\s*\[\s*\d{1,2}:\d{2}\s*,\s*\d{1,2}/\d{1,2}/\d{2,4}\s*\]\s*')  # [hh:mm, d/m/yyyy]
TS_GENERAL_RE = re.compile(r'^\s*\[\s*\d{1,2}:\d{2}\s*,\s*[^]]+\]\s*')  # más permisivo
SPEAKER_RE = re.compile(r'^\s*([^:]{1,80}?)\s*:\s*(.*)$')  # "Nombre...: texto"


# caracteres inválidos en Windows: <>:"/\\|?*  (y evitar nombres terminados en espacio o punto)
INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

def sanitize_filename(name: str, max_len: int = 200) -> str:
    """
    Devuelve un nombre de fichero seguro:
     - normaliza unicode (NFKC),
     - reemplaza caracteres invalidos por '_',
     - sustituye múltiples guiones bajos por uno,
     - recorta longitud y quita espacios/puntos al final.
    """
    if not name:
        return "file"
    # normalizar
    name = unicodedata.normalize("NFKC", name)
    # reemplazar caracteres invalidos
    name = INVALID_CHARS_RE.sub("_", name)
    # reemplazar rutas relativas accidentales
    name = name.replace(os.sep, "_")
    # compactar guiones bajos repetidos
    name = re.sub(r'[_\s]+', "_", name).strip("_")
    # limitar longitud
    if len(name) > max_len:
        name = name[:max_len]
    # evitar que termine en espacio o punto
    name = name.rstrip(" .")
    if not name:
        return "file"
    return name


def load_input(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_root(mapping: Dict[str,Any]) -> str:
    # prefer 'client-created-root', si no busca el que parent is None
    if 'client-created-root' in mapping:
        return 'client-created-root'
    for k,v in mapping.items():
        if v.get('parent') is None:
            return k
    # fallback: cualquier key
    return next(iter(mapping.keys()))

def traverse_ordered(mapping: Dict[str,Any], node_id: str, out: List[str]):
    """Recorrido pre-order siguiendo children list si existe."""
    node = mapping.get(node_id)
    if node is None:
        return
    out.append(node_id)
    for ch in node.get('children', []) or []:
        traverse_ordered(mapping, ch, out)

def extract_message_text(node_message: Dict[str,Any]) -> str:
    """Extrae texto de message.content.parts dependiendo de content_type."""
    if not node_message:
        return ""
    content = node_message.get('content', {})
    ctype = content.get('content_type')
    parts = content.get('parts') or []
    # parts puede contener strings o dicts (multimodal). Normalizamos
    out_parts = []
    for p in parts:
        if isinstance(p, str):
            out_parts.append(p)
        elif isinstance(p, dict):
            # multimodal_text parts: buscaremos "text" o descriptions
            # si tiene 'content_type' internamente, ignoramos imagenes y tomamos texto si existe
            txt = p.get('text') or p.get('alt') or p.get('description') or ""
            if txt:
                out_parts.append(txt)
    return "\n".join(out_parts).strip()

def clean_whatsapp_like_lines(raw_text: str) -> List[str]:
    """
    Transforma líneas con timestamps y "Name: message" en "Name (quien lo dice): message".
    Si no hay timestamp, intenta usar SPEAKER_RE o deja la línea tal cual.
    """
    lines = []
    for ln in raw_text.splitlines():
        s = ln.strip()
        if not s:
            continue
        # si comienza con timestamp [hh:mm, dd/...]
        if TS_GENERAL_RE.match(s):
            s2 = TS_GENERAL_RE.sub('', s).strip()
            m = SPEAKER_RE.match(s2)
            if m:
                speaker = m.group(1).strip()
                message = m.group(2).strip()
                lines.append(f"{speaker} (quien lo dice):  \n{message}")
            else:
                # si no hay "Name:", puede ser "Name message" o un texto simple
                lines.append(s2)
        else:
            # no timestamp: intentar coger speaker de la propia línea "Nombre: texto"
            m = SPEAKER_RE.match(s)
            if m:
                speaker = m.group(1).strip()
                message = m.group(2).strip()
                lines.append(f"{speaker} (quien lo dice):  \n{message}")
            else:
                lines.append(s)
    return lines

def build_clean_conversation(mapping: Dict[str,Any]) -> (str, List[str], Counter):
    root = find_root(mapping)
    order = []
    traverse_ordered(mapping, root, order)
    conversation_lines = []
    weird_counter = Counter()

    for node_id in order:
        node = mapping.get(node_id, {})
        msg = node.get('message')
        if not msg:
            continue
        # extra info about author
        author = msg.get('author', {}).get('role') if isinstance(msg.get('author', {}), dict) else None
        create_time = msg.get('create_time')
        raw = extract_message_text(msg)
        if not raw:
            continue
        # Normalizar espacios y unicode básico
        raw = raw.replace('\r\n', '\n').replace('\r', '\n').strip()
        # Produce cleaned lines: si raw contiene muchas líneas tipo whatsapp, procesarlas
        cleaned_lines = clean_whatsapp_like_lines(raw)
        # si no resultó detectarse, fallback: prefix with author role
        if not cleaned_lines:
            label = author or "user"
            cleaned_lines = [f"{label}: {raw}"]
        # append to conversation with spacing
        for cl in cleaned_lines:
            conversation_lines.append(cl)
            # count weird chars
            for ch in cl:
                if ord(ch) > 127:
                    weird_counter[ch] += 1

    # join with double newline and append end token
    full_text = "\n\n".join(conversation_lines).strip() + "\n\n" + DOC_END_TOKEN + "\n"
    return full_text, conversation_lines, weird_counter

def save_outputs(base_name: str, out_dir: str, full_text: str, conversation_lines: List[str], weird_counter: Counter, meta: Dict[str,Any]):
    os.makedirs(out_dir, exist_ok=True)
    # sanitizamos base_name para evitar caracteres invalidos
    base_safe = sanitize_filename(base_name)
    txt_path = os.path.join(out_dir, f"{base_safe}_clean.txt")
    with open(txt_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(full_text)
    # write training jsonl (one doc)
    jsonl_path = os.path.join(out_dir, f"{base_safe}_train.jsonl")
    train_obj = {
        "doc_id": base_safe,
        "title": meta.get("title"),
        "text": full_text,
        "speakers": list({   # set unique speakers extracting labels from lines
            l.split(" (quien lo dice):")[0] if " (quien lo dice):" in l else l.split(":")[0]
            for l in conversation_lines
        }),
        "created_at": meta.get("create_time"),
        "source": "chat_export",
        "metadata": meta
    }
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(train_obj, ensure_ascii=False) + "\n")

    # weird chars
    weird_path = os.path.join(out_dir, f"{base_safe}_weird_chars.json")
    with open(weird_path, "w", encoding="utf-8") as f:
        json.dump({ch: {"count": c, "codepoint": hex(ord(ch))} for ch,c in weird_counter.items()},
                  f, ensure_ascii=False, indent=2)

    print("Guardado:")
    print(" - Texto TTS:", txt_path)
    print(" - JSONL entrenamiento:", jsonl_path)
    print(" - Weird chars:", weird_path)
    return txt_path, jsonl_path, weird_path


# optional chunking using transformers tokenizer
def chunk_and_write(jsonl_in_path: str, out_dir: str, tokenizer_name: str="gpt2",
                    block_size:int=1024, stride:int=512):
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        print("[INFO] transformers no instalado, omitiendo chunking:", e)
        return None
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    with open(jsonl_in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    chunks_out = os.path.join(out_dir, "chunks_"+os.path.basename(jsonl_in_path))
    with open(chunks_out, "w", encoding="utf-8") as outf:
        for ln in lines:
            obj = json.loads(ln)
            text = obj["text"]
            ids = tokenizer(text, return_attention_mask=False)["input_ids"]
            n = len(ids)
            i = 0
            while i < n:
                window = ids[i:i+block_size]
                if len(window) < 2:
                    break
                chunk_text = tokenizer.decode(window, skip_special_tokens=False)
                chunk_obj = {
                    "doc_id": obj["doc_id"],
                    "text": chunk_text,
                    "meta": {"orig_len_tokens": n, "start_token": i}
                }
                outf.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")
                i += (block_size - stride)
    print("Chunks escritos en:", chunks_out)
    return chunks_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="archivo JSON export (lista de conversaciones)")
    parser.add_argument("--out", "-o", help="directorio salida", default="parsed_out")
    parser.add_argument("--chunk", action="store_true", help="hacer chunking (requiere transformers)")
    parser.add_argument("--tokenizer", default="gpt2", help="tokenizer HF para chunking (por ejemplo gpt2)")
    args = parser.parse_args()

    data = load_input(args.input)
    if not isinstance(data, list):
        print("Se esperaba una lista de conversaciones en el JSON.")
        data = [data]

    for idx, conv in enumerate(data):
        mapping = conv.get("mapping") or {}
        title = conv.get("title") or f"conversation_{idx}"
        full_text, conv_lines, weird = build_clean_conversation(mapping)
        meta = {
            "create_time": conv.get("create_time"),
            "update_time": conv.get("update_time"),
            "conversation_id": conv.get("conversation_id") or conv.get("id") or f"conv_{idx}"
        }
        base = title.replace(" ", "_")[:80]
        txt_path, jsonl_path, weird_path = save_outputs(base, args.out, full_text, conv_lines, weird, meta)
        if args.chunk:
            chunk_and_write(jsonl_path, args.out, tokenizer_name=args.tokenizer)

if __name__ == "__main__":
    main()
