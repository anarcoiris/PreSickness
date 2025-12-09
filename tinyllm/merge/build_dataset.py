#!/usr/bin/env python3
"""
build_dataset.py

Construye un `corpus.txt` a partir de los JSONs (corpus.json, index.json, equations.json)
Diseñado para integrarse con el proyecto del usuario (main.py):
- Produce un corpus de texto plano UTF-8
- Ordena chunks según index.json (si disponible) o metadata.chunk_id
- Permite inyectar marcas de separador de documentos (por defecto <|doc|>)
- Opciones para incluir ecuaciones (inline / appended / none), normalizar texto,
  eliminar duplicados y filtrar por tamaño mínimo

Uso:
  python build_dataset.py --corpus-json corpus.json --out corpus.txt

Hecho para ser robusto: detecta JSONL o JSON array, gestiona metadatos, atiende
valores faltantes y realiza limpieza mínima (ftfy + unicodedata) si está disponible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import unicodedata
import re

try:
    import ftfy
except Exception:
    ftfy = None


def load_json_maybe_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Carga un archivo que puede ser JSONL (line-per-object) o un JSON array.
    Devuelve una lista de dicts.
    """
    text = path.read_text(encoding='utf-8')
    text_stripped = text.lstrip()
    if not text_stripped:
        return []

    # Detect JSONL: muchas líneas que empiezan con '{'
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1 and all(ln.strip().startswith('{') for ln in lines[:5]):
        items: List[Dict[str, Any]] = []
        for i, ln in enumerate(lines):
            try:
                items.append(json.loads(ln))
            except json.JSONDecodeError:
                # intenta reparar líneas problemáticas ignorando
                print(f"[warn] Ignorando línea JSON inválida en {path}: {i}", file=sys.stderr)
        return items

    # Si no JSONL, intenta parsear el archivo completo
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            # Caso en que el JSON es un dict con una clave que contiene la lista
            # Buscamos las claves habituales
            for k in ("items", "data", "chunks", "documents"):
                if k in parsed and isinstance(parsed[k], list):
                    return parsed[k]
            # Si es dict con keys numeradas, devolvemos sus values
            if all(isinstance(v, dict) for v in parsed.values()):
                return list(parsed.values())
            # Fallback: envolver en lista
            return [parsed]
    except json.JSONDecodeError as e:
        raise RuntimeError(f"No se pudo parsear JSON: {e}")


def load_index(path: Path) -> List[Dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(raw, list):
            return raw
        else:
            # Intenta extraer lista si viene en una clave
            for k in ("index", "chunks", "items"):
                if k in raw and isinstance(raw[k], list):
                    return raw[k]
            raise RuntimeError("index.json no contiene una lista de entradas")
    except Exception as e:
        raise


def canonical_key_from_metadata(md: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Extrae (chunk_id, sha1, source_file) desde metadata si existe."""
    chunk_id = None
    sha1 = None
    source_file = None
    if not md:
        return chunk_id, sha1, source_file
    if isinstance(md, dict):
        chunk_id = md.get('chunk_id') or md.get('id') or md.get('section_id')
        sha1 = md.get('sha1') or md.get('section_sha1')
        source_file = md.get('source_file')
    return chunk_id, sha1, source_file


def normalize_text(s: str, preserve_case: bool = True, remove_control: bool = True) -> str:
    if ftfy is not None:
        s = ftfy.fix_text(s)
    s = unicodedata.normalize('NFKC', s)

    # Reemplazos útiles
    s = s.replace('\u00A0', ' ')  # non-breaking space
    s = s.replace('\u200B', '')
    s = s.replace('\u2013', '-')
    s = s.replace('\u2014', '-')
    s = s.replace('\u2018', "'").replace('\u2019', "'")
    s = s.replace('\u201C', '"').replace('\u201D', '"')

    if remove_control:
        # mantener 
 y 	
        s = ''.join(ch for ch in s if unicodedata.category(ch)[0] != 'C' or ch in ('\n','\t'))

    # Normaliza múltiples espacios/lines
    s = re.sub(r'\s+',' ', s)
    s = s.strip()

    if not preserve_case:
        s = s.lower()

    return s


def build_lookup_from_corpus(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Construye un lookup por sha1 y por chunk_id (stringified) para acceso rápido."""
    lookup = {}
    for it in items:
        md = it.get('metadata') or {}
        chunk_id, sha1, source_file = canonical_key_from_metadata(md)
        # clave sha1
        if sha1:
            lookup[f"sha1:{sha1}"] = it
        if chunk_id is not None:
            lookup[f"chunk:{chunk_id}"] = it
    return lookup


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description='Construye corpus.txt desde corpus.json / index.json / equations.json')
    p.add_argument('--corpus-json', required=True, help='Archivo corpus.json (JSONL o JSON array)')
    p.add_argument('--index-json', help='Archivo index.json (opcional) para forzar orden)')
    p.add_argument('--equations-json', help='Archivo equations.json (opcional)')
    p.add_argument('--out', required=True, help='Salida: corpus.txt')

    p.add_argument('--doc-sep', default='<|doc|>', help='Token separador entre documentos (por defecto <|doc|>)')
    p.add_argument('--include-equations', choices=('none','inline','append'), default='none',
                   help='Cómo incluir ecuaciones: none, inline (al final del chunk), append (todas al final)')
    p.add_argument('--preserve-case', action='store_true', default=True, help='No bajar a minúsculas')
    p.add_argument('--no-preserve-case', dest='preserve_case', action='store_false', help='Bajar todo a minúsculas')
    p.add_argument('--normalize', action='store_true', default=True, help='Aplicar ftfy + unicode normalization si disponible')
    p.add_argument('--min-chars', type=int, default=30, help='Descartar chunks con menos chars')
    p.add_argument('--dedup', action='store_true', help='Eliminar duplicados exactos')
    p.add_argument('--verbose', '-v', action='store_true')

    args = p.parse_args(argv)

    corpus_path = Path(args.corpus_json)
    if not corpus_path.exists():
        print(f"[ERROR] corpus-json no encontrado: {corpus_path}", file=sys.stderr)
        return 2

    items = load_json_maybe_jsonl(corpus_path)
    if args.verbose:
        print(f"[info] Cargados {len(items)} items desde {corpus_path}")

    # Index (orden preferente)
    index_entries = None
    if args.index_json:
        index_path = Path(args.index_json)
        if index_path.exists():
            try:
                index_entries = load_index(index_path)
                if args.verbose:
                    print(f"[info] Cargados {len(index_entries)} entradas desde index.json")
            except Exception as e:
                print(f"[warn] No se pudo cargar index.json: {e}", file=sys.stderr)
                index_entries = None
        else:
            print(f"[warn] index.json especificado no encontrado: {index_path}", file=sys.stderr)

    # Ecuaciones
    equations_map = {}
    if args.equations_json:
        eq_path = Path(args.equations_json)
        if eq_path.exists():
            try:
                eqs = json.loads(eq_path.read_text(encoding='utf-8'))
                if isinstance(eqs, list):
                    for e in eqs:
                        # buscamos source.sha1 o source.chunk_id
                        src = e.get('source', {})
                        sha1 = src.get('sha1') or src.get('section_sha1')
                        chk = src.get('chunk_id')
                        if sha1:
                            equations_map[f"sha1:{sha1}"] = e.get('equation')
                        if chk is not None:
                            equations_map[f"chunk:{chk}"] = e.get('equation')
                else:
                    print('[warn] equations.json no es una lista. Ignorando.', file=sys.stderr)
            except Exception as e:
                print(f"[warn] No se pudo parsear equations.json: {e}", file=sys.stderr)
        else:
            print(f"[warn] equations.json no encontrado: {eq_path}", file=sys.stderr)

    # Lookup corpus items por sha1 / chunk
    lookup = build_lookup_from_corpus(items)

    ordered_items: List[Dict[str, Any]] = []
    used_keys = set()

    if index_entries:
        # Intentamos usar index para ordenar
        for ent in index_entries:
            key_matched = None
            # intentamos sha1
            sha1 = ent.get('sha1') or ent.get('section_sha1')
            chk = ent.get('chunk_id') or ent.get('chunk')
            if sha1 and f"sha1:{sha1}" in lookup:
                ordered_items.append(lookup[f"sha1:{sha1}"])
                used_keys.add(f"sha1:{sha1}")
                continue
            if chk is not None and f"chunk:{chk}" in lookup:
                ordered_items.append(lookup[f"chunk:{chk}"])
                used_keys.add(f"chunk:{chk}")
                continue
            # Fallback: si index entry tiene source_file+chunk, buscamos
            sf = ent.get('source_file')
            if sf:
                # buscar primera coincidencia con mismo source_file y chunk_id
                for it in items:
                    md = it.get('metadata') or {}
                    if md.get('source_file') == sf and md.get('chunk_id') == chk:
                        ordered_items.append(it)
                        used_keys.add(f"source:{sf}:chunk:{chk}")
                        break
        # Añadimos resto que no vinieron en index
        for it in items:
            md = it.get('metadata') or {}
            chunk_id, sha1, _ = canonical_key_from_metadata(md)
            key1 = f"sha1:{sha1}" if sha1 else None
            key2 = f"chunk:{chunk_id}" if chunk_id is not None else None
            if (key1 and key1 in used_keys) or (key2 and key2 in used_keys):
                continue
            ordered_items.append(it)
    else:
        # Orden por metadata.chunk_id si está, sino por orden original
        with_id = []
        without_id = []
        for it in items:
            md = it.get('metadata') or {}
            chunk_id, sha1, _ = canonical_key_from_metadata(md)
            if chunk_id is not None:
                try:
                    with_id.append((int(chunk_id), it))
                except Exception:
                    with_id.append((chunk_id, it))
            else:
                without_id.append(it)
        try:
            with_id.sort(key=lambda x: x[0])
            ordered_items = [it for _, it in with_id] + without_id
        except Exception:
            ordered_items = [it for _, it in with_id] + without_id

    # Construcción de textos
    docs: List[str] = []
    seen_texts = set()

    for it in ordered_items:
        text = it.get('text') or ''
        md = it.get('metadata') or {}
        chunk_id, sha1, _ = canonical_key_from_metadata(md)

        if args.normalize:
            text = normalize_text(text, preserve_case=args.preserve_case)

        if len(text) < args.min_chars:
            if args.verbose:
                print(f"[debug] Ignorando chunk (muy corto): chunk_id={chunk_id} len={len(text)}")
            continue

        if args.dedup:
            if text in seen_texts:
                if args.verbose:
                    print(f"[debug] Ignorando duplicado exacto: chunk_id={chunk_id}")
                continue
            seen_texts.add(text)

        # Añadir ecuación si corresponde y modo inline
        eq_text = None
        if args.include_equations == 'inline':
            key_sha = f"sha1:{sha1}" if sha1 else None
            key_chunk = f"chunk:{chunk_id}" if chunk_id is not None else None
            if key_sha and key_sha in equations_map:
                eq_text = equations_map[key_sha]
            elif key_chunk and key_chunk in equations_map:
                eq_text = equations_map[key_chunk]
            if eq_text:
                # añadimos la ecuación al final del chunk entre delimitadores LaTeX
                text = text + '\n\n[EQUATION]\n$$' + eq_text + '$$'

        docs.append(text)

    # Si include_equations==append, añadimos todas al final agrupadas
    if args.include_equations == 'append' and equations_map:
        append_lines = []
        for k, eq in equations_map.items():
            if not eq:
                continue
            append_lines.append(f"[{k}] \n$$ {eq} $$\n")
        if append_lines:
            docs.append('\n\n'.join(append_lines))

    # Construir corpus final con separador
    sep = f"\n\n{args.doc_sep}\n\n"
    final_text = sep.join(docs)

    # Guardar
    out_path = Path(args.out)
    out_path.write_text(final_text, encoding='utf-8')

    print(f"[info] Corpus guardado en: {out_path} -> {len(docs)} documentos, {out_path.stat().st_size} bytes")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
