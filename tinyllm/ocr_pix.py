#!/usr/bin/env python3
# pdf_to_text_with_pix2tex.py
"""
Pipeline para:
 - OCR (ocrmypdf o imagen->tesseract)
 - detección de regiones matemáticas
 - conversión region->LaTeX usando pix2tex (local)
 - generación de weird_chars.json y symbol mapping
 - export: texto plano con <|endoc|>, y JSONL index con ecuaciones/metadata

Uso:
    python pdf_to_text_with_pix2tex.py input.pdf
"""

import os
import re
import io
import json
import tempfile
import unicodedata
from typing import List, Tuple, Dict, Optional

from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import ocrmypdf
from pdfminer.high_level import extract_text

import cv2
import numpy as np

# Intentamos importar pix2tex (LatexOCR). Si no está instalado, declaramos flag.
try:
    from pix2tex.cli import LatexOCR
    PIX2TEX_AVAILABLE = True
except Exception:
    PIX2TEX_AVAILABLE = False

# ---------- CONFIG ----------
TESSERACT_LANG = 'spa+eng'
DPI = 300
USE_OCRMYPDF = True

# token especial para marcar fin de documento (añadir al texto plano exportado)
DOC_END_TOKEN = "<|endoc|>"

# heurística para detectar líneas con matemática
MATH_LINE_RE = re.compile(r'[=+\-*/×÷^_{}\\]|\\\(|\\\[|\\begin|\\alpha|\\beta|[0-9]\s*[=+\-]')

# mapping inicial (extiéndelo con los weird_chars.json generados)
SYMBOL_MAPPING = {
    '\uf0a8': '—',
    '': '÷',
    '': 'A',
    '': 'ν',
    '': 'α',
    '': ':',
    '': '-',
}

# regex para detectar número de ecuación (ej: (1.2) o (3) o (2.1.4))
EQ_ID_RE = re.compile(r'\(\s*\d+(?:\.\d+)*\s*\)')

# ---------- FUNCIONES ----------

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def apply_symbol_mapping(text: str, mapping: Dict[str,str]) -> str:
    for bad, good in mapping.items():
        text = text.replace(bad, good)
    return text

def ocr_pdf_to_searchable(input_pdf: str, output_pdf: str):
    print(f"[OCR] creando PDF con capa de texto: {output_pdf}")
    ocrmypdf.ocr(input_pdf, output_pdf, language=TESSERACT_LANG, dpi=DPI, force_ocr=True)
    print("[OCR] completado.")

def extract_text_layer(pdf_path: str) -> str:
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print("[WARN] pdfminer fallo:", e)
        return ""

def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    return convert_from_path(pdf_path, dpi=DPI)

def tesseract_image_to_text(img: Image.Image, lang=TESSERACT_LANG) -> str:
    return pytesseract.image_to_string(img, lang=lang)

def list_weird_characters(text: str) -> Dict[str,int]:
    counts = {}
    for ch in text:
        if ord(ch) > 127:
            counts[ch] = counts.get(ch, 0) + 1
    return dict(sorted(counts.items(), key=lambda x:-x[1]))

def find_math_regions_on_image(cv_img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # adaptative threshold can help with varied scans
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h_img, w_img = bw.shape
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < 400 or w < 30:
            continue
        # heuristic to prefer horizontal blocks (likely equations) and reasonable aspect ratio
        aspect = w / (h + 1e-9)
        if aspect < 0.8 and h < 30:
            continue
        boxes.append((x,y,w,h))
    boxes = sorted(boxes, key=lambda b: b[1])
    return boxes

# pix2tex wrapper
class Pix2TexWrapper:
    def __init__(self):
        if not PIX2TEX_AVAILABLE:
            raise RuntimeError("pix2tex package no encontrado. Instala 'pix2tex' y sus dependencias.")
        # init model once
        self.model = LatexOCR()

    def image_to_latex(self, pil_img: Image.Image) -> Tuple[Optional[str], float]:
        # devuelve (latex_string or None, confidence_estimate)
        try:
            latex = self.model.predict(pil_img)
            # pix2tex may not return confidence; we set a heuristic high confidence if non-empty
            conf = 0.9 if latex else 0.0
            return latex, conf
        except Exception as e:
            print("[WARN] pix2tex fallo:", e)
            return None, 0.0

def try_detect_eq_id_from_context(expanded_crop_text: str) -> Optional[str]:
    # busca (1.2) u otras formas en el texto alrededor del crop
    m = EQ_ID_RE.search(expanded_crop_text)
    if m:
        return m.group(0)
    return None

def save_crop_image(pil_img: Image.Image, out_dir: str, doc_base: str, page_idx: int, i: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{doc_base}_p{page_idx+1}_box{i+1}.png"
    path = os.path.join(out_dir, fname)
    pil_img.save(path)
    return path

def process_pdf(input_pdf: str, output_dir: str, use_ocrmypdf: bool = USE_OCRMYPDF, pix2tex_enabled: bool = PIX2TEX_AVAILABLE):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_pdf))[0]
    tmpdir = tempfile.mkdtemp(prefix=f"proc_{base}_")
    ocr_pdf_path = os.path.join(tmpdir, base + "_ocr.pdf")
    text_result = ""

    # 1) OCR layer
    if use_ocrmypdf:
        try:
            ocr_pdf_to_searchable(input_pdf, ocr_pdf_path)
            text_result = extract_text_layer(ocr_pdf_path)
        except Exception as e:
            print("[WARN] ocrmypdf fallo:", e)
            text_result = ""

    # 2) si no hay capa usable, usar imagen->tesseract full pages
    imgs = []
    if not text_result.strip():
        imgs = pdf_to_images(input_pdf)
    else:
        # aún así queremos las imágenes para detección de regiones
        imgs = pdf_to_images(input_pdf)

    # inicializar pix2tex si está activado
    p2t = None
    if pix2tex_enabled:
        try:
            p2t = Pix2TexWrapper()
        except Exception as e:
            print("[WARN] No se pudo inicializar pix2tex:", e)
            p2t = None

    # resultados agregados
    doc_text_pages = []
    eq_index = []  # lista de dicts que guardaremos en JSONL
    global_weird_chars = {}

    for p_idx, pil_img in enumerate(imgs):
        print(f"[PAGE] {p_idx+1}/{len(imgs)} procesando...")
        # texto de página (OCR)
        page_text = tesseract_image_to_text(pil_img)
        page_text = normalize_unicode(page_text)
        page_text = apply_symbol_mapping(page_text, SYMBOL_MAPPING)
        doc_text_pages.append(page_text)

        # detectar regiones
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        boxes = find_math_regions_on_image(cv_img)
        page_maths = []

        for i, (x,y,w,h) in enumerate(boxes):
            # expand bbox ligeramente para capturar labels cercanos
            margin = 10
            ex = max(0, x - margin)
            ey = max(0, y - margin)
            ew = min(cv_img.shape[1] - ex, w + 2*margin)
            eh = min(cv_img.shape[0] - ey, h + 2*margin)
            crop_cv = cv_img[ey:ey+eh, ex:ex+ew]
            pil_crop = Image.fromarray(cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB))

            # OCR del crop
            crop_text = pytesseract.image_to_string(pil_crop, lang=TESSERACT_LANG)
            crop_text = normalize_unicode(crop_text)
            crop_text = apply_symbol_mapping(crop_text, SYMBOL_MAPPING)

            # intentar con pix2tex si disponible
            latex = None
            confidence = 0.0
            if p2t is not None:
                latex, confidence = p2t.image_to_latex(pil_crop)

            # intentar reconocer eq id en el crop+alrededores
            expanded_text = crop_text
            # añadir un poco de contexto: extraer texto de la región superior/inferior cercana
            # para simplicidad concatenamos un recorte mayor y OCR
            # (ya hicimos expand bbox, así expanded_text tiene algo de contexto)
            eq_id = try_detect_eq_id_from_context(expanded_text)

            # guardar crop imagen
            crops_dir = os.path.join(output_dir, "crops")
            crop_path = save_crop_image(pil_crop, crops_dir, base, p_idx, i)

            entry = {
                "doc_id": base,
                "page": p_idx + 1,
                "bbox": [int(ex), int(ey), int(ew), int(eh)],
                "eq_id": eq_id,
                "latex": latex,
                "latex_confidence": confidence,
                "ocr_text": crop_text,
                "crop_path": crop_path,
                "source": "pix2tex" if (latex is not None) else "ocr",
            }
            page_maths.append(entry)
            eq_index.append(entry)

        # weird chars por pagina (para mapping)
        page_weird = list_weird_characters(page_text)
        # merge counts
        for ch,c in page_weird.items():
            global_weird_chars[ch] = global_weird_chars.get(ch, 0) + c

        # si se detectaron math regions guardamos metadata por pagina
        if page_maths:
            math_out = os.path.join(output_dir, f"{base}_page{p_idx+1}_math.json")
            with open(math_out, "w", encoding="utf-8") as f:
                json.dump(page_maths, f, ensure_ascii=False, indent=2)

    # unir texto de todas las páginas y añadir token endoc
    full_text = "\n\n".join(doc_text_pages).strip()
    full_text = full_text + "\n\n" + DOC_END_TOKEN + "\n"
    # aplicar mapping final a todo el texto
    full_text = apply_symbol_mapping(normalize_unicode(full_text), SYMBOL_MAPPING)

    text_out_path = os.path.join(output_dir, f"{base}_text.txt")
    with open(text_out_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    # escribir índice JSONL
    idx_out = os.path.join(output_dir, f"{base}_equations_index.jsonl")
    with open(idx_out, "w", encoding="utf-8") as f:
        for item in eq_index:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # escribir weird_chars global
    weird_out = os.path.join(output_dir, f"{base}_weird_chars.json")
    with open(weird_out, "w", encoding="utf-8") as f:
        json.dump({ch: {"count":c, "codepoint": hex(ord(ch))} for ch,c in global_weird_chars.items()},
                  f, ensure_ascii=False, indent=2)

    print("Proceso finalizado.")
    print("Texto guardado en:", text_out_path)
    print("Índice de ecuaciones (JSONL):", idx_out)
    print("Weird chars:", weird_out)
    print("Crops:", os.path.join(output_dir, "crops"))

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extrae texto y ecuaciones (pix2tex) de PDFs")
    parser.add_argument("pdf", help="archivo PDF de entrada")
    parser.add_argument("--out", "-o", help="directorio de salida", default="out_pdf_proc")
    parser.add_argument("--no-ocrmypdf", action="store_true", help="no usar ocrmypdf (usa imagen->tesseract)")
    parser.add_argument("--no-pix2tex", action="store_true", help="deshabilita pix2tex (usa solo OCR)")
    args = parser.parse_args()
    if args.no_ocrmypdf:
        USE = False
    else:
        USE = True
    PIX = (not args.no_pix2tex) and PIX2TEX_AVAILABLE
    if (not PIX) and (not PIX2TEX_AVAILABLE) and (not args.no_pix2tex):
        print("[INFO] pix2tex no disponible; la extracción LaTeX se omitirá. Instala pix2tex si quieres esa funcionalidad.")
    process_pdf(args.pdf, args.out, use_ocrmypdf=USE, pix2tex_enabled=PIX)
