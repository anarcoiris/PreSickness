#!/usr/bin/env python3
# merge_txts.py
import os, json, argparse
from glob import glob
from datetime import datetime
from pathlib import Path

DEFAULT_SPLITTER = "<|doc|>"

def sanitize_docid(name):
    # simple sanitize
    return "".join(c for c in name if c.isalnum() or c in "-_.").rstrip("._-")

def main(indir, outdir, splitter):
    os.makedirs(outdir, exist_ok=True)
    txt_files = sorted(glob(os.path.join(indir, "*.txt")))
    merged_txt_path = os.path.join(outdir, "merged.txt")
    merged_jsonl_path = os.path.join(outdir, "merged.jsonl")

    with open(merged_txt_path, "w", encoding="utf-8") as mf, open(merged_jsonl_path, "w", encoding="utf-8") as jf:
        for p in txt_files:
            name = Path(p).stem
            doc_id = sanitize_docid(name)
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            # write to merged text with splitter
            mf.write(txt + "\n\n" + splitter + "\n\n")
            # build json doc
            doc = {
                "doc_id": doc_id,
                "title": name,
                "text": txt + "\n\n" + splitter,
                "created_at": datetime.fromtimestamp(Path(p).stat().st_mtime).isoformat(),
                "source_file": os.path.abspath(p)
            }
            jf.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print("Hecho:")
    print(" - merged txt ->", merged_txt_path)
    print(" - merged jsonl ->", merged_jsonl_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("indir", help="directorio con .txt")
    p.add_argument("--out", "-o", default="merged_out", help="directorio salida")
    p.add_argument("--splitter", default=DEFAULT_SPLITTER, help="token separador")
    args = p.parse_args()
    main(args.indir, args.out, args.splitter)
