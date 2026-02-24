from pathlib import Path
import os

filepath = Path(r"c:\Users\soyko\Documents\PreSickness\services\unified_app\analysis.py")
project_root = filepath.resolve().parent.parent.parent
data_path = project_root / "data" / "processed"
target_path = data_path / "paciente1"

print(f"File: {filepath}")
print(f"Project Root: {project_root}")
print(f"Data Path: {data_path}")
print(f"Target Path: {target_path}")
print(f"Exists: {target_path.exists()}")

if target_path.exists():
    print("Files in target_path:")
    for f in target_path.iterdir():
        print(f" - {f.name}")
