import re
import sys

def parse_whatsapp_line(line: str):
    """Parsea una línea de export de WhatsApp."""
    # Formato: 19/7/24, 1:11 - Nombre: Mensaje
    pattern = r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2}) - ([^:]+): (.*)$"
    match = re.match(pattern, line)
    if match:
        return match.groups()
    return None

file_path = r"c:\Users\soyko\Documents\PreSickness\datos\paciente1_whatsapp.txt"
print(f"Reading {file_path}...")

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
except UnicodeDecodeError:
    print("UTF-8 read failed, trying latin-1")
    with open(file_path, 'r', encoding='latin-1') as f:
        lines = f.readlines()

print(f"Total lines: {len(lines)}")
parsed_count = 0
first_parsed = None
last_parsed = None

for i, line in enumerate(lines):
    line = line.strip()
    if not line: continue
    parsed = parse_whatsapp_line(line)
    if parsed:
        parsed_count += 1
        if not first_parsed: first_parsed = (i, parsed)
        last_parsed = (i, parsed)
    else:
        if i < 10:
            print(f"Line {i} NO MATCH: {line}")

print(f"Parsed {parsed_count} messages out of {len(lines)} lines.")
if first_parsed: print(f"First: {first_parsed}")
if last_parsed: print(f"Last: {last_parsed}")
