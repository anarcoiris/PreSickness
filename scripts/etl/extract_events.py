#!/usr/bin/env python3
"""
Extractor de eventos clínicos desde exports de WhatsApp.

Busca menciones de síntomas relevantes para EM (fatiga, dolor, brote, hospitalización)
y genera un archivo _events.csv con las fechas y secciones marcadas.

Uso:
    python extract_events.py datos/paciente1_whatsapp.txt --output datos/paciente1_events.csv
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE KEYWORDS
# ══════════════════════════════════════════════════════════════════════════════

SYMPTOM_KEYWORDS = {
    "relapse": {
        "keywords": [
            r"\bbrote\b", r"\bbrotes\b", r"\brecaída\b", r"\brecaer\b",
            r"\bempeorar\b", r"\bempeorando\b", r"\bcrisis\b"
        ],
        "severity_modifiers": {
            "severe": [r"\bhospital", r"\bingres", r"\burgencia", r"\bemergencia", r"\bgrave"],
            "moderate": [r"\bfuerte\b", r"\bmalo\b", r"\bpeor\b", r"\bfrustr"],
            "mild": []  # default
        }
    },
    "fatigue": {
        "keywords": [
            r"\bfatiga\b", r"\bcansad[oa]\b", r"\bagotad[oa]\b", r"\bexhaust",
            r"\bsin energía\b", r"\bno puedo\b", r"\bestoy muert[oa]\b",
            r"\bestoy frit[oa]\b", r"\bestoy débil\b"
        ],
        "severity_modifiers": {
            "severe": [r"\bno puedo moverme\b", r"\bno puedo levantarme\b"],
            "moderate": [r"\bmuy\b", r"\bdemasiado\b"],
            "mild": []
        }
    },
    "pain": {
        "keywords": [
            r"\bdolor\b", r"\bduele\b", r"\bdolores\b", r"\bpinchazos?\b",
            r"\bhormigueo\b", r"\bentumecid[oa]\b", r"\bquemazón\b",
            r"\bpunzadas?\b", r"\bcalambres?\b"
        ],
        "severity_modifiers": {
            "severe": [r"\binsoportable\b", r"\bterrible\b"],
            "moderate": [r"\bmucho\b", r"\bfuerte\b"],
            "mild": []
        }
    },
    "cognitive": {
        "keywords": [
            r"\bniebla mental\b", r"\bconfus[oa]\b", r"\bolvidé\b",
            r"\bno recuerdo\b", r"\bno me concentro\b", r"\bdificultad para pensar\b",
            r"\bcerebro.*lento\b", r"\bcognitiv[oa]\b", r"\bCI\b.*\bdecai"
        ],
        "severity_modifiers": {
            "severe": [r"\bno puedo pensar\b"],
            "moderate": [r"\bmucho\b"],
            "mild": []
        }
    },
    "vision": {
        "keywords": [
            r"\bvisión borrosa\b", r"\bno veo bien\b", r"\bdouble visión\b",
            r"\bneuritis\b", r"\bojo\b.*\bdolor\b"
        ],
        "severity_modifiers": {
            "severe": [r"\bceguera\b", r"\bno veo nada\b"],
            "moderate": [r"\bmuy borroso\b"],
            "mild": []
        }
    },
    "mobility": {
        "keywords": [
            r"\bno puedo caminar\b", r"\bdificultad.*caminar\b",
            r"\bpiernas.*pesad[ao]s\b", r"\bpierdo el equilibrio\b",
            r"\bme caí\b", r"\btropiezo\b", r"\bdébil.*piernas\b"
        ],
        "severity_modifiers": {
            "severe": [r"\bsilla de ruedas\b", r"\bno puedo moverme\b"],
            "moderate": [r"\bbastón\b", r"\bayuda\b"],
            "mild": []
        }
    },
    "hospitalization": {
        "keywords": [
            r"\bhospital\b", r"\bingres[aé]\b", r"\burgencias\b",
            r"\bemergencia\b", r"\bmédico\b.*\bfui\b"
        ],
        "severity_modifiers": {
            "severe": [r"\bgrave\b", r"\buci\b"],
            "moderate": [],
            "mild": []
        }
    },
    "medication": {
        "keywords": [
            r"\bmedicación\b", r"\btratamiento\b", r"\bpastillas?\b",
            r"\bcortisona\b", r"\binmunosupresor\b", r"\bclonazepam\b",
            r"\brebif\b", r"\btecfidera\b", r"\bocrevus\b", r"\bkesimpta\b"
        ],
        "severity_modifiers": {
            "severe": [],
            "moderate": [],
            "mild": []
        }
    }
}

# Mensajes a ignorar (sistema WhatsApp)
SYSTEM_PATTERNS = [
    r"<Multimedia omitido>",
    r"Los mensajes y las llamadas están cifrados",
    r"Se editó este mensaje",
    r"Eliminaste este mensaje",
    r"es un contacto",
    r"cambió el código de seguridad"
]


@dataclass
class EventMatch:
    """Representa un evento detectado."""
    date: str
    event_type: str
    severity: str
    message: str
    sender: str
    line_number: int
    context_before: list = field(default_factory=list)
    context_after: list = field(default_factory=list)


def parse_whatsapp_line(line: str) -> Optional[tuple]:
    """
    Parsea una línea de WhatsApp y extrae fecha, sender y texto.
    
    Formato esperado: DD/MM/YY, HH:MM - Sender: Mensaje
    """
    # Patrón para líneas de mensaje nuevas
    pattern = r"^(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2})\s*-\s*([^:]+):\s*(.*)$"
    match = re.match(pattern, line.strip())
    
    if match:
        date_str, time_str, sender, text = match.groups()
        return (date_str, time_str, sender.strip(), text.strip())
    return None


def is_system_message(text: str) -> bool:
    """Verifica si es un mensaje de sistema."""
    for pattern in SYSTEM_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def detect_severity(text: str, modifiers: dict) -> str:
    """Detecta la severidad basándose en modificadores."""
    text_lower = text.lower()
    
    for severity, patterns in modifiers.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return severity
    
    return "mild"


def extract_events(
    input_path: Path,
    target_sender: str = "<?>",
    include_counterparts: bool = True
) -> list[EventMatch]:
    """
    Extrae eventos clínicos del archivo de WhatsApp.
    
    Args:
        input_path: Ruta al archivo de WhatsApp
        target_sender: Marcador del paciente principal
        include_counterparts: Si incluir mensajes de otros participantes
    
    Returns:
        Lista de EventMatch encontrados
    """
    events = []
    lines = []
    
    # Leer todas las líneas
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📄 Leyendo {len(lines)} líneas...")
    
    # Parsear mensajes
    messages = []
    current_msg = None
    
    for i, line in enumerate(lines):
        parsed = parse_whatsapp_line(line)
        if parsed:
            if current_msg:
                messages.append(current_msg)
            date_str, time_str, sender, text = parsed
            current_msg = {
                "line": i + 1,
                "date": date_str,
                "time": time_str,
                "sender": sender,
                "text": text
            }
        elif current_msg and line.strip():
            # Línea de continuación
            current_msg["text"] += " " + line.strip()
    
    if current_msg:
        messages.append(current_msg)
    
    print(f"📨 Parseados {len(messages)} mensajes")
    
    # Buscar keywords
    for i, msg in enumerate(messages):
        # Filtrar mensajes de sistema
        if is_system_message(msg["text"]):
            continue
        
        # Filtrar por sender si es necesario
        is_target = msg["sender"] == target_sender
        if not is_target and not include_counterparts:
            continue
        
        text_lower = msg["text"].lower()
        
        # Buscar cada tipo de evento
        for event_type, config in SYMPTOM_KEYWORDS.items():
            for pattern in config["keywords"]:
                if re.search(pattern, text_lower):
                    severity = detect_severity(msg["text"], config["severity_modifiers"])
                    
                    # Obtener contexto
                    context_before = [m["text"][:100] for m in messages[max(0,i-2):i]]
                    context_after = [m["text"][:100] for m in messages[i+1:min(len(messages),i+3)]]
                    
                    event = EventMatch(
                        date=msg["date"],
                        event_type=event_type,
                        severity=severity,
                        message=msg["text"][:200],
                        sender=msg["sender"],
                        line_number=msg["line"],
                        context_before=context_before,
                        context_after=context_after
                    )
                    events.append(event)
                    break  # Un evento por tipo por mensaje
    
    return events


def aggregate_events_by_date(events: list[EventMatch]) -> dict:
    """Agrupa eventos por fecha y tipo."""
    aggregated = defaultdict(lambda: defaultdict(list))
    
    for event in events:
        # Normalizar fecha a formato ISO
        try:
            # Intentar varios formatos
            for fmt in ["%d/%m/%y", "%d/%m/%Y"]:
                try:
                    dt = datetime.strptime(event.date, fmt)
                    iso_date = dt.strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            else:
                iso_date = event.date
        except Exception:
            iso_date = event.date
        
        aggregated[iso_date][event.event_type].append(event)
    
    return aggregated


def generate_events_csv(events: list[EventMatch], output_path: Path):
    """Genera archivo CSV con eventos agregados."""
    aggregated = aggregate_events_by_date(events)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("date,event_type,severity,notes\n")
        
        for date in sorted(aggregated.keys()):
            for event_type, event_list in aggregated[date].items():
                # Tomar la severidad máxima del día
                severity_order = {"severe": 3, "moderate": 2, "mild": 1}
                max_severity = max(e.severity for e in event_list)
                
                # Generar notas
                notes = f"Auto-detectado: {len(event_list)} menciones"
                if event_list[0].sender != "<?>":
                    notes += f" (contraparte: {event_list[0].sender})"
                
                # Escapar comillas en notas
                notes = notes.replace('"', "'")
                
                f.write(f'{date},{event_type},{max_severity},"{notes}"\n')
    
    print(f"✅ Generado: {output_path}")


def generate_detailed_report(events: list[EventMatch], output_path: Path):
    """Genera reporte JSON detallado con contexto."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_events": len(events),
        "by_type": defaultdict(int),
        "events": []
    }
    
    for event in events:
        report["by_type"][event.event_type] += 1
        report["events"].append({
            "date": event.date,
            "type": event.event_type,
            "severity": event.severity,
            "sender": event.sender,
            "line": event.line_number,
            "message": event.message,
            "context_before": event.context_before,
            "context_after": event.context_after
        })
    
    report["by_type"] = dict(report["by_type"])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Reporte detallado: {output_path}")


def print_summary(events: list[EventMatch]):
    """Imprime resumen de eventos encontrados."""
    print("\n" + "="*60)
    print("📊 RESUMEN DE EVENTOS DETECTADOS")
    print("="*60)
    
    by_type = defaultdict(list)
    for event in events:
        by_type[event.event_type].append(event)
    
    for event_type, event_list in sorted(by_type.items()):
        print(f"\n🔹 {event_type.upper()}: {len(event_list)} menciones")
        
        # Mostrar primeras 3
        for event in event_list[:3]:
            severity_icon = {"severe": "🔴", "moderate": "🟡", "mild": "🟢"}.get(event.severity, "⚪")
            print(f"   {severity_icon} [{event.date}] {event.message[:80]}...")
        
        if len(event_list) > 3:
            print(f"   ... y {len(event_list) - 3} más")


def main():
    parser = argparse.ArgumentParser(
        description="Extrae eventos clínicos de exports de WhatsApp"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Archivo de WhatsApp (.txt)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Archivo de salida CSV (default: input_events.csv)"
    )
    parser.add_argument(
        "--target-sender", "-t",
        type=str,
        default="<?>",
        help="Identificador del paciente (default: <?>)"
    )
    parser.add_argument(
        "--include-counterparts",
        action="store_true",
        default=True,
        help="Incluir mensajes de contrapartes"
    )
    parser.add_argument(
        "--detailed-report",
        action="store_true",
        help="Generar reporte JSON detallado"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Archivo no encontrado: {args.input}")
        return 1
    
    output_path = args.output or args.input.with_name(
        args.input.stem.replace("_whatsapp", "") + "_events.csv"
    )
    
    # Extraer eventos
    events = extract_events(
        args.input,
        target_sender=args.target_sender,
        include_counterparts=args.include_counterparts
    )
    
    if not events:
        print("⚠️ No se encontraron eventos relevantes")
        return 0
    
    # Generar outputs
    generate_events_csv(events, output_path)
    
    if args.detailed_report:
        report_path = output_path.with_suffix('.json')
        generate_detailed_report(events, report_path)
    
    print_summary(events)
    
    return 0


if __name__ == "__main__":
    exit(main())
