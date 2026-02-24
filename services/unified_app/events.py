"""
Endpoints de Eventos Clínicos - MS-Predictor
CRUD completo + validación de clusters + importación CSV
"""
import io
import re
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from pydantic import BaseModel, Field
import csv
import io
import logging
from dependencies import get_current_patient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["events"])


# ============================================================================
# ENUMS Y SCHEMAS
# ============================================================================

class EventType(str, Enum):
    SYMPTOM_ONSET = "symptom_onset"
    CONFIRMED_RELAPSE = "confirmed_relapse"
    MEDICATION_START = "medication_start"
    HOSPITAL_VISIT = "hospital_visit"
    DOCTOR_APPOINTMENT = "doctor_appointment"


class EventSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class EventSource(str, Enum):
    MANUAL = "manual"
    AUTO_DETECTED = "auto_detected"
    IMPORTED = "imported"


class ValidationRole(str, Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"


class ClusterStatus(str, Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"


# Request/Response Schemas
class EventCreate(BaseModel):
    event_date: datetime
    event_type: EventType
    severity: Optional[EventSeverity] = None
    notes: Optional[str] = None
    medication_start_date: Optional[datetime] = None


class EventUpdate(BaseModel):
    event_date: Optional[datetime] = None
    event_type: Optional[EventType] = None
    severity: Optional[EventSeverity] = None
    notes: Optional[str] = None
    medication_start_date: Optional[datetime] = None


class EventResponse(BaseModel):
    id: UUID
    patient_id: str
    event_date: datetime
    event_type: EventType
    severity: Optional[EventSeverity]
    notes: Optional[str]
    medication_start_date: Optional[datetime]
    source: EventSource
    validated_by: Optional[str]
    validated_at: Optional[datetime]
    validation_role: Optional[ValidationRole]
    requires_retraining: bool
    created_at: datetime
    updated_at: datetime


class ClusterResponse(BaseModel):
    id: UUID
    patient_id: str
    start_date: datetime
    end_date: datetime
    peak_date: datetime
    total_signals: int
    unique_types: int
    max_severity: Optional[str]
    severity_score: float
    density: float
    is_probable_relapse: bool
    confidence: Optional[float]
    status: ClusterStatus
    created_at: datetime


class ValidateClusterRequest(BaseModel):
    event_date: datetime
    event_type: EventType = EventType.CONFIRMED_RELAPSE
    severity: Optional[EventSeverity] = None
    notes: Optional[str] = None
    medication_start_date: Optional[datetime] = None


class RejectClusterRequest(BaseModel):
    reason: str = Field(min_length=5, max_length=500)


class EventsStatsResponse(BaseModel):
    total_events: int
    confirmed_relapses: int
    medication_starts: int
    pending_clusters: int
    pending_retraining: int
    last_event_date: Optional[datetime]


class LabelSettingsUpdate(BaseModel):
    horizons: Optional[List[int]] = None
    label_event_types: Optional[List[str]] = None
    censor_days_before_end: Optional[int] = None
    use_auto_clusters: Optional[bool] = None
    auto_cluster_min_confidence: Optional[float] = None


class LabelSettingsResponse(BaseModel):
    patient_id: str
    horizons: List[int]
    label_event_types: List[str]
    censor_days_before_end: int
    use_auto_clusters: bool
    auto_cluster_min_confidence: float
    pending_changes: int
    last_labels_generated_at: Optional[datetime]


class ImportPreviewResponse(BaseModel):
    valid_events: int
    invalid_events: int
    errors: List[str]
    preview: List[dict]


class RetrainingStatus(BaseModel):
    pending_changes: int
    requires_retraining: bool
    last_trained_at: Optional[datetime]


# ============================================================================
# BASE DE DATOS EN MEMORIA (Prototipo)
import db

# ============================================================================
# PARSERS
# ============================================================================

def parse_whatsapp_line(line: str):
    """Parsea una línea de export de WhatsApp."""
    # Limpiar caracteres de control Unicode invisibles (LTR marks, BOM, zero-width, etc.)
    import unicodedata
    cleaned = ''.join(
        c for c in line
        if unicodedata.category(c) not in ('Cc', 'Cf') or c in '\n\r\t'
    )
    cleaned = cleaned.strip()
    
    # Formato: 19/7/24, 1:11 - Nombre: Mensaje
    pattern = r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2}) - ([^:]+): (.*)$"
    match = re.match(pattern, cleaned)
    if match:
        date_str, time_str, sender, content = match.groups()
        # Convertir DD/MM/YY a YYYY-MM-DD
        d, m, y = date_str.split("/")
        if len(y) == 2: y = "20" + y
        iso_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}T{time_str.zfill(5)}:00Z"
        return {
            "date": iso_date,
            "sender": sender,
            "content": content
        }
    return None

# ============================================================================
# PERSISTENCIA Y CARGA DE DATOS
# ============================================================================

async def load_initial_data_async():
    """Carga datos existentes desde CSV y TXT al arrancar el servidor."""
    patient_id = "paciente1"
    
    # 1. Determinar ruta base de 'datos'
    from pathlib import Path
    base_paths = [
        Path("datos"),
        Path("../../datos"),
        Path(__file__).parent.parent.parent / "datos",
        Path("/app/datos"), # Para Docker
    ]
    
    datos_path = None
    for p in base_paths:
        if p.exists():
            datos_path = p
            break
            
    if not datos_path:
        print("[WARN] No se encontró directorio 'datos' en ninguna de las rutas probadas")
        return

    print(f"[INFO] Cargando datos iniciales desde: {datos_path.absolute()}")

    # 2. Cargar eventos clínicos si están vacíos
    stats = await db.get_event_stats(patient_id)
    if stats["total_events"] == 0:
        events_csv = datos_path / "paciente1_events.csv"
        if events_csv.exists():
            try:
                with open(events_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    imported = 0
                    for row in reader:
                        event_type = row.get("event_type", "relapse")
                        if event_type == "relapse":
                            event_type = "confirmed_relapse"
                            
                        await db.create_event({
                            "patient_id": patient_id,
                            "event_date": datetime.fromisoformat(row["date"]),
                            "event_type": event_type,
                            "severity": row.get("severity"),
                            "notes": row.get("notes"),
                            "source": EventSource.IMPORTED
                        })
                        imported += 1
                print(f"[INFO] {imported} eventos cargados desde {events_csv.name}")
            except Exception as e:
                print(f"[ERROR] Fallo al cargar eventos: {e}")

    # 3. Cargar mensajes de WhatsApp si están vacíos
    msg_count = await db.count_raw_messages(patient_id)
    if msg_count == 0:
        wa_txt = datos_path / "paciente1_whatsapp.txt"
        if wa_txt.exists():
            try:
                print(f"[INFO] Parseando mensajes de WhatsApp desde {wa_txt.name}...")
                parsed_messages = []
                current_msg = None
                
                with open(wa_txt, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        
                        parsed = parse_whatsapp_line(line)
                        if parsed:
                            if current_msg:
                                parsed_messages.append(current_msg)
                            current_msg = {
                                "date": parsed["date"],
                                "content": parsed["content"],
                                "metadata": {"sender": parsed["sender"]}
                            }
                        elif current_msg:
                            # Continuación de mensaje multilínea
                            current_msg["content"] += "\n" + line
                
                if current_msg:
                    parsed_messages.append(current_msg)
                
                if parsed_messages:
                    print(f"[INFO] Insertando {len(parsed_messages)} mensajes en batch...")
                    # Insertar en bloques para no saturar memoria o conexión
                    chunk_size = 1000
                    for i in range(0, len(parsed_messages), chunk_size):
                        chunk = parsed_messages[i:i + chunk_size]
                        await db.batch_store_raw_messages(patient_id, chunk)
                    print(f"[INFO] Importación exitosa de {len(parsed_messages)} mensajes.")
                    
            except Exception as e:
                print(f"[ERROR] Fallo al cargar WhatsApp: {e}")

    # 4. Inicializar settings si no existen
    await db.upsert_label_settings(patient_id, {
        "horizons": [7, 14, 30],
        "label_event_types": ["confirmed_relapse", "medication_start"]
    })


def get_patient_id(patient: dict) -> str:
    """Extrae ID de paciente del objeto de usuario."""
    return patient.get("user_id_hash", "anonymous")


# ============================================================================
# ENDPOINTS - EVENTOS CRUD
# ============================================================================

@router.post("/", response_model=EventResponse, status_code=201)
async def create_event(
    data: EventCreate,
    patient: dict = Depends(get_current_patient)  # Proper auth
):
    """Crea un nuevo evento clínico."""
    patient_id = get_patient_id(patient)
    
    event_data = data.model_dump()
    event_data.update({
        "patient_id": patient_id,
        "source": EventSource.MANUAL,
        "validated_by": patient_id,
        "validated_at": datetime.now(timezone.utc),
        "validation_role": ValidationRole.PATIENT,
        "requires_retraining": True
    })
    
    event = await db.create_event(event_data)
    
    # Incrementar contador de cambios pendientes
    await db.increment_pending_changes(patient_id)
    
    return event


@router.get("/", response_model=List[EventResponse])
async def list_events(
    patient: dict = Depends(get_current_patient),
    # Filtros ya no se aplican en memoria, pero db.get_events soporta offset/limit
    # TODO: Implementar filtros en SQL si es necesario
    limit: int = Query(default=50, le=200),
    offset: int = 0,
):
    """Lista eventos del paciente."""
    patient_id = get_patient_id(patient)
    events = await db.get_events(patient_id, limit=limit, offset=offset)
    return events


# ============================================================================
# ENDPOINTS - ESTADÍSTICAS Y CONFIGURACIÓN (rutas estáticas primero)
# ============================================================================

@router.get("/stats", response_model=EventsStatsResponse)
async def get_events_stats(
    patient: dict = Depends(get_current_patient),
):
    """Obtiene estadísticas de eventos del paciente."""
    patient_id = get_patient_id(patient)
    stats = await db.get_event_stats(patient_id)
    return EventsStatsResponse(**stats)


@router.get("/settings", response_model=LabelSettingsResponse)
async def get_label_settings(
    patient: dict = Depends(get_current_patient),
):
    """Obtiene configuración de labels del paciente."""
    patient_id = get_patient_id(patient)
    settings = await db.get_label_settings(patient_id)
    
    if not settings:
        # Default settings if none exist
        return LabelSettingsResponse(
            patient_id=patient_id,
            horizons=[7, 14, 30],
            label_event_types=["confirmed_relapse", "medication_start"]
        )
        
    return LabelSettingsResponse(**settings)


@router.put("/settings", response_model=LabelSettingsResponse)
async def update_label_settings(
    data: LabelSettingsUpdate,
    patient: dict = Depends(get_current_patient),
):
    """Actualiza configuración de labels."""
    patient_id = get_patient_id(patient)
    update_data = data.model_dump(exclude_unset=True)
    settings = await db.upsert_label_settings(patient_id, update_data)
    return LabelSettingsResponse(**settings)


@router.get("/retraining-status", response_model=RetrainingStatus)
async def get_retraining_status(
    patient: dict = Depends(get_current_patient),
):
    """Verifica si hay cambios pendientes que requieren re-entrenamiento."""
    patient_id = get_patient_id(patient)
    settings = await db.get_label_settings(patient_id)
    pending = settings.get("pending_changes", 0) if settings else 0
    
    return RetrainingStatus(
        pending_changes=pending,
        requires_retraining=pending > 0,
        last_trained_at=settings.get("last_labels_generated_at") if settings else None,
    )


@router.post("/trigger-retraining", response_model=dict)
async def trigger_retraining(
    background_tasks: BackgroundTasks,
    patient: dict = Depends(get_current_patient),
):
    """Dispara regeneración de labels y re-entrenamiento."""
    from fastapi import BackgroundTasks
    from tasks import run_retraining_pipeline
    
    patient_id = get_patient_id(patient)
    job_id = uuid4()
    
    # Run in background
    background_tasks.add_task(run_retraining_pipeline, patient_id, job_id)
    
    # Reset pending changes immediately (since pipeline is triggered)
    await db.reset_pending_changes(patient_id)
    
    return {
        "status": "triggered",
        "message": "Re-entrenamiento iniciado. Recibirás notificación cuando termine.",
        "job_id": str(job_id),
    }


# ============================================================================
# ENDPOINTS - CLUSTERS (rutas estáticas)
# ============================================================================

@router.get("/clusters", response_model=List[ClusterResponse])
async def list_clusters(
    patient: dict = Depends(get_current_patient),
    status: Optional[ClusterStatus] = None,
):
    """Lista clusters auto-detectados."""
    patient_id = get_patient_id(patient)
    status_str = status.value if status else None
    clusters = await db.get_clusters(patient_id, status=status_str)
    return clusters


@router.post("/clusters/{cluster_id}/validate", response_model=EventResponse)
async def validate_cluster(
    cluster_id: UUID,
    data: ValidateClusterRequest,
    patient: dict = Depends(get_current_patient),
):
    """Valida un cluster y crea evento asociado."""
    patient_id = get_patient_id(patient)
    
    cluster = await db.get_cluster_by_id(cluster_id, patient_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster no encontrado")
    
    if cluster["status"] != ClusterStatus.PENDING:
        raise HTTPException(status_code=400, detail="Cluster ya procesado")
    
    # Crear evento
    event_data = {
        "patient_id": patient_id,
        "event_date": data.event_date,
        "event_type": data.event_type,
        "severity": data.severity,
        "notes": data.notes or f"Validado desde cluster {cluster_id}",
        "medication_start_date": data.medication_start_date,
        "source": EventSource.AUTO_DETECTED,
        "validated_by": patient_id,
        "validated_at": datetime.now(timezone.utc),
        "validation_role": ValidationRole.PATIENT,
        "requires_retraining": True
    }
    
    event = await db.create_event(event_data)
    
    # Actualizar cluster
    updated_cluster = await db.update_cluster_status(
        cluster_id, 
        patient_id, 
        status=ClusterStatus.VALIDATED.value,
        validated_event_id=event["id"]
    )
    
    if not updated_cluster:
        logger.error(f"Failed to update cluster status for {cluster_id}")
        # Consider rollback or warning
    else:
        logger.info(f"Cluster {cluster_id} validated successfully with event {event['id']}")
    
    # Incrementar pendientes
    await db.increment_pending_changes(patient_id)
    
    return event


@router.post("/clusters/{cluster_id}/reject", status_code=204)
async def reject_cluster(
    cluster_id: UUID,
    data: RejectClusterRequest,
    patient: dict = Depends(get_current_patient),
):
    """Rechaza un cluster auto-detectado."""
    patient_id = get_patient_id(patient)
    
    cluster = await db.get_cluster_by_id(cluster_id, patient_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster no encontrado")
    
    if cluster["status"] != ClusterStatus.PENDING:
        raise HTTPException(status_code=400, detail="Cluster ya procesado")
    
    await db.update_cluster_status(
        cluster_id,
        patient_id,
        status=ClusterStatus.REJECTED.value,
        rejection_reason=data.reason
    )


# ============================================================================
# ENDPOINTS - EVENTOS POR ID (rutas dinámicas después)
# ============================================================================

@router.get("/{event_id}", response_model=EventResponse)
async def get_event(
    event_id: UUID,
    patient: dict = Depends(get_current_patient),
):
    """Obtiene un evento por ID."""
    patient_id = get_patient_id(patient)
    
    event = await db.get_event_by_id(event_id, patient_id)
    if not event:
        raise HTTPException(status_code=404, detail="Evento no encontrado")
    
    return event


@router.put("/{event_id}", response_model=EventResponse)
async def update_event(
    event_id: UUID,
    data: EventUpdate,
    patient: dict = Depends(get_current_patient),
):
    """Actualiza un evento existente."""
    patient_id = get_patient_id(patient)
    
    event = await db.get_event_by_id(event_id, patient_id)
    if not event:
        raise HTTPException(status_code=404, detail="Evento no encontrado")
    
    update_data = data.model_dump(exclude_unset=True)
    updated_event = await db.update_event(event_id, patient_id, update_data)
    
    return updated_event


@router.delete("/{event_id}", status_code=204)
async def delete_event(
    event_id: UUID,
    patient: dict = Depends(get_current_patient),
):
    """Elimina un evento."""
    patient_id = get_patient_id(patient)
    
    # Check exists
    event = await db.get_event_by_id(event_id, patient_id)
    if not event:
        raise HTTPException(status_code=404, detail="Evento no encontrado")
        
    await db.delete_event(event_id, patient_id)
    
    # Incrementar pendientes
    await db.increment_pending_changes(patient_id)


# ============================================================================
# ENDPOINTS - EVENTOS POR ID (rutas dinámicas después)
# ============================================================================


# ============================================================================
# ENDPOINTS - IMPORTACIÓN
# ============================================================================

@router.post("/import/preview", response_model=ImportPreviewResponse)
async def preview_import(
    file: UploadFile = File(...),
    patient: dict = Depends(get_current_patient),
):
    """Preview de importación (CSV Eventos o TXT WhatsApp)."""
    content = await file.read()
    filename = file.filename.lower()
    
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    
    valid_items = []
    errors = []
    preview_data = []
    
    if filename.endswith(".txt"):
        # Modo WhatsApp
        lines = text.splitlines()
        for i, line in enumerate(lines[:1000]): # Limit preview scan
            line = line.strip()
            if not line: continue
            
            parsed = parse_whatsapp_line(line)
            if parsed:
                valid_items.append(parsed)
                if len(preview_data) < 5:
                    preview_data.append(parsed)
            # No contamos errores en líneas que no matchean, son info del sistema o multilínea
            
        return ImportPreviewResponse(
            valid_events=len(valid_items), # Reusamos campo para count
            invalid_events=0,
            errors=[],
            preview=preview_data
        )

    else:
        # Modo CSV Eventos
        reader = csv.DictReader(io.StringIO(text))
        
        for i, row in enumerate(reader, start=2):
            try:
                # Validar campos requeridos
                if "date" not in row or not row["date"]:
                    errors.append(f"Fila {i}: Fecha requerida")
                    continue
                
                event_type = row.get("event_type", "confirmed_relapse")
                if event_type not in [e.value for e in EventType]:
                    # Intento de corrección simple
                    if event_type == "relapse": event_type = "confirmed_relapse"
                    else:
                        errors.append(f"Fila {i}: Tipo de evento inválido: {event_type}")
                        continue
                
                item = {
                    "date": row["date"],
                    "event_type": event_type,
                    "severity": row.get("severity"),
                    "notes": row.get("notes"),
                }
                valid_items.append(item)
                if len(preview_data) < 5:
                    preview_data.append(item)
                    
            except Exception as e:
                errors.append(f"Fila {i}: {str(e)}")
        
        return ImportPreviewResponse(
            valid_events=len(valid_items),
            invalid_events=len(errors),
            errors=errors[:10],
            preview=preview_data,
        )


@router.post("/import/confirm", response_model=dict)
async def confirm_import(
    file: UploadFile = File(...),
    patient: dict = Depends(get_current_patient),
):
    """Confirma e importa datos (CSV o WhatsApp)."""
    patient_id = get_patient_id(patient)
    content = await file.read()
    filename = file.filename.lower()
    
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    
    imported_count = 0
    type_label = ""
    
    if filename.endswith(".txt"):
        # Importar WhatsApp
        type_label = "mensajes"
        lines = text.splitlines()
        parsed_messages = []
        current_msg = None
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            parsed = parse_whatsapp_line(line)
            if parsed:
                if current_msg:
                    parsed_messages.append(current_msg)
                current_msg = {
                    "date": parsed["date"],
                    "content": parsed["content"],
                    "metadata": {"sender": parsed["sender"]}
                }
            elif current_msg:
                current_msg["content"] += "\n" + line
        
        if current_msg:
            parsed_messages.append(current_msg)
            
        # Batch insert
        chunk_size = 1000
        for i in range(0, len(parsed_messages), chunk_size):
            chunk = parsed_messages[i:i + chunk_size]
            await db.batch_store_raw_messages(patient_id, chunk)
            
        imported_count = len(parsed_messages)
        
    else:
        # Importar CSV Eventos
        type_label = "eventos"
        reader = csv.DictReader(io.StringIO(text))
        
        for row in reader:
            try:
                event_date = datetime.fromisoformat(row["date"].replace("Z", "+00:00"))
                event_type = row.get("event_type", "confirmed_relapse")
                if event_type == "relapse": event_type = "confirmed_relapse"
                
                event_data = {
                    "patient_id": patient_id,
                    "event_date": event_date,
                    "event_type": event_type,
                    "severity": row.get("severity"),
                    "notes": row.get("notes"),
                    "source": EventSource.IMPORTED,
                    "requires_retraining": True
                }
                
                await db.create_event(event_data)
                imported_count += 1
            except Exception as e:
                print(f"[WARN] Error importando fila: {e}")
                continue
        
        if imported_count > 0:
            await db.increment_pending_changes(patient_id, count=imported_count)
    
    # Trigger background processing for WhatsApp messages
    if filename.endswith(".txt") and imported_count > 0:
        import asyncio
        from processor import process_hybrid_full
        # Run processing in background (fire and forget)
        asyncio.create_task(process_hybrid_full(patient_id))
        print(f"[INFO] Background hybrid processing started for {patient_id}")
    
    return {"imported": imported_count, "type": type_label}


# ============================================================================
# ENDPOINTS - MESSAGE PROCESSING
# ============================================================================

@router.get("/messages/stats")
async def get_message_stats(
    patient: dict = Depends(get_current_patient),
):
    """Get stats on raw messages and processed datapoints."""
    patient_id = get_patient_id(patient)
    stats = await db.get_message_stats(patient_id)
    return stats


@router.post("/messages/process")
async def trigger_message_processing(
    patient: dict = Depends(get_current_patient),
):
    """Manually trigger hybrid message processing."""
    patient_id = get_patient_id(patient)
    from processor import process_hybrid_full
    result = await process_hybrid_full(patient_id)
    return result

