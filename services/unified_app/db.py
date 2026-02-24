"""
Database Connection Module for unified_app.
Provides async database access using psycopg 3 (async) for PostgreSQL/TimescaleDB.
Consolidated version (Phase 1 & 2).
"""
import os
import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from uuid import UUID
import logging
import json

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = logging.getLogger(__name__)

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "empredictor")
DB_USER = os.getenv("DB_USER", "emuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "changeme")

DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Connection pool singleton
_pool = None

async def get_pool():
    """Get or create the database connection pool."""
    global _pool
    if _pool is None:
        try:
            from psycopg_pool import AsyncConnectionPool
            from psycopg.rows import dict_row

            _pool = AsyncConnectionPool(
                DSN,
                min_size=2,
                max_size=10,
                open=False,
                kwargs={"row_factory": dict_row}
            )
            await _pool.open()
            logger.info(f"[DB] Connected to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME}")
        except Exception as e:
            logger.error(f"[DB] Failed to connect: {e}")
            raise
    return _pool

async def close_pool():
    """Close the database connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("[DB] Connection pool closed")

@asynccontextmanager
async def get_connection():
    """Context manager for database connections."""
    pool = await get_pool()
    async with pool.connection() as conn:
        yield conn

async def execute_query(query: str, *args) -> str:
    """Execute a query without returning results."""
    async with get_connection() as conn:
        await conn.execute(query, args)
        return "OK"

async def fetch_all(query: str, *args) -> List[Dict[str, Any]]:
    """Execute a query and return all rows as dictionaries."""
    async with get_connection() as conn:
        cursor = await conn.execute(query, args)
        rows = await cursor.fetchall()
        return rows

async def fetch_one(query: str, *args) -> Optional[Dict[str, Any]]:
    """Execute a query and return a single row as dictionary."""
    async with get_connection() as conn:
        cursor = await conn.execute(query, args)
        row = await cursor.fetchone()
        return row

async def fetch_val(query: str, *args) -> Any:
    """Execute a query and return a single value."""
    async with get_connection() as conn:
        cursor = await conn.execute(query, args)
        row = await cursor.fetchone()
        if row:
            return list(row.values())[0]
        return None

# ============================================================================
# USERS
# ============================================================================

async def get_user_by_email(email: str) -> Optional[Dict]:
    """Get a user by email."""
    return await fetch_one("SELECT * FROM users WHERE email = %s", email)

async def get_user_by_id(user_id_hash: str) -> Optional[Dict]:
    """Get a user by user_id_hash."""
    return await fetch_one("SELECT * FROM users WHERE user_id_hash = %s", user_id_hash)

async def create_user(data: Dict) -> Dict:
    """Create a new user with optional role."""
    role = data.get('role', 'patient')
    return await fetch_one(
        """
        INSERT INTO users (user_id_hash, email, password_hash, name, role, consent_given_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        RETURNING *
        """,
        data['user_id_hash'], data['email'], data['password_hash'], data['name'], role
    )

async def update_user_role(user_id_hash: str, role: str) -> Optional[Dict]:
    """Update user role."""
    return await fetch_one(
        "UPDATE users SET role = %s WHERE user_id_hash = %s RETURNING *",
        role, user_id_hash
    )

async def get_all_users() -> List[Dict]:
    """Get all users for admin panel."""
    return await fetch_all("SELECT user_id_hash, email, name, role, status, created_at FROM users ORDER BY created_at DESC")

async def update_user_status(user_id_hash: str, status: str) -> Optional[Dict]:
    """Update user status (active, paused, deleted)."""
    return await fetch_one("UPDATE users SET status = %s WHERE user_id_hash = %s RETURNING *", status, user_id_hash)

async def delete_user_permanently(user_id_hash: str) -> bool:
    """Delete a user and all their data (careful!)."""
    # Order matters for foreign keys
    await execute_query("DELETE FROM doctor_patients WHERE patient_id = %s OR doctor_id = %s", user_id_hash, user_id_hash)
    await execute_query("DELETE FROM datapoints WHERE user_id_hash = %s", user_id_hash)
    await execute_query("DELETE FROM feature_windows WHERE user_id_hash = %s", user_id_hash)
    await execute_query("DELETE FROM clinical_events WHERE patient_id = %s", user_id_hash)
    await execute_query("DELETE FROM users WHERE user_id_hash = %s", user_id_hash)
    return True

# ============================================================================
# DOCTOR-PATIENT RELATIONSHIPS
# ============================================================================

async def add_patient_to_doctor(doctor_id: str, patient_id: str, granted_by: str = None) -> Dict:
    """Add a patient to a doctor's care."""
    return await fetch_one(
        """
        INSERT INTO doctor_patients (doctor_id, patient_id, granted_by, status)
        VALUES (%s, %s, %s, 'active')
        ON CONFLICT (doctor_id, patient_id) 
        DO UPDATE SET status = 'active', revoked_at = NULL, granted_at = NOW()
        RETURNING *
        """,
        doctor_id, patient_id, granted_by or patient_id
    )

async def get_doctor_patients(doctor_id: str, include_pending: bool = False) -> List[Dict]:
    """Get all patients for a doctor."""
    if include_pending:
        return await fetch_all(
            """
            SELECT dp.*, u.name as patient_name, u.email as patient_email
            FROM doctor_patients dp
            JOIN users u ON dp.patient_id = u.user_id_hash
            WHERE dp.doctor_id = %s AND dp.status IN ('active', 'pending')
            ORDER BY dp.granted_at DESC
            """,
            doctor_id
        )
    return await fetch_all(
        """
        SELECT dp.*, u.name as patient_name, u.email as patient_email
        FROM doctor_patients dp
        JOIN users u ON dp.patient_id = u.user_id_hash
        WHERE dp.doctor_id = %s AND dp.status = 'active'
        ORDER BY dp.granted_at DESC
        """,
        doctor_id
    )

async def get_patient_doctors(patient_id: str) -> List[Dict]:
    """Get all doctors for a patient."""
    return await fetch_all(
        """
        SELECT dp.*, u.name as doctor_name, u.email as doctor_email
        FROM doctor_patients dp
        JOIN users u ON dp.doctor_id = u.user_id_hash
        WHERE dp.patient_id = %s AND dp.status = 'active'
        ORDER BY dp.granted_at DESC
        """,
        patient_id
    )

async def remove_patient_from_doctor(doctor_id: str, patient_id: str) -> bool:
    """Revoke a doctor-patient relationship."""
    await execute_query(
        """
        UPDATE doctor_patients 
        SET status = 'revoked', revoked_at = NOW()
        WHERE doctor_id = %s AND patient_id = %s
        """,
        doctor_id, patient_id
    )
    return True

async def doctor_has_patient_access(doctor_id: str, patient_id: str) -> bool:
    """Check if doctor has active access to patient."""
    result = await fetch_val(
        """
        SELECT COUNT(*) FROM doctor_patients
        WHERE doctor_id = %s AND patient_id = %s AND status = 'active'
        """,
        doctor_id, patient_id
    )
    return (result or 0) > 0


async def store_upload(data: Dict) -> Dict:
    """Store upload record."""
    return await fetch_one(
        """
        INSERT INTO uploads (id, patient_id, filename, file_path, uploaded_at, processed)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING *
        """,
        data['id'], data['patient_id'], data['filename'], data['file_path'], data['uploaded_at'], data.get('processed', False)
    )

async def get_uploads_by_patient(patient_id: str) -> List[Dict]:
    """Get all uploads for a patient."""
    return await fetch_all("SELECT * FROM uploads WHERE patient_id = %s ORDER BY uploaded_at DESC", patient_id)

async def get_system_metrics() -> Dict:
    """Get global system metrics including message processing stats."""
    total_patients = await fetch_val("SELECT COUNT(*) FROM users")
    total_events = await fetch_val("SELECT COUNT(*) FROM clinical_events")
    total_uploads = await fetch_val("SELECT COUNT(*) FROM uploads")
    
    msg_stats = await fetch_one("""
        SELECT 
            (SELECT COUNT(*) FROM raw_messages) as total_messages,
            (SELECT COUNT(*) FROM datapoints) as total_datapoints,
            (SELECT COUNT(*) FROM datapoints WHERE nlp_level > 0) as nlp_processed
    """)
    
    return {
        "total_patients": total_patients or 0,
        "total_events": total_events or 0,
        "total_uploads": total_uploads or 0,
        "total_messages": msg_stats["total_messages"] or 0,
        "total_datapoints": msg_stats["total_datapoints"] or 0,
        "nlp_processed": msg_stats["nlp_processed"] or 0
    }

# ============================================================================
# RAW MESSAGES
# ============================================================================

async def store_raw_message(patient_id: str, message_date: str, content: str, source: str, metadata: Dict = None) -> Dict:
    """Store a raw message with hash to prevent duplicates."""
    import hashlib
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    from psycopg.types.json import Json
    
    return await fetch_one(
        """
        INSERT INTO raw_messages (patient_id, message_date, content_encrypted, content_hash, source, metadata)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (patient_id, content_hash) DO UPDATE SET created_at = NOW()
        RETURNING *
        """,
        patient_id, message_date, content, content_hash, source, Json(metadata or {})
    )

async def count_raw_messages(patient_id: str) -> int:
    """Count raw messages for a patient."""
    return await fetch_val("SELECT COUNT(*) FROM raw_messages WHERE patient_id = %s", patient_id) or 0

async def batch_store_raw_messages(patient_id: str, messages: List[dict]):
    """Store multiple messages efficiently."""
    import hashlib
    from psycopg.types.json import Json
    
    data = []
    for msg in messages:
        content = msg["content"]
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        data.append((
            patient_id,
            msg["date"],
            content,
            content_hash,
            msg.get("source", "whatsapp"),
            Json(msg.get("metadata", {}))
        ))
    
    if not data:
        return

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.executemany(
                """
                INSERT INTO raw_messages (patient_id, message_date, content_encrypted, content_hash, source, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (patient_id, content_hash) DO NOTHING
                """,
                data
            )

# ============================================================================
# CLINICAL EVENTS
# ============================================================================

async def get_events(patient_id: str, limit: int = 50, offset: int = 0) -> List[Dict]:
    return await fetch_all(
        "SELECT * FROM clinical_events WHERE patient_id = %s ORDER BY event_date DESC LIMIT %s OFFSET %s",
        patient_id, limit, offset
    )

async def create_event(data: Dict) -> Dict:
    return await fetch_one(
        """
        INSERT INTO clinical_events (
            patient_id, event_date, event_type, severity, notes,
            medication_start_date, source, validated_by, validated_at,
            validation_role, requires_retraining
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING *
        """,
        data['patient_id'], data['event_date'], data['event_type'],
        data.get('severity'), data.get('notes'), data.get('medication_start_date'),
        data.get('source', 'manual'), data.get('validated_by'),
        data.get('validated_at'), data.get('validation_role'),
        data.get('requires_retraining', True)
    )

# ... (simplified version of other methods for brevity, keeping only essential ones from the previous block)
async def get_event_by_id(event_id: UUID, patient_id: str) -> Optional[Dict]:
    return await fetch_one("SELECT * FROM clinical_events WHERE id = %s AND patient_id = %s", event_id, patient_id)

async def update_event(event_id: UUID, patient_id: str, data: Dict) -> Optional[Dict]:
    set_parts = []
    values = []
    for key, value in data.items():
        if value is not None:
            set_parts.append(f"{key} = %s")
            values.append(value)
    if not set_parts: return await get_event_by_id(event_id, patient_id)
    values.extend([event_id, patient_id])
    return await fetch_one(f"UPDATE clinical_events SET {', '.join(set_parts)}, updated_at = NOW() WHERE id = %s AND patient_id = %s RETURNING *", *values)

async def delete_event(event_id: UUID, patient_id: str) -> bool:
    await execute_query("DELETE FROM clinical_events WHERE id = %s AND patient_id = %s", event_id, patient_id)
    return True

# ============================================================================
# CLUSTERS & SETTINGS
# ============================================================================

async def get_clusters(patient_id: str, status: str = None) -> List[Dict]:
    if status: return await fetch_all("SELECT * FROM auto_clusters WHERE patient_id = %s AND status = %s ORDER BY peak_date DESC", patient_id, status)
    return await fetch_all("SELECT * FROM auto_clusters WHERE patient_id = %s ORDER BY peak_date DESC", patient_id)

async def get_cluster_by_id(cluster_id: UUID, patient_id: str) -> Optional[Dict]:
    return await fetch_one("SELECT * FROM auto_clusters WHERE id = %s AND patient_id = %s", cluster_id, patient_id)

async def update_cluster_status(cluster_id: UUID, patient_id: str, status: str, validated_event_id: UUID = None, rejection_reason: str = None) -> Optional[Dict]:
    return await fetch_one(
        """
        UPDATE auto_clusters 
        SET status = %s, validated_event_id = %s, rejection_reason = %s, 
            processed_at = NOW(), processed_by = %s 
        WHERE id = %s AND patient_id = %s 
        RETURNING *
        """, 
        status, validated_event_id, rejection_reason, patient_id, cluster_id, patient_id
    )

async def get_label_settings(patient_id: str) -> Optional[Dict]:
    return await fetch_one("SELECT * FROM label_settings WHERE patient_id = %s", patient_id)

async def upsert_label_settings(patient_id: str, data: Dict) -> Dict:
    from psycopg.types.json import Json
    return await fetch_one(
        """
        INSERT INTO label_settings (patient_id, horizons, label_event_types, censor_days_before_end, use_auto_clusters)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (patient_id) DO UPDATE SET
            horizons = EXCLUDED.horizons, label_event_types = EXCLUDED.label_event_types, 
            censor_days_before_end = EXCLUDED.censor_days_before_end, use_auto_clusters = EXCLUDED.use_auto_clusters,
            updated_at = NOW()
        RETURNING *
        """,
        patient_id, Json(data.get('horizons', [7, 14, 30])), Json(data.get('label_event_types', ['confirmed_relapse', 'medication_start'])),
        data.get('censor_days_before_end', 30), data.get('use_auto_clusters', False)
    )

async def increment_pending_changes(patient_id: str, count: int = 1) -> int:
    result = await fetch_val("UPDATE label_settings SET pending_changes = pending_changes + %s, updated_at = NOW() WHERE patient_id = %s RETURNING pending_changes", count, patient_id)
    if result is None:
        await upsert_label_settings(patient_id, {})
        return count
    return result

async def reset_pending_changes(patient_id: str) -> None:
    await execute_query("UPDATE label_settings SET pending_changes = 0, last_labels_generated_at = NOW(), updated_at = NOW() WHERE patient_id = %s", patient_id)

async def get_event_stats(patient_id: str) -> Dict:
    stats = await fetch_one(
        """
        SELECT 
            COUNT(*) as total_events, 
            COUNT(*) FILTER (WHERE event_type = 'confirmed_relapse') as confirmed_relapses,
            COUNT(*) FILTER (WHERE event_type = 'medication_start') as medication_starts,
            MAX(event_date) as last_event_date 
        FROM clinical_events WHERE patient_id = %s
        """, 
        patient_id
    )
    pending_clusters = await fetch_val("SELECT COUNT(*) FROM auto_clusters WHERE patient_id = %s AND status = 'pending'", patient_id)
    settings = await get_label_settings(patient_id)
    return {
        "total_events": stats['total_events'] or 0,
        "confirmed_relapses": stats['confirmed_relapses'] or 0,
        "medication_starts": stats['medication_starts'] or 0,
        "pending_clusters": pending_clusters or 0,
        "pending_retraining": settings.get('pending_changes', 0) if settings else 0,
        "last_event_date": stats['last_event_date'],
    }

# ============================================================================
# DATAPOINTS (Message -> Feature Pipeline)
# ============================================================================

async def get_unprocessed_messages(patient_id: str, limit: int = 500) -> List[Dict]:
    """Get raw messages that haven't been converted to datapoints yet."""
    return await fetch_all(
        """
        SELECT rm.id, rm.patient_id, rm.message_date, rm.content_encrypted as content, 
               rm.content_hash, rm.metadata
        FROM raw_messages rm
        LEFT JOIN datapoints dp ON rm.content_hash = dp.source_hash AND rm.patient_id = dp.user_id_hash
        WHERE rm.patient_id = %s AND dp.source_hash IS NULL
        ORDER BY rm.message_date ASC
        LIMIT %s
        """,
        patient_id, limit
    )

async def store_datapoint(data: Dict) -> Dict:
    """Store a single datapoint (feature vector extracted from a message)."""
    from psycopg.types.json import Json
    from datetime import datetime
    
    return await fetch_one(
        """
        INSERT INTO datapoints (
            user_id_hash, time, source, source_hash, numeric_features, nlp_level
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (time, user_id_hash, source_hash) DO UPDATE SET
            numeric_features = EXCLUDED.numeric_features,
            nlp_level = EXCLUDED.nlp_level
        RETURNING *
        """,
        data['user_id_hash'],
        data['time'],
        data.get('source', 'whatsapp'),
        data['source_hash'],
        Json(data['numeric_features']),
        data.get('nlp_level', 0)
    )

async def batch_store_datapoints(datapoints: List[Dict]) -> int:
    """Store multiple datapoints efficiently."""
    from psycopg.types.json import Json
    
    if not datapoints:
        return 0
    
    data = [
        (
            dp['user_id_hash'],
            dp['time'],
            dp.get('source', 'whatsapp'),
            dp['source_hash'],
            Json(dp['numeric_features']),
            dp.get('nlp_level', 0)
        )
        for dp in datapoints
    ]
    
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.executemany(
                """
                INSERT INTO datapoints (user_id_hash, time, source, source_hash, numeric_features, nlp_level)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (time, user_id_hash, source_hash) DO UPDATE SET
                    numeric_features = EXCLUDED.numeric_features,
                    source = EXCLUDED.source,
                    nlp_level = EXCLUDED.nlp_level
                """,
                data
            )
    return len(datapoints)

async def get_message_stats(patient_id: str) -> Dict:
    """Get stats on messages and datapoints for a patient."""
    raw_count = await fetch_val("SELECT COUNT(*) FROM raw_messages WHERE patient_id = %s", patient_id)
    dp_stats = await fetch_one(
        """
        SELECT 
            COUNT(*) as processed_datapoints,
            COUNT(*) FILTER (WHERE nlp_level > 0) as nlp_processed
        FROM datapoints WHERE user_id_hash = %s
        """,
        patient_id
    )
    return {
        "raw_messages": raw_count or 0,
        "processed_datapoints": dp_stats['processed_datapoints'] or 0,
        "nlp_processed": dp_stats['nlp_processed'] or 0
    }

async def get_messages_needing_nlp(patient_id: str, limit: int = 100) -> List[Dict]:
    """
    Get raw messages for datapoints that are only basic-processed (nlp_level=0)
    but have 'interest' (e.g. high word count or specific keywords).
    """
    return await fetch_all(
        """
        SELECT rm.id, rm.patient_id, rm.message_date, rm.content_encrypted as content, 
               rm.content_hash, rm.metadata
        FROM datapoints dp
        JOIN raw_messages rm ON dp.source_hash = rm.content_hash AND dp.user_id_hash = rm.patient_id
        WHERE dp.user_id_hash = %s AND dp.nlp_level = 0
        AND (
            (dp.numeric_features->>'word_count')::int > 15
            OR (rm.content_encrypted ILIKE '%%dolor%%' OR rm.content_encrypted ILIKE '%%cansado%%' OR rm.content_encrypted ILIKE '%%brote%%')
        )
        ORDER BY dp.time ASC
        LIMIT %s
        """,
        patient_id, limit
    )
# ============================================================================
# PREDICTIONS & ALERTS
# ============================================================================

async def store_prediction(data: Dict) -> Dict:
    """Store a prediction result in the database."""
    from psycopg.types.json import Json
    return await fetch_one(
        """
        INSERT INTO predictions (
            user_id_hash, prediction_date, horizon_days, 
            relapse_probability, model_version, model_name,
            feature_importance
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING *
        """,
        data['user_id_hash'], data.get('prediction_date', datetime.now(timezone.utc).date()),
        data['horizon_days'], data['relapse_probability'],
        data.get('model_version', 'v1'), data.get('model_name', 'tft_prototype'),
        Json(data.get('feature_importance', {}))
    )

async def get_prediction_history(user_id_hash: str, limit: int = 30) -> List[Dict]:
    """Get historical predictions for a patient."""
    return await fetch_all(
        """
        SELECT * FROM predictions 
        WHERE user_id_hash = %s 
        ORDER BY created_at DESC 
        LIMIT %s
        """,
        user_id_hash, limit
    )

async def store_alert(data: Dict) -> Dict:
    """Store an alert triggered by the system."""
    from psycopg.types.json import Json
    return await fetch_one(
        """
        INSERT INTO alerts (
            user_id_hash, prediction_id, alert_level, alert_type, 
            triggered_at, notification_sent, notification_channels
        )
        VALUES (%s, %s, %s, %s, NOW(), %s, %s)
        RETURNING *
        """,
        data['user_id_hash'], data.get('prediction_id'),
        data['alert_level'], data['alert_type'],
        data.get('notification_sent', False), Json(data.get('notification_channels', []))
    )

async def get_alerts_by_patient(user_id_hash: str, limit: int = 20) -> List[Dict]:
    """Get recent alerts for a patient."""
    return await fetch_all(
        """
        SELECT * FROM alerts 
        WHERE user_id_hash = %s 
        ORDER BY triggered_at DESC 
        LIMIT %s
        """,
        user_id_hash, limit
    )

# ============================================================================
# SYSTEM INCIDENTS
# ============================================================================

async def get_incidents(limit: int = 50, include_resolved: bool = False) -> List[Dict]:
    """Get system incidents."""
    if include_resolved:
        return await fetch_all("SELECT * FROM system_incidents ORDER BY created_at DESC LIMIT %s", limit)
    return await fetch_all("SELECT * FROM system_incidents WHERE resolved = FALSE ORDER BY created_at DESC LIMIT %s", limit)

async def create_incident(severity: str, component: str, message: str, details: Dict = None) -> Dict:
    """Log a system incident."""
    from psycopg.types.json import Json
    return await fetch_one(
        """
        INSERT INTO system_incidents (severity, component, message, details)
        VALUES (%s, %s, %s, %s)
        RETURNING *
        """,
        severity, component, message, Json(details or {})
    )

async def resolve_incident(incident_id: UUID, resolved_by: str) -> Optional[Dict]:
    """Mark an incident as resolved."""
    return await fetch_one(
        """
        UPDATE system_incidents 
        SET resolved = TRUE, resolved_at = NOW(), resolved_by = %s 
        WHERE id = %s RETURNING *
        """,
        resolved_by, incident_id
    )

# ============================================================================
# OAUTH TOKENS
# ============================================================================

async def store_oauth_tokens(user_id_hash: str, provider: str, tokens: Dict) -> Dict:
    """Store or update OAuth tokens for a user."""
    return await fetch_one(
        """
        INSERT INTO user_oauth (
            user_id_hash, provider, access_token, refresh_token, 
            token_expiry, scope, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (user_id_hash, provider) 
        DO UPDATE SET
            access_token = EXCLUDED.access_token,
            refresh_token = COALESCE(EXCLUDED.refresh_token, user_oauth.refresh_token),
            token_expiry = EXCLUDED.token_expiry,
            scope = EXCLUDED.scope,
            updated_at = NOW()
        RETURNING *
        """,
        user_id_hash, provider, 
        tokens.get('access_token'), tokens.get('refresh_token'),
        tokens.get('expiry'), tokens.get('scope')
    )

async def get_oauth_tokens(user_id_hash: str, provider: str) -> Optional[Dict]:
    """Get OAuth tokens for a user."""
    return await fetch_one(
        "SELECT * FROM user_oauth WHERE user_id_hash = %s AND provider = %s",
        user_id_hash, provider
    )
