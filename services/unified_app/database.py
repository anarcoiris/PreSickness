"""
Database module para EM-Predictor Unified App
Conexión a PostgreSQL con asyncpg
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

try:
    import asyncpg
except ImportError:
    asyncpg = None  # Fallback to in-memory

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSettings(BaseSettings):
    db_dsn: str = os.getenv(
        "DB_DSN", 
        "postgresql://emuser:changeme@localhost:5432/empredictor"
    )
    db_min_connections: int = 2
    db_max_connections: int = 10
    
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = DBSettings()

# Pool global
_pool: Optional[asyncpg.Pool] = None


async def init_db() -> bool:
    """Inicializa la conexión a la base de datos."""
    global _pool
    
    if asyncpg is None:
        print("⚠️ asyncpg no instalado, usando modo in-memory")
        return False
    
    try:
        _pool = await asyncpg.create_pool(
            dsn=settings.db_dsn,
            min_size=settings.db_min_connections,
            max_size=settings.db_max_connections,
        )
        print(f"✓ Conectado a PostgreSQL")
        return True
    except Exception as e:
        print(f"⚠️ No se pudo conectar a PostgreSQL: {e}")
        print("  Usando modo in-memory para desarrollo")
        return False


async def close_db():
    """Cierra la conexión a la base de datos."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> Optional[asyncpg.Pool]:
    """Obtiene el pool de conexiones."""
    return _pool


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

async def create_patient(email: str, password_hash: str, name: str) -> Optional[Dict]:
    """Crea un nuevo paciente en la base de datos."""
    if not _pool:
        return None
    
    try:
        async with _pool.acquire() as conn:
            # Crear hash del email para user_id_hash
            import hashlib
            user_id_hash = hashlib.sha256(email.encode()).hexdigest()
            
            row = await conn.fetchrow("""
                INSERT INTO patients (email, password_hash, name, user_id_hash, created_at)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, email, name, created_at
            """, email, password_hash, name, user_id_hash, datetime.now(timezone.utc))
            
            if row:
                return dict(row)
    except asyncpg.UniqueViolationError:
        return None
    except Exception as e:
        print(f"Error creating patient: {e}")
    return None


async def get_patient_by_email(email: str) -> Optional[Dict]:
    """Obtiene un paciente por email."""
    if not _pool:
        return None
    
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, email, password_hash, name, user_id_hash, created_at
                FROM patients WHERE email = $1
            """, email)
            return dict(row) if row else None
    except Exception as e:
        print(f"Error getting patient: {e}")
    return None


async def get_patient_by_id(patient_id: int) -> Optional[Dict]:
    """Obtiene un paciente por ID."""
    if not _pool:
        return None
    
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, email, name, user_id_hash, created_at
                FROM patients WHERE id = $1
            """, patient_id)
            return dict(row) if row else None
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# DATA UPLOAD OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

async def save_upload(patient_id: int, filename: str, file_path: str) -> Optional[Dict]:
    """Guarda registro de archivo subido."""
    if not _pool:
        return None
    
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO data_uploads (patient_id, filename, file_path, uploaded_at, processed)
                VALUES ($1, $2, $3, $4, FALSE)
                RETURNING id, filename, uploaded_at, processed
            """, patient_id, filename, file_path, datetime.now(timezone.utc))
            return dict(row) if row else None
    except Exception as e:
        print(f"Error saving upload: {e}")
    return None


async def get_patient_uploads(patient_id: int) -> List[Dict]:
    """Obtiene uploads de un paciente."""
    if not _pool:
        return []
    
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, filename, uploaded_at, processed
                FROM data_uploads 
                WHERE patient_id = $1
                ORDER BY uploaded_at DESC
            """, patient_id)
            return [dict(row) for row in rows]
    except Exception:
        pass
    return []


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTIONS & ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

async def get_patient_predictions(user_id_hash: str, limit: int = 30) -> List[Dict]:
    """Obtiene historial de predicciones del paciente."""
    if not _pool:
        return []
    
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT prediction_date, relapse_probability, horizon_days, model_version
                FROM predictions
                WHERE user_id_hash = $1
                ORDER BY prediction_date DESC
                LIMIT $2
            """, user_id_hash, limit)
            return [dict(row) for row in rows]
    except Exception:
        pass
    return []


async def save_prediction(user_id_hash: str, probability: float, horizon_days: int, model_version: str) -> Optional[Dict]:
    """Guarda una predicción."""
    if not _pool:
        return None
    
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO predictions (user_id_hash, prediction_date, relapse_probability, horizon_days, model_version)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, prediction_date, relapse_probability
            """, user_id_hash, datetime.now(timezone.utc).date(), probability, horizon_days, model_version)
            return dict(row) if row else None
    except Exception as e:
        print(f"Error saving prediction: {e}")
    return None


async def get_patient_alerts(user_id_hash: str, limit: int = 20) -> List[Dict]:
    """Obtiene alertas del paciente."""
    if not _pool:
        return []
    
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT a.id, a.alert_level, a.alert_type, a.triggered_at, a.acknowledged_at,
                       p.relapse_probability
                FROM alerts a
                LEFT JOIN predictions p ON a.prediction_id = p.id
                WHERE a.user_id_hash = $1
                ORDER BY a.triggered_at DESC
                LIMIT $2
            """, user_id_hash, limit)
            return [dict(row) for row in rows]
    except Exception:
        pass
    return []


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA MIGRATION
# ══════════════════════════════════════════════════════════════════════════════

PATIENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    user_id_hash VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS data_uploads (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id),
    filename VARCHAR(255),
    file_path VARCHAR(512),
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE
);
"""


async def run_migrations():
    """Ejecuta migraciones necesarias."""
    if not _pool:
        return
    
    try:
        async with _pool.acquire() as conn:
            await conn.execute(PATIENTS_TABLE_SQL)
            print("✓ Migraciones ejecutadas")
    except Exception as e:
        print(f"Error en migraciones: {e}")
