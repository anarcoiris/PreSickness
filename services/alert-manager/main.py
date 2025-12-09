"""
Alert Manager Service - EM Predictor
Gestiona alertas basadas en predicciones de riesgo de brote.

Responsables:
- Agent Backend (Backus): implementación
- Agent Architect (Archie): integración

Funcionalidades:
- Evalúa predicciones y genera alertas según umbrales
- Envía notificaciones (email, webhook, push)
- Gestiona estado de alertas (pendiente, reconocida, resuelta)
- Programación de evaluaciones periódicas
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import asyncpg
import httpx
import orjson
import redis.asyncio as redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Logging estructurado
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger("alert-manager")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════


class Settings(BaseSettings):
    """Configuración del servicio de alertas."""

    db_dsn: str = Field(
        default="postgresql://emuser:changeme@postgres:5432/empredictor",
        description="DSN de la base de datos principal.",
    )
    redis_url: Optional[str] = Field(
        default="redis://redis:6379", description="URL de Redis para caché y pubsub."
    )
    redis_password: Optional[str] = None

    # Umbrales de alerta
    threshold_warning: float = Field(
        default=0.35, description="Probabilidad mínima para alerta warning."
    )
    threshold_critical: float = Field(
        default=0.55, description="Probabilidad mínima para alerta critical."
    )

    # Notificaciones
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from: str = "alerts@empredictor.local"

    webhook_url: Optional[str] = Field(
        default=None, description="URL para enviar alertas vía webhook."
    )

    # Scheduling
    eval_interval_minutes: int = Field(
        default=60, description="Intervalo de evaluación de predicciones (minutos)."
    )

    # API de inferencia
    inference_api_url: str = Field(
        default="http://ml_inference:8001",
        description="URL del servicio de inferencia ML.",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()


# ══════════════════════════════════════════════════════════════════════════════
# MODELOS
# ══════════════════════════════════════════════════════════════════════════════


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"


class AlertCreate(BaseModel):
    """Datos para crear una alerta manualmente."""

    user_id_hash: str = Field(min_length=64, max_length=64)
    alert_level: AlertLevel = AlertLevel.WARNING
    alert_type: str = "relapse_risk"
    message: Optional[str] = None
    prediction_id: Optional[int] = None


class AlertResponse(BaseModel):
    """Respuesta con datos de una alerta."""

    id: int
    user_id_hash: str
    alert_level: AlertLevel
    alert_type: str
    message: Optional[str]
    triggered_at: datetime
    acknowledged_at: Optional[datetime]
    acknowledged_by: Optional[str]
    status: AlertStatus
    prediction_id: Optional[int]
    relapse_probability: Optional[float]


class AlertAcknowledge(BaseModel):
    """Datos para reconocer una alerta."""

    acknowledged_by: str = Field(min_length=1, max_length=100)
    action_taken: Optional[str] = None
    notes: Optional[str] = None


class AlertStats(BaseModel):
    """Estadísticas de alertas."""

    total_pending: int
    total_acknowledged: int
    total_resolved: int
    by_level: Dict[str, int]
    avg_response_time_hours: Optional[float]


# ══════════════════════════════════════════════════════════════════════════════
# SERVICIOS
# ══════════════════════════════════════════════════════════════════════════════


class NotificationService:
    """Servicio de envío de notificaciones."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.http_client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        if self.http_client:
            await self.http_client.aclose()

    async def send_email(
        self, to: str, subject: str, body: str
    ) -> bool:
        """Envía email de alerta."""
        if not all([self.settings.smtp_host, self.settings.smtp_user]):
            logger.warning("email_not_configured", to=to)
            return False

        try:
            import aiosmtplib
            from email.message import EmailMessage

            msg = EmailMessage()
            msg["From"] = self.settings.smtp_from
            msg["To"] = to
            msg["Subject"] = subject
            msg.set_content(body)

            await aiosmtplib.send(
                msg,
                hostname=self.settings.smtp_host,
                port=self.settings.smtp_port,
                username=self.settings.smtp_user,
                password=self.settings.smtp_password,
                start_tls=True,
            )
            logger.info("email_sent", to=to, subject=subject)
            return True
        except Exception as e:
            logger.error("email_failed", to=to, error=str(e))
            return False

    async def send_webhook(self, payload: dict) -> bool:
        """Envía alerta vía webhook."""
        if not self.settings.webhook_url:
            return False

        try:
            response = await self.http_client.post(
                self.settings.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            logger.info("webhook_sent", url=self.settings.webhook_url)
            return True
        except Exception as e:
            logger.error("webhook_failed", error=str(e))
            return False

    async def notify_alert(self, alert: dict, channels: List[str] = None):
        """Envía notificación por los canales configurados."""
        channels = channels or ["webhook"]
        results = {}

        payload = {
            "alert_id": alert.get("id"),
            "user_id_hash": alert.get("user_id_hash", "")[:8] + "...",
            "level": alert.get("alert_level"),
            "type": alert.get("alert_type"),
            "message": alert.get("message"),
            "probability": alert.get("relapse_probability"),
            "triggered_at": alert.get("triggered_at"),
        }

        if "webhook" in channels:
            results["webhook"] = await self.send_webhook(payload)

        # Email se enviaría al clínico responsable (requiere lookup)
        # if "email" in channels:
        #     results["email"] = await self.send_email(...)

        return results


class AlertEvaluator:
    """Evalúa predicciones y genera alertas."""

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        redis_client: redis.Redis,
        notification_service: NotificationService,
        settings: Settings,
    ):
        self.db_pool = db_pool
        self.redis = redis_client
        self.notifier = notification_service
        self.settings = settings
        self.http_client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        if self.http_client:
            await self.http_client.aclose()

    async def evaluate_all_users(self):
        """Evalúa predicciones para todos los usuarios activos."""
        logger.info("evaluation_started")

        async with self.db_pool.acquire() as conn:
            # Obtener usuarios activos con predicciones recientes
            users = await conn.fetch(
                """
                SELECT DISTINCT p.user_id_hash
                FROM predictions p
                JOIN users u ON p.user_id_hash = u.user_id_hash
                WHERE u.status = 'active'
                  AND p.created_at >= NOW() - INTERVAL '24 hours'
                """
            )

        alerts_created = 0
        for user in users:
            try:
                alert = await self.evaluate_user(user["user_id_hash"])
                if alert:
                    alerts_created += 1
            except Exception as e:
                logger.error(
                    "user_evaluation_failed",
                    user_id=user["user_id_hash"][:8],
                    error=str(e),
                )

        logger.info("evaluation_completed", alerts_created=alerts_created)
        return alerts_created

    async def evaluate_user(self, user_id_hash: str) -> Optional[dict]:
        """Evalúa predicciones de un usuario y genera alerta si procede."""

        # Obtener última predicción
        async with self.db_pool.acquire() as conn:
            prediction = await conn.fetchrow(
                """
                SELECT id, relapse_probability, horizon_days, model_version
                FROM predictions
                WHERE user_id_hash = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                user_id_hash,
            )

            if not prediction:
                return None

            prob = prediction["relapse_probability"]

            # Determinar nivel de alerta
            if prob >= self.settings.threshold_critical:
                level = AlertLevel.CRITICAL
            elif prob >= self.settings.threshold_warning:
                level = AlertLevel.WARNING
            else:
                return None  # No genera alerta

            # Verificar si ya existe alerta pendiente reciente
            existing = await conn.fetchrow(
                """
                SELECT id FROM alerts
                WHERE user_id_hash = $1
                  AND acknowledged_at IS NULL
                  AND triggered_at >= NOW() - INTERVAL '24 hours'
                """,
                user_id_hash,
            )

            if existing:
                logger.debug(
                    "alert_already_exists",
                    user_id=user_id_hash[:8],
                    existing_id=existing["id"],
                )
                return None

            # Crear alerta
            message = self._generate_message(prob, prediction["horizon_days"], level)

            alert_id = await conn.fetchval(
                """
                INSERT INTO alerts (
                    user_id_hash, prediction_id, alert_level, alert_type,
                    triggered_at, notification_sent
                )
                VALUES ($1, $2, $3, $4, NOW(), FALSE)
                RETURNING id
                """,
                user_id_hash,
                prediction["id"],
                level.value,
                "relapse_risk",
            )

        alert_data = {
            "id": alert_id,
            "user_id_hash": user_id_hash,
            "alert_level": level.value,
            "alert_type": "relapse_risk",
            "message": message,
            "relapse_probability": prob,
            "triggered_at": datetime.utcnow().isoformat(),
        }

        # Enviar notificación
        await self.notifier.notify_alert(alert_data)

        # Marcar como notificada
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE alerts SET notification_sent = TRUE WHERE id = $1",
                alert_id,
            )

        logger.info(
            "alert_created",
            alert_id=alert_id,
            user_id=user_id_hash[:8],
            level=level.value,
            probability=prob,
        )

        return alert_data

    def _generate_message(
        self, probability: float, horizon_days: int, level: AlertLevel
    ) -> str:
        """Genera mensaje descriptivo para la alerta."""
        pct = int(probability * 100)

        if level == AlertLevel.CRITICAL:
            return (
                f"⚠️ ALERTA CRÍTICA: Riesgo elevado de brote ({pct}%) "
                f"en los próximos {horizon_days} días. "
                "Se recomienda contacto inmediato con el paciente."
            )
        else:
            return (
                f"⚡ Alerta: Riesgo moderado de brote ({pct}%) "
                f"en los próximos {horizon_days} días. "
                "Considerar seguimiento cercano."
            )


# ══════════════════════════════════════════════════════════════════════════════
# APLICACIÓN FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="EM Predictor - Alert Manager",
    version="0.1.0",
    description="Gestión de alertas clínicas basadas en predicciones de riesgo.",
)

# Estado global
db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None
notification_service: Optional[NotificationService] = None
alert_evaluator: Optional[AlertEvaluator] = None
scheduler: Optional[AsyncIOScheduler] = None


@app.on_event("startup")
async def startup():
    global db_pool, redis_client, notification_service, alert_evaluator, scheduler

    # Conexión a DB
    try:
        db_pool = await asyncpg.create_pool(settings.db_dsn, min_size=2, max_size=10)
        logger.info("database_connected")
    except Exception as e:
        logger.error("database_connection_failed", error=str(e))

    # Conexión a Redis
    if settings.redis_url:
        try:
            redis_client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=False,
            )
            await redis_client.ping()
            logger.info("redis_connected")
        except Exception as e:
            logger.warning("redis_connection_failed", error=str(e))
            redis_client = None

    # Servicios
    notification_service = NotificationService(settings)
    await notification_service.initialize()

    if db_pool:
        alert_evaluator = AlertEvaluator(
            db_pool, redis_client, notification_service, settings
        )
        await alert_evaluator.initialize()

        # Scheduler para evaluaciones periódicas
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            alert_evaluator.evaluate_all_users,
            "interval",
            minutes=settings.eval_interval_minutes,
            id="evaluate_predictions",
            replace_existing=True,
        )
        scheduler.start()
        logger.info(
            "scheduler_started", interval_minutes=settings.eval_interval_minutes
        )


@app.on_event("shutdown")
async def shutdown():
    if scheduler:
        scheduler.shutdown(wait=False)
    if alert_evaluator:
        await alert_evaluator.close()
    if notification_service:
        await notification_service.close()
    if redis_client:
        await redis_client.close()
    if db_pool:
        await db_pool.close()
    logger.info("service_stopped")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/v1/health")
async def health():
    """Health check del servicio."""
    return {
        "status": "ok" if db_pool else "degraded",
        "database": bool(db_pool),
        "redis": bool(redis_client),
        "scheduler_running": scheduler.running if scheduler else False,
    }


@app.get("/v1/alerts", response_model=List[AlertResponse])
async def list_alerts(
    status: Optional[AlertStatus] = None,
    level: Optional[AlertLevel] = None,
    user_id_hash: Optional[str] = None,
    limit: int = 50,
):
    """Lista alertas con filtros opcionales."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    query = """
        SELECT 
            a.id, a.user_id_hash, a.alert_level, a.alert_type,
            a.triggered_at, a.acknowledged_at, a.acknowledged_by,
            a.prediction_id, p.relapse_probability,
            CASE 
                WHEN a.acknowledged_at IS NOT NULL THEN 'acknowledged'
                WHEN a.triggered_at < NOW() - INTERVAL '7 days' THEN 'expired'
                ELSE 'pending'
            END as status
        FROM alerts a
        LEFT JOIN predictions p ON a.prediction_id = p.id
        WHERE 1=1
    """
    params = []
    param_idx = 1

    if status:
        if status == AlertStatus.PENDING:
            query += " AND a.acknowledged_at IS NULL AND a.triggered_at >= NOW() - INTERVAL '7 days'"
        elif status == AlertStatus.ACKNOWLEDGED:
            query += " AND a.acknowledged_at IS NOT NULL"

    if level:
        query += f" AND a.alert_level = ${param_idx}"
        params.append(level.value)
        param_idx += 1

    if user_id_hash:
        query += f" AND a.user_id_hash = ${param_idx}"
        params.append(user_id_hash)
        param_idx += 1

    query += f" ORDER BY a.triggered_at DESC LIMIT ${param_idx}"
    params.append(limit)

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [
        AlertResponse(
            id=row["id"],
            user_id_hash=row["user_id_hash"],
            alert_level=AlertLevel(row["alert_level"]),
            alert_type=row["alert_type"],
            message=None,
            triggered_at=row["triggered_at"],
            acknowledged_at=row["acknowledged_at"],
            acknowledged_by=row["acknowledged_by"],
            status=AlertStatus(row["status"]),
            prediction_id=row["prediction_id"],
            relapse_probability=row["relapse_probability"],
        )
        for row in rows
    ]


@app.get("/v1/alerts/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: int):
    """Obtiene detalle de una alerta."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT 
                a.id, a.user_id_hash, a.alert_level, a.alert_type,
                a.triggered_at, a.acknowledged_at, a.acknowledged_by,
                a.action_taken, a.prediction_id, p.relapse_probability,
                CASE 
                    WHEN a.acknowledged_at IS NOT NULL THEN 'acknowledged'
                    WHEN a.triggered_at < NOW() - INTERVAL '7 days' THEN 'expired'
                    ELSE 'pending'
                END as status
            FROM alerts a
            LEFT JOIN predictions p ON a.prediction_id = p.id
            WHERE a.id = $1
            """,
            alert_id,
        )

    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")

    return AlertResponse(
        id=row["id"],
        user_id_hash=row["user_id_hash"],
        alert_level=AlertLevel(row["alert_level"]),
        alert_type=row["alert_type"],
        message=row.get("action_taken"),
        triggered_at=row["triggered_at"],
        acknowledged_at=row["acknowledged_at"],
        acknowledged_by=row["acknowledged_by"],
        status=AlertStatus(row["status"]),
        prediction_id=row["prediction_id"],
        relapse_probability=row["relapse_probability"],
    )


@app.post("/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, data: AlertAcknowledge):
    """Reconoce una alerta (marca como vista por un clínico)."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    async with db_pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE alerts
            SET acknowledged_at = NOW(),
                acknowledged_by = $2,
                action_taken = $3
            WHERE id = $1 AND acknowledged_at IS NULL
            """,
            alert_id,
            data.acknowledged_by,
            data.action_taken,
        )

        if result == "UPDATE 0":
            raise HTTPException(
                status_code=404, detail="Alert not found or already acknowledged"
            )

        # Log en audit
        await conn.execute(
            """
            INSERT INTO audit_log (user_id_hash, actor, action, details)
            SELECT user_id_hash, $2, 'alert_acknowledged', $3
            FROM alerts WHERE id = $1
            """,
            alert_id,
            data.acknowledged_by,
            orjson.dumps({"alert_id": alert_id, "notes": data.notes}).decode(),
        )

    logger.info(
        "alert_acknowledged", alert_id=alert_id, acknowledged_by=data.acknowledged_by
    )
    return {"status": "acknowledged", "alert_id": alert_id}


@app.post("/v1/alerts", response_model=AlertResponse)
async def create_alert(data: AlertCreate, background_tasks: BackgroundTasks):
    """Crea una alerta manualmente."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    async with db_pool.acquire() as conn:
        alert_id = await conn.fetchval(
            """
            INSERT INTO alerts (
                user_id_hash, prediction_id, alert_level, alert_type,
                triggered_at, notification_sent
            )
            VALUES ($1, $2, $3, $4, NOW(), FALSE)
            RETURNING id
            """,
            data.user_id_hash,
            data.prediction_id,
            data.alert_level.value,
            data.alert_type,
        )

    # Notificar en background
    alert_data = {
        "id": alert_id,
        "user_id_hash": data.user_id_hash,
        "alert_level": data.alert_level.value,
        "alert_type": data.alert_type,
        "message": data.message,
        "triggered_at": datetime.utcnow().isoformat(),
    }
    background_tasks.add_task(notification_service.notify_alert, alert_data)

    return AlertResponse(
        id=alert_id,
        user_id_hash=data.user_id_hash,
        alert_level=data.alert_level,
        alert_type=data.alert_type,
        message=data.message,
        triggered_at=datetime.utcnow(),
        acknowledged_at=None,
        acknowledged_by=None,
        status=AlertStatus.PENDING,
        prediction_id=data.prediction_id,
        relapse_probability=None,
    )


@app.post("/v1/evaluate")
async def trigger_evaluation(background_tasks: BackgroundTasks):
    """Dispara evaluación manual de todas las predicciones."""
    if not alert_evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not available")

    background_tasks.add_task(alert_evaluator.evaluate_all_users)
    return {"status": "evaluation_triggered"}


@app.get("/v1/stats", response_model=AlertStats)
async def get_stats():
    """Obtiene estadísticas de alertas."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    async with db_pool.acquire() as conn:
        stats = await conn.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE acknowledged_at IS NULL AND triggered_at >= NOW() - INTERVAL '7 days') as pending,
                COUNT(*) FILTER (WHERE acknowledged_at IS NOT NULL) as acknowledged,
                COUNT(*) FILTER (WHERE outcome = 'resolved') as resolved,
                COUNT(*) FILTER (WHERE alert_level = 'warning') as warning_count,
                COUNT(*) FILTER (WHERE alert_level = 'critical') as critical_count,
                COUNT(*) FILTER (WHERE alert_level = 'info') as info_count,
                AVG(EXTRACT(EPOCH FROM (acknowledged_at - triggered_at)) / 3600) 
                    FILTER (WHERE acknowledged_at IS NOT NULL) as avg_response_hours
            FROM alerts
            WHERE triggered_at >= NOW() - INTERVAL '30 days'
            """
        )

    return AlertStats(
        total_pending=stats["pending"] or 0,
        total_acknowledged=stats["acknowledged"] or 0,
        total_resolved=stats["resolved"] or 0,
        by_level={
            "info": stats["info_count"] or 0,
            "warning": stats["warning_count"] or 0,
            "critical": stats["critical_count"] or 0,
        },
        avg_response_time_hours=stats["avg_response_hours"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")

