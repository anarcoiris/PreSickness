"""
Message Processor - Hybrid implementation (Basic + NLP Chunks)
1. Basic: Fast processing of all raw messages (Level 0)
2. NLP: Targeted processing of "interesting" messages (Level 1)
"""
import asyncio
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict
import aiohttp

import db

logger = logging.getLogger("processor")

NLP_AGENT_URL = os.getenv("NLP_AGENT_URL", "http://localhost:8002/v1/process")


async def extract_features_from_message(
    session: aiohttp.ClientSession,
    message: dict,
    patient_id: str
) -> Optional[dict]:
    """Call NLP Agent to extract features from a single message."""
    try:
        # Handle datetime serialization
        msg_date = message["message_date"]
        if hasattr(msg_date, 'isoformat'):
            msg_date = msg_date.isoformat()
        elif not isinstance(msg_date, str):
            msg_date = str(msg_date)
        
        payload = {
            "message_id": str(message.get("id", "unknown")),
            "user_id": patient_id,
            "text": message["content"],
            "timestamp": msg_date,
            "language_hint": "es"
        }
        
        async with session.post(NLP_AGENT_URL, json=payload, timeout=aiohttp.ClientTimeout(total=300.0)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data
            else:
                logger.warning(f"NLP Agent returned {resp.status} for message {message.get('id')}")
                return None
    except Exception as e:
        logger.error(f"Error calling NLP Agent ({type(e).__name__}): {e}")
        return None


def map_nlp_to_features(nlp_data: dict, existing_features: dict = None) -> dict:
    """Map NLP Agent response to numeric_features, preserving existing if needed."""
    scores = nlp_data.get("symptom_scores", {})
    meta = nlp_data.get("linguistic_meta", {})
    
    features = existing_features.copy() if existing_features else {}
    
    # NLP overrides/additions
    sentiment = 0.0
    if "mood" in scores:
        p = scores["mood"].get("prob", 0.5)
        sentiment = (p * 2) - 1.0
    
    features.update({
        "sentiment_score": sentiment,
        "word_count": meta.get("tokens", features.get("word_count", 0)),
        "avg_sentence_len": meta.get("tokens", 0) / 1.0,
        "pronoun_ratio": meta.get("pronoun_ratio", 0.0),
        "symptom_pain_prob": scores.get("pain", {}).get("prob", 0.0),
        "symptom_fatigue_prob": scores.get("fatigue", {}).get("prob", 0.0),
        "symptom_anxiety_prob": scores.get("anxiety", {}).get("prob", 0.0),
        "nlp_processed_at": datetime.utcnow().isoformat(),
    })
    
    return features


def get_basic_features(message: dict) -> dict:
    """Extract basic features without calling NLP Agent."""
    content = message["content"]
    words = content.split()
    
    # Basic hour detection
    hour = 12
    is_night = 0
    try:
        msg_date = message["message_date"]
        # Handle both string and datetime
        if isinstance(msg_date, str):
            dt = datetime.fromisoformat(msg_date.replace("Z", "+00:00"))
        else:
            dt = msg_date
        hour = dt.hour
        is_night = 1 if (hour >= 23 or hour < 6) else 0
    except:
        pass

    return {
        "sentiment_score": 0.0,
        "word_count": len(words),
        "avg_sentence_len": len(words),
        "type_token_ratio": 0.5,
        "pronoun_ratio": 0.0,
        "hour": hour,
        "is_night": is_night,
        "symptom_pain_prob": 0.0,
        "symptom_fatigue_prob": 0.0,
        "symptom_anxiety_prob": 0.0,
        "num_messages": 1,
        "processed_at": datetime.utcnow().isoformat(),
        "processing_mode": "basic"
    }


async def process_basic_for_patient(patient_id: str, limit: int = 5000) -> dict:
    """Fast processing of raw messages to basic datapoints (Level 0)."""
    logger.info(f"[Processor] Starting BASIC processing for {patient_id}")
    
    unprocessed = await db.get_unprocessed_messages(patient_id, limit=limit)
    if not unprocessed:
        return {"processed": 0}

    datapoints = []
    for msg in unprocessed:
        features = get_basic_features(msg)
        
        # Consistent timestamp
        try:
            if isinstance(msg["message_date"], str):
                msg_time = datetime.fromisoformat(msg["message_date"].replace("Z", "+00:00"))
            else:
                msg_time = msg["message_date"]
        except:
            msg_time = datetime.utcnow()

        datapoints.append({
            "user_id_hash": patient_id,
            "time": msg_time,
            "source": msg.get("source", "whatsapp"),
            "source_hash": msg["content_hash"],
            "numeric_features": features,
            "nlp_level": 0
        })

    # Batch store
    chunk_size = 500
    for i in range(0, len(datapoints), chunk_size):
        await db.batch_store_datapoints(datapoints[i:i + chunk_size])
    
    logger.info(f"[Processor] BASIC processing complete for {patient_id}: {len(datapoints)} messages")
    return {"processed": len(datapoints)}


async def upgrade_nlp_for_patient(patient_id: str, limit: int = 100) -> dict:
    """Upgrade basic datapoints to NLP level for interesting messages (Level 1)."""
    logger.info(f"[Processor] Starting NLP UPGRADE for {patient_id}")
    
    needing_nlp = await db.get_messages_needing_nlp(patient_id, limit=limit)
    if not needing_nlp:
        return {"upgraded": 0}

    semaphore = asyncio.Semaphore(2)  # Max 2 parallel requests to prevent Ollama deepseek-r1 queue timeout
    
    async def process_single(session: aiohttp.ClientSession, msg: dict):
        async with semaphore:
            nlp_result = await extract_features_from_message(session, msg, patient_id)
            if not nlp_result:
                return False

            features = map_nlp_to_features(nlp_result)
            features["processing_mode"] = "nlp_hybrid"
            
            try:
                if isinstance(msg["message_date"], str):
                    msg_time = datetime.fromisoformat(msg["message_date"].replace("Z", "+00:00"))
                else:
                    msg_time = msg["message_date"]
            except:
                msg_time = datetime.utcnow()

            await db.store_datapoint({
                "user_id_hash": patient_id,
                "time": msg_time,
                "source_hash": msg["content_hash"],
                "numeric_features": features,
                "nlp_level": 1
            })
            return True

    upgraded_count = 0
    async with aiohttp.ClientSession() as session:
        tasks = [process_single(session, msg) for msg in needing_nlp]
        results = await asyncio.gather(*tasks)
        upgraded_count = sum(1 for r in results if r)

    logger.info(f"[Processor] NLP UPGRADE complete for {patient_id}: {upgraded_count} messages")
    return {"upgraded": upgraded_count}


async def process_hybrid_full(patient_id: str):
    """Run full hybrid pipeline: Basic all, then NLP Chunks."""
    # 1. Process all new messages in basic mode (Level 0)
    # Target all messages - 40K chunk is fine for basic
    basic_res = await process_basic_for_patient(patient_id, limit=40000)
    
    # 2. Upgrade interesting messages to NLP (Level 1)
    # Increased limit due to parallel processing
    nlp_res = await upgrade_nlp_for_patient(patient_id, limit=1000)
    
    return {
        "basic": basic_res,
        "nlp_upgrade": nlp_res
    }


async def process_all_patients():
    """Driver for background processing of all active patients."""
    # Simple list of patients with raw messages
    patients = await db.fetch_all("SELECT DISTINCT patient_id FROM raw_messages")
    for row in patients:
        try:
            await process_hybrid_full(row["patient_id"])
        except Exception as e:
            logger.error(f"Error processing patient {row['patient_id']}: {e}")
