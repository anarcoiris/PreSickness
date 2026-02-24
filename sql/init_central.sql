-- ============================================================================
-- Centralized Schema for EM-Predictor
-- Merged: db_schema.sql + 02-events-schema.sql + raw_messages
-- ============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ══════════════════════════════════════════════════════════════════════════════
-- 1. USER & SYSTEM STATE
-- ══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id_hash VARCHAR(64) UNIQUE NOT NULL,  -- SHA-256 of user_id
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255),
    name VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    consent_given_at TIMESTAMPTZ DEFAULT NOW(),
    consent_revoked_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'deleted')),
    
    -- Clinical metadata (encrypted)
    age_encrypted BYTEA,
    gender_encrypted BYTEA,
    ms_type_encrypted BYTEA,  -- RRMS, SPMS, PPMS
    edss_baseline_encrypted BYTEA,
    
    data_retention_days INT DEFAULT 730
);

CREATE TABLE IF NOT EXISTS devices (
    id SERIAL PRIMARY KEY,
    device_id_hash VARCHAR(64) UNIQUE NOT NULL,
    user_id_hash VARCHAR(64) REFERENCES users(user_id_hash),
    secret TEXT NOT NULL,
    registered_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ,
    device_type VARCHAR(50),
    app_version VARCHAR(20),
    permissions JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS uploads (
    id VARCHAR(16) PRIMARY KEY,
    patient_id VARCHAR(64) NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE,
    error_message TEXT
);

-- ══════════════════════════════════════════════════════════════════════════════
-- 2. RAW DATA STORAGE
-- ══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS raw_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(64) NOT NULL,
    message_date TIMESTAMPTZ NOT NULL,
    content_encrypted TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    source VARCHAR(20) CHECK (source IN ('whatsapp', 'telegram', 'app', 'imported')),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT unique_message_hash UNIQUE (patient_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_raw_messages_patient ON raw_messages(patient_id, message_date DESC);

-- ══════════════════════════════════════════════════════════════════════════════
-- 3. CLINICAL DOMAIN (WebApp Core)
-- ══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS clinical_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(64) NOT NULL,
    event_date TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN (
        'symptom_onset',
        'confirmed_relapse', 
        'medication_start',
        'hospital_visit',
        'doctor_appointment'
    )),
    severity VARCHAR(20) CHECK (severity IN ('mild', 'moderate', 'severe')),
    notes TEXT,
    medication_start_date TIMESTAMPTZ,
    source VARCHAR(20) DEFAULT 'manual' CHECK (source IN ('manual', 'auto_detected', 'imported')),
    cluster_id UUID,
    validated_by VARCHAR(64),
    validated_at TIMESTAMPTZ,
    validation_role VARCHAR(20) CHECK (validation_role IN ('patient', 'doctor')),
    requires_retraining BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(64),
    CONSTRAINT unique_patient_event UNIQUE (patient_id, event_date, event_type)
);

CREATE TABLE IF NOT EXISTS auto_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(64) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    peak_date DATE NOT NULL,
    total_signals INT,
    unique_types INT,
    max_severity VARCHAR(20),
    severity_score FLOAT,
    density FLOAT,
    is_probable_relapse BOOLEAN DEFAULT FALSE,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'validated', 'rejected')),
    validated_event_id UUID REFERENCES clinical_events(id) ON DELETE SET NULL,
    processed_by VARCHAR(64),
    processed_at TIMESTAMPTZ,
    rejection_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS label_settings (
    patient_id VARCHAR(64) PRIMARY KEY,
    horizons JSONB DEFAULT '[7, 14, 30]',
    label_event_types JSONB DEFAULT '["confirmed_relapse", "medication_start"]',
    censor_days_before_end INT DEFAULT 30,
    use_auto_clusters BOOLEAN DEFAULT FALSE,
    auto_cluster_min_confidence FLOAT DEFAULT 0.8,
    last_labels_generated_at TIMESTAMPTZ,
    pending_changes INT DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ══════════════════════════════════════════════════════════════════════════════
-- 4. ML FEATURES & PREDICTIONS (Hypertable)
-- ══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS datapoints (
    time TIMESTAMPTZ NOT NULL,
    user_id_hash VARCHAR(64) NOT NULL,
    device_id_hash VARCHAR(64),
    source VARCHAR(32) DEFAULT 'unknown',
    source_hash VARCHAR(128),
    embedding_encrypted TEXT,
    embedding_dim INT DEFAULT 768,
    numeric_features JSONB NOT NULL,
    data_quality_score FLOAT,
    PRIMARY KEY (time, user_id_hash)
);

-- Add source_hash index for deduplication
CREATE UNIQUE INDEX IF NOT EXISTS idx_datapoints_source_hash 
    ON datapoints(user_id_hash, time, source_hash) WHERE source_hash IS NOT NULL;

-- Convert to hypertable if Timescale is available
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('datapoints', 'time', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS feature_windows (
    user_id_hash VARCHAR(64) NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    window_size_days INT NOT NULL,
    features JSONB NOT NULL,
    feature_version VARCHAR(20) DEFAULT 'v1',
    num_datapoints INT,
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id_hash, window_end, window_size_days)
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    user_id_hash VARCHAR(64) NOT NULL,
    prediction_date DATE NOT NULL,
    horizon_days INT NOT NULL,
    relapse_probability FLOAT NOT NULL,
    confidence_interval JSONB,
    model_version VARCHAR(50) NOT NULL,
    model_name VARCHAR(50),
    feature_importance JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT valid_probability CHECK (relapse_probability BETWEEN 0 AND 1)
);

CREATE TABLE IF NOT EXISTS retraining_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(64) NOT NULL,
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    triggered_by VARCHAR(64),
    events_used INT,
    labels_positive INT,
    labels_negative INT,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    error_message TEXT,
    completed_at TIMESTAMPTZ,
    model_auroc FLOAT,
    model_version VARCHAR(50)
);

-- ══════════════════════════════════════════════════════════════════════════════
-- 5. ALERTS & AUDIT
-- ══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    user_id_hash VARCHAR(64) NOT NULL,
    prediction_id INT REFERENCES predictions(id),
    alert_level VARCHAR(20) NOT NULL CHECK (alert_level IN ('info', 'warning', 'critical')),
    alert_type VARCHAR(50) DEFAULT 'relapse_risk',
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100),
    action_taken TEXT,
    outcome VARCHAR(50),
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_channels JSONB
);

CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id_hash VARCHAR(64),
    actor VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    details JSONB,
    ip_address INET,
    user_agent TEXT
);

-- ══════════════════════════════════════════════════════════════════════════════
-- 6. TRIGGERS & PROCEDURES
-- ══════════════════════════════════════════════════════════════════════════════

-- updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Helper functions for encrypted fields (pgcrypto)
CREATE OR REPLACE FUNCTION encrypt_clinical_data(p_data TEXT, p_key TEXT)
RETURNS BYTEA AS $$
BEGIN
    RETURN pgp_sym_encrypt(p_data, p_key);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION decrypt_clinical_data(p_data BYTEA, p_key TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN pgp_sym_decrypt(p_data, p_key);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trigger_events_updated_at
    BEFORE UPDATE ON clinical_events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE TRIGGER trigger_label_settings_updated_at
    BEFORE UPDATE ON label_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- pending_changes increment
CREATE OR REPLACE FUNCTION update_pending_changes()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO label_settings (patient_id, pending_changes)
    VALUES (NEW.patient_id, 1)
    ON CONFLICT (patient_id) 
    DO UPDATE SET 
        pending_changes = label_settings.pending_changes + 1,
        updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trigger_increment_pending_changes
    AFTER INSERT OR UPDATE OR DELETE ON clinical_events
    FOR EACH ROW EXECUTE FUNCTION update_pending_changes();

-- ══════════════════════════════════════════════════════════════════════════════
-- 7. INDEXES
-- ══════════════════════════════════════════════════════════════════════════════

CREATE INDEX IF NOT EXISTS idx_events_patient_date ON clinical_events(patient_id, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_datapoints_user ON datapoints(user_id_hash, time DESC);
CREATE INDEX IF NOT EXISTS idx_raw_messages_hash ON raw_messages(content_hash);
CREATE INDEX IF NOT EXISTS idx_alerts_pending ON alerts(acknowledged_at) WHERE acknowledged_at IS NULL;
