-- Database schema para EM Predictor
-- PostgreSQL 15+ con extensiones de cifrado

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- === Users & Consent ===

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    user_id_hash VARCHAR(64) UNIQUE NOT NULL,  -- SHA-256 of user_id
    created_at TIMESTAMPTZ DEFAULT NOW(),
    consent_given_at TIMESTAMPTZ NOT NULL,
    consent_revoked_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'active',  -- active, paused, deleted
    
    -- Clinical metadata (encrypted)
    age_encrypted BYTEA,
    gender_encrypted BYTEA,
    ms_type_encrypted BYTEA,  -- RRMS, SPMS, PPMS
    edss_baseline_encrypted BYTEA,
    
    -- Retention policy
    data_retention_days INT DEFAULT 730,  -- 2 years
    
    CONSTRAINT valid_status CHECK (status IN ('active', 'paused', 'deleted'))
);

CREATE INDEX idx_users_hash ON users(user_id_hash);
CREATE INDEX idx_users_status ON users(status) WHERE status = 'active';


-- === Devices ===

CREATE TABLE devices (
    id SERIAL PRIMARY KEY,
    device_id_hash VARCHAR(64) UNIQUE NOT NULL,
    user_id_hash VARCHAR(64) REFERENCES users(user_id_hash),
    secret TEXT NOT NULL,  -- HMAC secret (encrypted at rest)
    registered_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ,
    device_type VARCHAR(50),  -- android, ios
    app_version VARCHAR(20),
    
    -- Permissions granted
    permissions JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_devices_user ON devices(user_id_hash);


-- === Time-series Data (Hypertable with TimescaleDB) ===

CREATE TABLE datapoints (
    time TIMESTAMPTZ NOT NULL,
    user_id_hash VARCHAR(64) NOT NULL,
    device_id_hash VARCHAR(64) NOT NULL,
    
    -- Encrypted embedding
    embedding_encrypted TEXT NOT NULL,
    embedding_dim INT DEFAULT 768,
    embedding_salt VARCHAR(32),
    
    -- Numeric features (stored as JSONB for flexibility)
    numeric_features JSONB NOT NULL,
    
    -- Metadata
    data_quality_score FLOAT,  -- 0-1, computed by validator
    
    PRIMARY KEY (time, user_id_hash)
);

-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable('datapoints', 'time', chunk_time_interval => INTERVAL '1 day');

-- Indexes for common queries
CREATE INDEX idx_datapoints_user ON datapoints(user_id_hash, time DESC);
CREATE INDEX idx_datapoints_quality ON datapoints(data_quality_score) WHERE data_quality_score < 0.5;

-- Retention policy: auto-delete after retention period
SELECT add_retention_policy('datapoints', INTERVAL '2 years');


-- === Clinical Events (Ground Truth) ===

CREATE TABLE clinical_events (
    id SERIAL PRIMARY KEY,
    user_id_hash VARCHAR(64) NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- relapse, pseudorelapse, mri_activity
    event_date DATE NOT NULL,
    reported_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Clinical details (encrypted)
    edss_score_encrypted BYTEA,
    symptoms_encrypted BYTEA,  -- JSON encrypted
    treatment_encrypted BYTEA,
    
    -- Source of truth
    source VARCHAR(50),  -- patient_reported, clinician_confirmed, mri
    confidence VARCHAR(20) DEFAULT 'medium',  -- low, medium, high
    
    -- Link to medical records
    ehr_reference_encrypted BYTEA,
    
    CONSTRAINT valid_event_type CHECK (event_type IN ('relapse', 'pseudorelapse', 'mri_activity', 'hospitalization'))
);

CREATE INDEX idx_events_user_date ON clinical_events(user_id_hash, event_date DESC);
CREATE INDEX idx_events_type ON clinical_events(event_type);


-- === Features Store (Aggregated) ===

CREATE TABLE feature_windows (
    user_id_hash VARCHAR(64) NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    window_size_days INT NOT NULL,  -- 1, 3, 7, 14, 30
    
    -- Aggregated features
    features JSONB NOT NULL,  -- {embedding_mean: [...], sentiment_mean: 0.15, ...}
    feature_version VARCHAR(20) DEFAULT 'v1',
    
    -- Metadata
    num_datapoints INT,
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (user_id_hash, window_end, window_size_days)
);

CREATE INDEX idx_features_user_window ON feature_windows(user_id_hash, window_end DESC);


-- === Model Predictions ===

CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    user_id_hash VARCHAR(64) NOT NULL,
    prediction_date DATE NOT NULL,
    horizon_days INT NOT NULL,  -- 7, 14, 30
    
    -- Prediction outputs
    relapse_probability FLOAT NOT NULL,  -- 0-1
    confidence_interval JSONB,  -- {lower: 0.1, upper: 0.3}
    
    -- Model metadata
    model_version VARCHAR(50) NOT NULL,
    model_name VARCHAR(50),  -- tft_ensemble_v2
    
    -- Explainability
    feature_importance JSONB,  -- SHAP values
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_probability CHECK (relapse_probability BETWEEN 0 AND 1)
);

CREATE INDEX idx_predictions_user ON predictions(user_id_hash, prediction_date DESC);
CREATE INDEX idx_predictions_threshold ON predictions(relapse_probability) WHERE relapse_probability > 0.3;


-- === Alerts ===

CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    user_id_hash VARCHAR(64) NOT NULL,
    prediction_id INT REFERENCES predictions(id),
    
    alert_level VARCHAR(20) NOT NULL,  -- info, warning, critical
    alert_type VARCHAR(50) DEFAULT 'relapse_risk',
    
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100),  -- clinician_id_hash
    
    -- Action taken
    action_taken TEXT,
    outcome VARCHAR(50),  -- true_positive, false_positive, pending
    
    -- Notification status
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_channels JSONB,  -- {sms: true, push: true, email: false}
    
    CONSTRAINT valid_alert_level CHECK (alert_level IN ('info', 'warning', 'critical'))
);

CREATE INDEX idx_alerts_user ON alerts(user_id_hash, triggered_at DESC);
CREATE INDEX idx_alerts_pending ON alerts(acknowledged_at) WHERE acknowledged_at IS NULL;


-- === Audit Log (immutable) ===

CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id_hash VARCHAR(64),
    actor VARCHAR(100),  -- system, clinician_id, patient_id
    action VARCHAR(100) NOT NULL,  -- data_access, consent_given, consent_revoked, etc.
    
    -- Details
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    
    -- Immutable
    CONSTRAINT no_update CHECK (timestamp = NOW())
);

CREATE INDEX idx_audit_user ON audit_log(user_id_hash, timestamp DESC);
CREATE INDEX idx_audit_action ON audit_log(action, timestamp DESC);


-- === Model Registry ===

CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    
    -- Training metadata
    trained_at TIMESTAMPTZ DEFAULT NOW(),
    training_samples INT,
    validation_auroc FLOAT,
    validation_auprc FLOAT,
    
    -- Artifacts
    artifact_uri TEXT NOT NULL,  -- S3/MinIO path
    config JSONB,
    
    -- Deployment
    status VARCHAR(20) DEFAULT 'candidate',  -- candidate, staging, production, retired
    deployed_at TIMESTAMPTZ,
    
    UNIQUE(model_name, version)
);


-- === Helper Functions ===

-- Function to get recent features for a user
CREATE OR REPLACE FUNCTION get_recent_features(
    p_user_id_hash VARCHAR,
    p_days INT DEFAULT 30
)
RETURNS TABLE (
    timestamp TIMESTAMPTZ,
    numeric_features JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT time, numeric_features
    FROM datapoints
    WHERE user_id_hash = p_user_id_hash
      AND time >= NOW() - (p_days || ' days')::INTERVAL
    ORDER BY time DESC;
END;
$$ LANGUAGE plpgsql;


-- Function to check if user has active consent
CREATE OR REPLACE FUNCTION has_active_consent(p_user_id_hash VARCHAR)
RETURNS BOOLEAN AS $$
    SELECT EXISTS (
        SELECT 1 FROM users
        WHERE user_id_hash = p_user_id_hash
          AND status = 'active'
          AND consent_revoked_at IS NULL
    );
$$ LANGUAGE sql;


-- === Initial Data (for testing) ===

-- Sample user (hash of "test_user_001")
INSERT INTO users (user_id_hash, consent_given_at, age_encrypted, gender_encrypted, ms_type_encrypted)
VALUES (
    'c24d8c0e96c58f0f0d2c3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b',
    NOW(),
    pgp_sym_encrypt('35', 'encryption_key'),
    pgp_sym_encrypt('F', 'encryption_key'),
    pgp_sym_encrypt('RRMS', 'encryption_key')
);
