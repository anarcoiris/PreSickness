-- ============================================================================
-- Schema de Eventos Clínicos - MS-Predictor
-- Versión 1.0
-- ============================================================================

-- Tabla principal de eventos clínicos
CREATE TABLE IF NOT EXISTS clinical_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(64) NOT NULL,
    
    -- Datos del evento
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
    
    -- Para relapses: fecha de inicio de medicación
    medication_start_date TIMESTAMPTZ,
    
    -- Origen del evento
    source VARCHAR(20) DEFAULT 'manual' CHECK (source IN (
        'manual',
        'auto_detected',
        'imported'
    )),
    cluster_id UUID,  -- Referencia al cluster si fue auto-detectado
    
    -- Validación clínica
    validated_by VARCHAR(64),  -- ID del usuario que validó
    validated_at TIMESTAMPTZ,
    validation_role VARCHAR(20) CHECK (validation_role IN ('patient', 'doctor')),
    
    -- Flag para indicar si requiere re-entrenamiento
    requires_retraining BOOLEAN DEFAULT TRUE,
    
    -- Metadatos
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(64),
    
    -- Foreign key a patients (si existe)
    -- CONSTRAINT fk_patient FOREIGN KEY (patient_id) 
    --     REFERENCES patients(id) ON DELETE CASCADE
    
    -- Índice único para evitar duplicados
    CONSTRAINT unique_patient_event UNIQUE (patient_id, event_date, event_type)
);

-- Tabla de clusters auto-detectados
CREATE TABLE IF NOT EXISTS auto_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(64) NOT NULL,
    
    -- Rango temporal del cluster
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    peak_date DATE NOT NULL,
    
    -- Métricas de detección
    total_signals INT,
    unique_types INT,
    max_severity VARCHAR(20),
    severity_score FLOAT,
    density FLOAT,
    
    -- Clasificación automática
    is_probable_relapse BOOLEAN DEFAULT FALSE,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Estado de validación
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN (
        'pending',
        'validated',
        'rejected'
    )),
    
    -- Referencia al evento creado si fue validado
    validated_event_id UUID REFERENCES clinical_events(id) ON DELETE SET NULL,
    
    -- Usuario que procesó el cluster
    processed_by VARCHAR(64),
    processed_at TIMESTAMPTZ,
    rejection_reason TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Configuración de labels por paciente
CREATE TABLE IF NOT EXISTS label_settings (
    patient_id VARCHAR(64) PRIMARY KEY,
    
    -- Horizontes de predicción activos (en días)
    horizons JSONB DEFAULT '[7, 14, 30]',
    
    -- Tipos de eventos que cuentan como labels positivos
    label_event_types JSONB DEFAULT '["confirmed_relapse", "medication_start"]',
    
    -- Días a censurar al final del dataset (labels incompletos)
    censor_days_before_end INT DEFAULT 30,
    
    -- Usar clusters auto-detectados como labels
    use_auto_clusters BOOLEAN DEFAULT FALSE,
    auto_cluster_min_confidence FLOAT DEFAULT 0.8,
    
    -- Último re-entrenamiento
    last_labels_generated_at TIMESTAMPTZ,
    pending_changes INT DEFAULT 0,  -- Contador de cambios pendientes
    
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Historial de re-entrenamientos
CREATE TABLE IF NOT EXISTS retraining_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(64) NOT NULL,
    
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    triggered_by VARCHAR(64),
    
    -- Estadísticas del entrenamiento
    events_used INT,
    labels_positive INT,
    labels_negative INT,
    
    -- Resultado
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN (
        'pending',
        'running',
        'completed',
        'failed'
    )),
    error_message TEXT,
    completed_at TIMESTAMPTZ,
    
    -- Métricas del modelo resultante
    model_auroc FLOAT,
    model_version VARCHAR(50)
);

-- ============================================================================
-- ÍNDICES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_events_patient_date 
    ON clinical_events(patient_id, event_date DESC);

CREATE INDEX IF NOT EXISTS idx_events_type 
    ON clinical_events(event_type);

CREATE INDEX IF NOT EXISTS idx_events_requires_retraining 
    ON clinical_events(patient_id, requires_retraining) 
    WHERE requires_retraining = TRUE;

CREATE INDEX IF NOT EXISTS idx_clusters_patient_status 
    ON auto_clusters(patient_id, status);

CREATE INDEX IF NOT EXISTS idx_clusters_pending 
    ON auto_clusters(status) 
    WHERE status = 'pending';

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger para actualizar updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trigger_events_updated_at
    BEFORE UPDATE ON clinical_events
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Trigger para incrementar pending_changes en label_settings
CREATE OR REPLACE FUNCTION increment_pending_changes()
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

CREATE OR REPLACE TRIGGER trigger_event_pending_changes
    AFTER INSERT OR UPDATE OR DELETE ON clinical_events
    FOR EACH ROW
    EXECUTE FUNCTION increment_pending_changes();

-- ============================================================================
-- VISTAS
-- ============================================================================

-- Vista de resumen de eventos por paciente
CREATE OR REPLACE VIEW patient_events_summary AS
SELECT 
    patient_id,
    COUNT(*) as total_events,
    COUNT(*) FILTER (WHERE event_type = 'confirmed_relapse') as confirmed_relapses,
    COUNT(*) FILTER (WHERE event_type = 'medication_start') as medication_starts,
    COUNT(*) FILTER (WHERE validated_at IS NOT NULL) as validated_events,
    MIN(event_date) as first_event,
    MAX(event_date) as last_event,
    COUNT(*) FILTER (WHERE requires_retraining = TRUE) as pending_retraining
FROM clinical_events
GROUP BY patient_id;

-- Vista de clusters pendientes de validación
CREATE OR REPLACE VIEW pending_clusters AS
SELECT 
    c.*,
    CASE 
        WHEN c.is_probable_relapse THEN 'high'
        WHEN c.severity_score > 50 THEN 'medium'
        ELSE 'low'
    END as priority
FROM auto_clusters c
WHERE c.status = 'pending'
ORDER BY c.is_probable_relapse DESC, c.severity_score DESC;
