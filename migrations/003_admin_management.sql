-- === System Incidents ===

CREATE TABLE IF NOT EXISTS system_incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    component VARCHAR(50) NOT NULL, -- backend, ml_inference, nlp_agent, scheduler
    message TEXT NOT NULL,
    details JSONB,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_incidents_resolved ON system_incidents(resolved) WHERE resolved = FALSE;
CREATE INDEX idx_incidents_severity ON system_incidents(severity);

-- Add some sample incidents if they don't exist
INSERT INTO system_incidents (severity, component, message)
VALUES 
('warning', 'ml_inference', 'Modelo TFT cargado con heurística fallback debido a falta de artefactos en MLflow'),
('info', 'scheduler', 'Limpieza de base de datos completada satisfactoriamente');
