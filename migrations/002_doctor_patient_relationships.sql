-- Migration: Add multi-tenant doctor-patient relationships
-- Run against empredictor database

-- 1. Add role column to users if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'users' AND column_name = 'role') THEN
        ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'patient';
    END IF;
END $$;

-- 2. Create doctor_patients relationship table
CREATE TABLE IF NOT EXISTS doctor_patients (
    id SERIAL PRIMARY KEY,
    doctor_id VARCHAR(64) NOT NULL,  -- user_id_hash of doctor
    patient_id VARCHAR(64) NOT NULL, -- user_id_hash of patient
    granted_at TIMESTAMPTZ DEFAULT NOW(),
    granted_by VARCHAR(64),  -- who approved the relationship
    access_level VARCHAR(20) DEFAULT 'read',  -- read, write, admin
    status VARCHAR(20) DEFAULT 'active',  -- active, pending, revoked
    revoked_at TIMESTAMPTZ,
    
    CONSTRAINT fk_doctor FOREIGN KEY (doctor_id) REFERENCES users(user_id_hash),
    CONSTRAINT fk_patient FOREIGN KEY (patient_id) REFERENCES users(user_id_hash),
    CONSTRAINT unique_doctor_patient UNIQUE (doctor_id, patient_id),
    CONSTRAINT valid_access CHECK (access_level IN ('read', 'write', 'admin')),
    CONSTRAINT valid_status CHECK (status IN ('active', 'pending', 'revoked'))
);

CREATE INDEX IF NOT EXISTS idx_doctor_patients_doctor ON doctor_patients(doctor_id);
CREATE INDEX IF NOT EXISTS idx_doctor_patients_patient ON doctor_patients(patient_id);
CREATE INDEX IF NOT EXISTS idx_doctor_patients_status ON doctor_patients(status);

-- 3. Ensure users table has all required columns for auth
DO $$
BEGIN
    -- Add email if not exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'users' AND column_name = 'email') THEN
        ALTER TABLE users ADD COLUMN email VARCHAR(255) UNIQUE;
    END IF;
    
    -- Add password_hash if not exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'users' AND column_name = 'password_hash') THEN
        ALTER TABLE users ADD COLUMN password_hash TEXT;
    END IF;
    
    -- Add name if not exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'users' AND column_name = 'name') THEN
        ALTER TABLE users ADD COLUMN name VARCHAR(255);
    END IF;
END $$;

-- 4. Create view for doctors with their patient counts
CREATE OR REPLACE VIEW doctor_dashboard AS
SELECT 
    d.user_id_hash as doctor_id,
    d.name as doctor_name,
    d.email as doctor_email,
    COUNT(dp.patient_id) as patient_count,
    COUNT(dp.patient_id) FILTER (WHERE dp.status = 'pending') as pending_requests
FROM users d
LEFT JOIN doctor_patients dp ON d.user_id_hash = dp.doctor_id AND dp.status IN ('active', 'pending')
WHERE d.role = 'doctor'
GROUP BY d.user_id_hash, d.name, d.email;

COMMENT ON TABLE doctor_patients IS 'Relationship table linking doctors to their patients';
COMMENT ON COLUMN doctor_patients.access_level IS 'read: view only, write: can add events, admin: full access';
