import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

export const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Gestión de contexto (paciente seleccionado para médicos)
export const patientContext = {
    selectedPatientId: localStorage.getItem('selected_patient_id'),
    setPatient: (id: string | null) => {
        if (id) {
            localStorage.setItem('selected_patient_id', id);
            patientContext.selectedPatientId = id;
        } else {
            localStorage.removeItem('selected_patient_id');
            patientContext.selectedPatientId = null;
        }
    }
};

// Interceptor para añadir token y contexto
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }

    // Si hay un paciente seleccionado y tenemos rol de médico (o queremos impersonar)
    // El backend validará si tenemos permiso
    if (patientContext.selectedPatientId) {
        config.headers['X-Patient-ID'] = patientContext.selectedPatientId;
    }

    return config;
});

// Tipos
export interface Event {
    id: string;
    patient_id: string;
    event_date: string;
    event_type: 'symptom_onset' | 'confirmed_relapse' | 'medication_start' | 'hospital_visit' | 'doctor_appointment';
    severity?: 'mild' | 'moderate' | 'severe';
    notes?: string;
    medication_start_date?: string;
    source: 'manual' | 'auto_detected' | 'imported';
    validated_by?: string;
    validated_at?: string;
    validation_role?: 'patient' | 'doctor';
    requires_retraining: boolean;
    created_at: string;
    updated_at: string;
}

export interface Cluster {
    id: string;
    patient_id: string;
    start_date: string;
    end_date: string;
    peak_date: string;
    total_signals: number;
    unique_types: number;
    max_severity?: string;
    severity_score: number;
    density: number;
    is_probable_relapse: boolean;
    confidence?: number;
    status: 'pending' | 'validated' | 'rejected';
    created_at: string;
}

export interface EventStats {
    total_events: number;
    confirmed_relapses: number;
    medication_starts: number;
    pending_clusters: number;
    pending_retraining: number;
    last_event_date?: string;
}

export interface RetrainingStatus {
    pending_changes: number;
    requires_retraining: boolean;
    last_trained_at?: string;
}

export interface LabelSettings {
    patient_id: string;
    horizons: number[];
    label_event_types: string[];
    censor_days_before_end: number;
    use_auto_clusters: boolean;
    auto_cluster_min_confidence: number;
    pending_changes: number;
    last_labels_generated_at?: string;
}

export interface SystemIncident {
    id: string;
    severity: 'info' | 'warning' | 'error' | 'critical';
    component: string;
    message: string;
    details?: any;
    resolved: boolean;
    resolved_at?: string;
    created_at: string;
}

export interface UserResponse {
    id: string;
    email: string;
    name: string;
    role: 'patient' | 'doctor' | 'admin';
    created_at: string;
}

// API de Eventos
export const eventsApi = {
    // CRUD
    list: (params?: { event_type?: string; start_date?: string; end_date?: string }) =>
        api.get<Event[]>('/api/events/', { params }),

    get: (id: string) =>
        api.get<Event>(`/api/events/${id}`),

    create: (data: Partial<Event>) =>
        api.post<Event>('/api/events/', data),

    update: (id: string, data: Partial<Event>) =>
        api.put<Event>(`/api/events/${id}`, data),

    delete: (id: string) =>
        api.delete(`/api/events/${id}`),

    // Clusters
    listClusters: (status?: string) =>
        api.get<Cluster[]>('/api/events/clusters', { params: { status } }),

    validateCluster: (id: string, data: Partial<Event>) =>
        api.post<Event>(`/api/events/clusters/${id}/validate`, data),

    rejectCluster: (id: string, reason: string) =>
        api.post(`/api/events/clusters/${id}/reject`, { reason }),

    // Estadísticas
    getStats: () =>
        api.get<EventStats>('/api/events/stats'),

    getRetrainingStatus: () =>
        api.get<RetrainingStatus>('/api/events/retraining-status'),

    triggerRetraining: () =>
        api.post('/api/events/trigger-retraining'),

    // Configuración
    getSettings: () =>
        api.get<LabelSettings>('/api/events/settings'),

    updateSettings: (data: Partial<LabelSettings>) =>
        api.put<LabelSettings>('/api/events/settings', data),

    // Importación
    previewImport: (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        return api.post('/api/events/import/preview', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
    },

    confirmImport: (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        return api.post('/api/events/import/confirm', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
    },
};

// Auth API
export const authApi = {
    login: (email: string, password: string) =>
        api.post('/api/auth/login', new URLSearchParams({ username: email, password }), {
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        }),

    register: (email: string, password: string, name: string, role: string = 'patient') =>
        api.post('/api/auth/register', { email, password, name, role }),

    getProfile: () =>
        api.get('/api/patients/me'),

    loginWithGoogle: (code: string, role: string = 'patient', redirect_uri: string = 'postmessage') =>
        api.post('/api/auth/google', { code, role, redirect_uri }),
};

// Prediction API
export interface PredictionResult {
    probability: number;
    risk_level: 'ok' | 'warning' | 'critical';
    horizon_days: number;
    generated_at: string;
}

export const predictApi = {
    predict: (horizonDays: number = 14) =>
        api.post<PredictionResult>('/api/predict', { horizon_days: horizonDays }),

    listHistory: (limit: number = 30) =>
        api.get<PredictionResult[]>('/api/predict/history', { params: { limit } }),
};

// Patient Data API
export interface PatientUpload {
    id: string;
    filename: string;
    uploaded_at: string;
    processed: boolean;
    error_message?: string;
}

export interface DoctorInfo {
    doctor_id: string;
    doctor_name: string;
    doctor_email: string;
    granted_at: string;
    status: string;
}

export const patientApi = {
    uploadData: (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        return api.post<PatientUpload>('/api/patients/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
    },

    listUploads: () =>
        api.get<PatientUpload[]>('/api/patients/data'),

    getMessageStats: () =>
        api.get<{ raw_messages: number; processed_datapoints: number; nlp_processed: number }>('/api/events/messages/stats'),

    triggerProcessing: () =>
        api.post<{ processed: number; errors: number; skipped: number }>('/api/events/messages/process'),

    listDoctors: () =>
        api.get<DoctorInfo[]>('/api/patient/doctors'),
    revokeDoctor: (doctorId: string) =>
        api.delete(`/api/patient/doctors/${doctorId}`),
};

// Doctor API
export interface DoctorPatient {
    patient_id: string;
    patient_name: string;
    patient_email: string;
    granted_at: string;
    access_level: string;
    status: string;
}

export const doctorApi = {
    listPatients: () =>
        api.get<DoctorPatient[]>('/api/doctor/patients'),

    addPatient: (email: string) =>
        api.post<DoctorPatient>('/api/doctor/patients', { patient_email: email }),

    removePatient: (patientId: string) =>
        api.delete(`/api/doctor/patients/${patientId}`),
};

// System API
export interface SystemMetrics {
    total_uploads: number;
    total_messages: number;
    total_datapoints: number;
    nlp_processed: number;
    services_status: {
        api_gateway: string;
        ml_inference: string;
        nlp_agent: string;
        [key: string]: string;
    };
}

export const systemApi = {
    health: () =>
        api.get('/health'),

    metrics: () =>
        api.get<SystemMetrics>('/api/metrics'),

    getAlerts: () =>
        api.get('/api/alerts'),
};

export const analysisApi = {
    getTrainingResults: () => api.get('/api/analysis/training'),
    getEnsembleResults: () => api.get('/api/analysis/ensemble'),
    getOptunaResults: () => api.get('/api/analysis/optuna'),
    getFeatureImportance: () => api.get('/api/analysis/features'),
};

export const adminApi = {
    listUsers: () =>
        api.get<UserResponse[]>('/api/admin/users'),

    updateUserRole: (userId: string, role: string) =>
        api.put(`/api/admin/users/${userId}/role`, null, { params: { role } }),

    listIncidents: (includeResolved: boolean = false) =>
        api.get<SystemIncident[]>('/api/admin/incidents', { params: { include_resolved: includeResolved } }),

    resolveIncident: (id: string) =>
        api.post(`/api/admin/incidents/${id}/resolve`),
};

