import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8080';

const api = axios.create({
    baseURL: API_BASE,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Interceptor para añadir token JWT
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// Interceptor para manejar errores de auth
api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            localStorage.removeItem('token');
            window.location.href = '/login';
        }
        return Promise.reject(error);
    }
);

// ══════════════════════════════════════════════════════════════════════════════
// AUTH
// ══════════════════════════════════════════════════════════════════════════════

export const authAPI = {
    register: async (email, password, name) => {
        const response = await api.post('/api/auth/register', { email, password, name });
        return response.data;
    },

    login: async (email, password) => {
        const formData = new URLSearchParams();
        formData.append('username', email);
        formData.append('password', password);

        const response = await api.post('/api/auth/login', formData, {
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        });

        localStorage.setItem('token', response.data.access_token);
        return response.data;
    },

    logout: () => {
        localStorage.removeItem('token');
    },

    isAuthenticated: () => {
        return !!localStorage.getItem('token');
    },
};

// ══════════════════════════════════════════════════════════════════════════════
// PATIENT
// ══════════════════════════════════════════════════════════════════════════════

export const patientAPI = {
    getProfile: async () => {
        const response = await api.get('/api/patients/me');
        return response.data;
    },

    uploadData: async (file) => {
        const formData = new FormData();
        formData.append('file', file);

        const response = await api.post('/api/patients/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return response.data;
    },

    listUploads: async () => {
        const response = await api.get('/api/patients/data');
        return response.data;
    },
};

// ══════════════════════════════════════════════════════════════════════════════
// PREDICTIONS
// ══════════════════════════════════════════════════════════════════════════════

export const predictAPI = {
    predict: async (horizonDays = 14) => {
        const response = await api.post('/api/predict', { horizon_days: horizonDays });
        return response.data;
    },
};

// ══════════════════════════════════════════════════════════════════════════════
// SYSTEM
// ══════════════════════════════════════════════════════════════════════════════

export const systemAPI = {
    health: async () => {
        const response = await api.get('/health');
        return response.data;
    },

    metrics: async () => {
        const response = await api.get('/api/metrics');
        return response.data;
    },

    getAlerts: async () => {
        const response = await api.get('/api/alerts');
        return response.data;
    },
};

export default api;
