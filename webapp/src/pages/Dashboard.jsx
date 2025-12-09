import { useState, useEffect } from 'react';
import { patientAPI, predictAPI, systemAPI } from '../api/client';

export default function Dashboard() {
    const [profile, setProfile] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            const [profileData, metricsData] = await Promise.all([
                patientAPI.getProfile(),
                systemAPI.metrics().catch(() => null),
            ]);
            setProfile(profileData);
            setMetrics(metricsData);
        } catch (err) {
            console.error('Error loading data:', err);
        } finally {
            setLoading(false);
        }
    };

    const runPrediction = async () => {
        try {
            const result = await predictAPI.predict(14);
            setPrediction(result);
        } catch (err) {
            console.error('Prediction error:', err);
        }
    };

    if (loading) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '4rem' }}>
                <div className="spinner" />
            </div>
        );
    }

    return (
        <div className="fade-in">
            <div className="page-header">
                <h1 className="page-title">
                    ¡Hola, {profile?.name || 'Usuario'}! 👋
                </h1>
                <p className="page-subtitle">
                    Panel de seguimiento de tu salud
                </p>
            </div>

            {/* Métricas principales */}
            <div className="grid grid-4" style={{ marginBottom: '2rem' }}>
                <div className="card metric-card">
                    <div className="metric-value primary">
                        {prediction ? `${Math.round(prediction.probability * 100)}%` : '--'}
                    </div>
                    <div className="metric-label">Riesgo Estimado</div>
                </div>

                <div className="card metric-card">
                    <div className="metric-value success">
                        {metrics?.total_uploads || 0}
                    </div>
                    <div className="metric-label">Datos Subidos</div>
                </div>

                <div className="card metric-card">
                    <div className="metric-value" style={{ color: 'var(--secondary)' }}>
                        14
                    </div>
                    <div className="metric-label">Días de Horizonte</div>
                </div>

                <div className="card metric-card">
                    <div className="metric-value" style={{
                        color: metrics?.services_status?.ml_inference === 'ok'
                            ? 'var(--success)'
                            : 'var(--warning)'
                    }}>
                        {metrics?.services_status?.ml_inference === 'ok' ? '✓' : '○'}
                    </div>
                    <div className="metric-label">Modelo ML</div>
                </div>
            </div>

            {/* Predicción */}
            <div className="grid grid-2">
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">🎯 Predicción de Riesgo</h3>
                        <button className="btn btn-primary" onClick={runPrediction}>
                            Calcular
                        </button>
                    </div>

                    {prediction ? (
                        <div style={{ textAlign: 'center', padding: '2rem 0' }}>
                            <div className={`risk-indicator ${prediction.risk_level}`} style={{ fontSize: '1.25rem', padding: '1rem 2rem' }}>
                                {prediction.risk_level === 'ok' && '🟢 Riesgo Bajo'}
                                {prediction.risk_level === 'warning' && '🟡 Riesgo Moderado'}
                                {prediction.risk_level === 'critical' && '🔴 Riesgo Alto'}
                            </div>
                            <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>
                                Probabilidad: {(prediction.probability * 100).toFixed(1)}%
                            </p>
                            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                                Horizonte: {prediction.horizon_days} días
                            </p>
                        </div>
                    ) : (
                        <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                            <p>Haz clic en "Calcular" para obtener tu predicción</p>
                            <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                                Basado en tus datos más recientes
                            </p>
                        </div>
                    )}
                </div>

                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">📊 Estado del Sistema</h3>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {['api_gateway', 'ml_inference'].map((service) => (
                            <div key={service} style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                padding: '0.75rem',
                                background: 'var(--bg-dark)',
                                borderRadius: '8px'
                            }}>
                                <span style={{ textTransform: 'capitalize' }}>
                                    {service.replace('_', ' ')}
                                </span>
                                <span className={`status-badge ${metrics?.services_status?.[service] === 'ok' ? 'online' : 'offline'}`}>
                                    <span style={{
                                        width: '8px',
                                        height: '8px',
                                        borderRadius: '50%',
                                        background: 'currentColor'
                                    }} />
                                    {metrics?.services_status?.[service] || 'unknown'}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Info adicional */}
            <div className="card" style={{ marginTop: '1.5rem' }}>
                <div className="card-header">
                    <h3 className="card-title">💡 Recomendaciones</h3>
                </div>
                <ul style={{
                    listStyle: 'none',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.75rem'
                }}>
                    <li style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
                        <span>📤</span>
                        <span>Sube tus datos de salud regularmente para mejores predicciones</span>
                    </li>
                    <li style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
                        <span>⏰</span>
                        <span>Revisa tu predicción al menos una vez por semana</span>
                    </li>
                    <li style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
                        <span>📝</span>
                        <span>Mantén un registro de tus síntomas y actividad física</span>
                    </li>
                </ul>
            </div>
        </div>
    );
}
