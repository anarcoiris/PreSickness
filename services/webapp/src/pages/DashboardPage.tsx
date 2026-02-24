import { useQuery, useMutation } from '@tanstack/react-query';
import { authApi, predictApi, systemApi } from '../api/client';
import { Activity, Database, CheckCircle, AlertCircle, TrendingUp, Cpu } from 'lucide-react';

export default function DashboardPage() {
    const { data: profile } = useQuery({
        queryKey: ['profile'],
        queryFn: () => authApi.getProfile().then(res => res.data),
    });

    const { data: metrics } = useQuery({
        queryKey: ['metrics'],
        queryFn: () => systemApi.metrics().then(res => res.data),
        refetchInterval: 30000,
    });

    const predictMutation = useMutation({
        mutationFn: (horizon: number) => predictApi.predict(horizon).then(res => res.data),
    });

    const prediction = predictMutation.data;
    const isLoading = predictMutation.isPending;

    return (
        <div className="fade-in">
            <div className="page-header">
                <h1 className="page-title">
                    ¡Hola, {profile?.name || 'Usuario'}! 👋
                </h1>
                <p className="page-subtitle">
                    Panel de seguimiento de tu salud y predicción de brotes.
                </p>
            </div>

            {/* Métricas principales */}
            <div className="grid grid-4" style={{ marginBottom: '2rem' }}>
                <div className="card metric-card">
                    <div className="metric-header">
                        <TrendingUp size={20} className="text-primary" />
                        <span className="metric-label">Riesgo Estimado</span>
                    </div>
                    <div className="metric-value primary">
                        {prediction ? `${Math.round(prediction.probability * 100)}%` : '--'}
                    </div>
                </div>

                <div className="card metric-card">
                    <div className="metric-header">
                        <Database size={20} className="text-success" />
                        <span className="metric-label">Datos Subidos</span>
                    </div>
                    <div className="metric-value success">
                        {metrics?.total_uploads || 0}
                    </div>
                </div>

                <div className="card metric-card">
                    <div className="metric-header">
                        <Activity size={20} className="text-secondary" />
                        <span className="metric-label">Mensajes Procesados</span>
                    </div>
                    <div className="metric-value" style={{ color: 'var(--secondary)' }}>
                        {metrics?.total_messages?.toLocaleString() || 0}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                        {metrics?.nlp_processed?.toLocaleString() || 0} con NLP
                    </div>
                </div>

                <div className="card metric-card">
                    <div className="metric-header">
                        <Cpu size={20} className={metrics?.services_status?.ml_inference === 'ok' ? 'text-success' : 'text-warning'} />
                        <span className="metric-label">Modelo ML</span>
                    </div>
                    <div className="metric-value" style={{
                        color: metrics?.services_status?.ml_inference === 'ok'
                            ? 'var(--success)'
                            : 'var(--warning)'
                    }}>
                        {metrics?.services_status?.ml_inference === 'ok' ? 'Activo' : 'Inactivo'}
                    </div>
                </div>
            </div>

            {/* Predicción */}
            <div className="grid grid-2">
                <div className="card">
                    <div className="card-header">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <TrendingUp size={24} color="var(--primary)" />
                            <h3 className="card-title">Predicción de Riesgo</h3>
                        </div>
                        <button
                            className="btn btn-primary"
                            onClick={() => predictMutation.mutate(14)}
                            disabled={isLoading}
                        >
                            {isLoading ? 'Calculando...' : 'Calcular'}
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
                                Basado en tus datos más recientes procesados por el motor EM-Predictor.
                            </p>
                        </div>
                    )}
                </div>

                <div className="card">
                    <div className="card-header">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <Activity size={24} color="var(--secondary)" />
                            <h3 className="card-title">Estado del Sistema</h3>
                        </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {['api_gateway', 'ml_inference', 'nlp_agent'].map((service) => (
                            <div key={service} style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                padding: '0.75rem',
                                background: 'rgba(255,255,255,0.03)',
                                borderRadius: '8px',
                                border: '1px solid rgba(255,255,255,0.05)'
                            }}>
                                <span style={{ textTransform: 'capitalize' }}>
                                    {service.replace('_', ' ')}
                                </span>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    {metrics?.services_status?.[service] === 'ok' ? (
                                        <CheckCircle size={16} color="var(--success)" />
                                    ) : (
                                        <AlertCircle size={16} color="var(--warning)" />
                                    )}
                                    <span style={{
                                        color: metrics?.services_status?.[service] === 'ok' ? 'var(--success)' : 'var(--warning)',
                                        fontSize: '0.875rem',
                                        fontWeight: 500
                                    }}>
                                        {metrics?.services_status?.[service] || 'unknown'}
                                    </span>
                                </div>
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
                <ul className="info-list">
                    <li>
                        <Database size={18} />
                        <span>Sube tus datos de salud regularmente para mejorar la precisión del modelo.</span>
                    </li>
                    <li>
                        <Activity size={18} />
                        <span>Revisa tu predicción al menos una vez por semana o ante nuevos síntomas.</span>
                    </li>
                    <li>
                        <CheckCircle size={18} />
                        <span>Los resultados son orientativos. Consulta siempre con tu neurólogo.</span>
                    </li>
                </ul>
            </div>
        </div>
    );
}
