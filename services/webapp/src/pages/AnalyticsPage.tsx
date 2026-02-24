import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { predictApi } from '../api/client';
import type { PredictionResult } from '../api/client';
import { TrendingUp, Calendar, Info, BarChart2 } from 'lucide-react';

export default function AnalyticsPage() {
    const [horizonDays, setHorizonDays] = useState(14);

    // Mock history for prototype demonstration
    const queryClient = useQueryClient();
    const { data: history = [] } = useQuery({
        queryKey: ['prediction-history'],
        queryFn: () => predictApi.listHistory().then(res => res.data),
    });

    const predictMutation = useMutation({
        mutationFn: (h: number) => predictApi.predict(h).then(res => res.data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['prediction-history'] });
            queryClient.invalidateQueries({ queryKey: ['alerts'] });
        }
    });

    const currentPrediction = predictMutation.data || history[0];
    const isLoading = predictMutation.isPending;

    const getRiskColor = (prob: number) => {
        if (prob >= 0.55) return 'var(--danger)';
        if (prob >= 0.35) return 'var(--warning)';
        return 'var(--success)';
    };

    return (
        <div className="fade-in">
            <div className="page-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <BarChart2 size={32} color="var(--primary)" />
                    <h1 className="page-title">Analytics</h1>
                </div>
                <p className="page-subtitle">
                    Análisis detallado de tendencias de riesgo y evolución temporal.
                </p>
            </div>

            <div className="grid grid-2" style={{ marginBottom: '2rem' }}>
                <div className="card">
                    <div className="card-header">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <TrendingUp size={24} color="var(--primary)" />
                            <h3 className="card-title">Nueva Predicción</h3>
                        </div>
                    </div>

                    <div style={{ marginBottom: '1.5rem' }}>
                        <label className="form-label">Horizonte de predicción (días)</label>
                        <select
                            className="form-input"
                            value={horizonDays}
                            onChange={(e) => setHorizonDays(parseInt(e.target.value))}
                        >
                            <option value={7}>7 días (Corto plazo)</option>
                            <option value={14}>14 días (Estándar)</option>
                            <option value={21}>21 días (Extendido)</option>
                            <option value={30}>30 días (Mes completo)</option>
                        </select>
                    </div>

                    <button
                        className="btn btn-primary"
                        style={{ width: '100%' }}
                        onClick={() => predictMutation.mutate(horizonDays)}
                        disabled={isLoading}
                    >
                        {isLoading ? 'Calculando...' : '🔮 Ejecutar Simulación'}
                    </button>

                    {currentPrediction && (
                        <div style={{ marginTop: '2rem', textAlign: 'center', padding: '1.5rem', background: 'rgba(255,255,255,0.02)', borderRadius: '12px' }}>
                            <div style={{
                                fontSize: '3.5rem',
                                fontWeight: 800,
                                color: getRiskColor(currentPrediction.probability),
                                lineHeight: 1
                            }}>
                                {(currentPrediction.probability * 100).toFixed(1)}%
                            </div>
                            <div style={{ color: 'var(--text-secondary)', marginTop: '0.75rem', fontSize: '0.875rem' }}>
                                Probabilidad de brote en {currentPrediction.horizon_days} días
                            </div>
                            <div className={`risk-indicator ${currentPrediction.risk_level}`} style={{ marginTop: '1.25rem' }}>
                                {currentPrediction.risk_level === 'ok' && '✓ Riesgo Bajo'}
                                {currentPrediction.risk_level === 'warning' && '⚠ Riesgo Moderado'}
                                {currentPrediction.risk_level === 'critical' && '🚨 Riesgo Alto'}
                            </div>
                        </div>
                    )}
                </div>

                <div className="card">
                    <div className="card-header">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <Calendar size={24} color="var(--secondary)" />
                            <h3 className="card-title">Tendencia (Últimos 14 días)</h3>
                        </div>
                    </div>

                    <div style={{
                        display: 'flex',
                        alignItems: 'flex-end',
                        gap: '6px',
                        height: '220px',
                        padding: '1.5rem 0',
                        borderBottom: '1px solid rgba(255,255,255,0.1)'
                    }}>
                        {history.slice(0, 14).reverse().map((p: PredictionResult, i: number) => (
                            <div
                                key={i}
                                style={{
                                    flex: 1,
                                    height: `${p.probability * 100}%`,
                                    minHeight: '8px',
                                    background: `linear-gradient(to top, ${getRiskColor(p.probability)}, ${getRiskColor(p.probability)}66)`,
                                    borderRadius: '6px 6px 2px 2px',
                                    transition: 'height 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                                    position: 'relative'
                                }}
                                title={`${new Date(p.generated_at).toLocaleDateString()}: ${(p.probability * 100).toFixed(1)}%`}
                            >
                                <div className="chart-tooltip">
                                    {(p.probability * 100).toFixed(0)}%
                                </div>
                            </div>
                        ))}
                    </div>

                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '0.75rem',
                        color: 'var(--text-muted)',
                        marginTop: '0.75rem'
                    }}>
                        <span>Hace 14 días</span>
                        <span>Hoy</span>
                    </div>

                    <div style={{ marginTop: '1.5rem', display: 'flex', flexWrap: 'wrap', gap: '1rem', fontSize: '0.75rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--success)' }}></div>
                            <span style={{ color: 'var(--text-secondary)' }}>Bajo (&lt;35%)</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--warning)' }}></div>
                            <span style={{ color: 'var(--text-secondary)' }}>Moderado (35-55%)</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--danger)' }}></div>
                            <span style={{ color: 'var(--text-secondary)' }}>Alto (&gt;55%)</span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="card">
                <div className="card-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <DatabaseIcon size={24} color="var(--primary-light)" />
                        <h3 className="card-title">Historial de Predicciones</h3>
                    </div>
                </div>

                <div className="table-container">
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Fecha</th>
                                <th>Probabilidad</th>
                                <th>Horizonte</th>
                                <th>Nivel de Riesgo</th>
                            </tr>
                        </thead>
                        <tbody>
                            {history.slice(0, 10).map((p: PredictionResult, i: number) => (
                                <tr key={i}>
                                    <td>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                            <Calendar size={14} className="text-muted" />
                                            {new Date(p.generated_at).toLocaleDateString('es-ES', {
                                                day: '2-digit',
                                                month: 'short',
                                                year: 'numeric',
                                            })}
                                        </div>
                                    </td>
                                    <td>
                                        <span style={{
                                            color: getRiskColor(p.probability),
                                            fontWeight: 700,
                                            fontSize: '1.1rem'
                                        }}>
                                            {(p.probability * 100).toFixed(1)}%
                                        </span>
                                    </td>
                                    <td style={{ color: 'var(--text-secondary)' }}>{p.horizon_days} días</td>
                                    <td>
                                        <span className={`risk-indicator ${p.risk_level}`} style={{ padding: '0.25rem 0.75rem', fontSize: '0.75rem' }}>
                                            {p.risk_level === 'ok' && 'BAJO'}
                                            {p.risk_level === 'warning' && 'MODERADO'}
                                            {p.risk_level === 'critical' && 'ALTO'}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="card" style={{ marginTop: '1.5rem', borderLeft: '4px solid var(--primary)' }}>
                <div style={{ display: 'flex', gap: '1rem' }}>
                    <Info size={24} color="var(--primary)" style={{ flexShrink: 0 }} />
                    <div>
                        <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-primary)' }}>Interpretación del Análisis</h4>
                        <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                            Este análisis combina múltiples fuentes de datos para detectar desviaciones sutiles en tus patrones habituales.
                            Una probabilidad creciente puede indicar una fase prodrómica. Se recomienda mantener la calma y contactar con
                            tu equipo médico si observas una tendencia ascendente persistente por encima del 55%.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}

const DatabaseIcon = ({ size, color, className }: any) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke={color || "currentColor"}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
    >
        <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
    </svg>
);
