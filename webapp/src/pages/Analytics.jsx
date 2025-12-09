import { useState, useEffect } from 'react';
import { predictAPI, patientAPI } from '../api/client';

export default function Analytics() {
    const [predictions, setPredictions] = useState([]);
    const [currentPrediction, setCurrentPrediction] = useState(null);
    const [loading, setLoading] = useState(true);
    const [horizonDays, setHorizonDays] = useState(14);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            // Intentar obtener predicción actual
            const prediction = await predictAPI.predict(14).catch(() => null);
            setCurrentPrediction(prediction);

            // Simular historial para prototipo
            const mockHistory = generateMockHistory();
            setPredictions(mockHistory);
        } catch (err) {
            console.error('Error loading analytics:', err);
        } finally {
            setLoading(false);
        }
    };

    const runNewPrediction = async () => {
        setLoading(true);
        try {
            const result = await predictAPI.predict(horizonDays);
            setCurrentPrediction(result);

            // Agregar al historial
            setPredictions(prev => [{
                date: new Date().toISOString(),
                probability: result.probability,
                risk_level: result.risk_level,
                horizon_days: result.horizon_days,
            }, ...prev].slice(0, 30));
        } catch (err) {
            console.error('Prediction error:', err);
        } finally {
            setLoading(false);
        }
    };

    // Generar datos mock para el historial
    const generateMockHistory = () => {
        const history = [];
        for (let i = 0; i < 14; i++) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            const probability = 0.25 + Math.random() * 0.3;
            history.push({
                date: date.toISOString(),
                probability,
                risk_level: probability > 0.35 ? 'warning' : 'ok',
                horizon_days: 14,
            });
        }
        return history;
    };

    const getRiskColor = (prob) => {
        if (prob >= 0.55) return '#ef4444';
        if (prob >= 0.35) return '#f59e0b';
        return '#10b981';
    };

    if (loading && predictions.length === 0) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '4rem' }}>
                <div className="spinner" />
            </div>
        );
    }

    return (
        <div className="fade-in">
            <div className="page-header">
                <h1 className="page-title">📈 Analytics</h1>
                <p className="page-subtitle">
                    Análisis de tu riesgo y tendencias
                </p>
            </div>

            {/* Panel principal de predicción */}
            <div className="grid grid-2" style={{ marginBottom: '2rem' }}>
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">🎯 Nueva Predicción</h3>
                    </div>

                    <div style={{ marginBottom: '1.5rem' }}>
                        <label className="form-label">Horizonte de predicción</label>
                        <select
                            className="form-input"
                            value={horizonDays}
                            onChange={(e) => setHorizonDays(parseInt(e.target.value))}
                        >
                            <option value={7}>7 días</option>
                            <option value={14}>14 días</option>
                            <option value={21}>21 días</option>
                            <option value={30}>30 días</option>
                        </select>
                    </div>

                    <button
                        className="btn btn-primary"
                        style={{ width: '100%' }}
                        onClick={runNewPrediction}
                        disabled={loading}
                    >
                        {loading ? <span className="spinner" /> : '🔮 Calcular Predicción'}
                    </button>

                    {currentPrediction && (
                        <div style={{ marginTop: '1.5rem', textAlign: 'center' }}>
                            <div style={{
                                fontSize: '3rem',
                                fontWeight: 700,
                                color: getRiskColor(currentPrediction.probability),
                            }}>
                                {(currentPrediction.probability * 100).toFixed(1)}%
                            </div>
                            <div style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                                Probabilidad de brote en {currentPrediction.horizon_days} días
                            </div>
                            <div className={`risk-indicator ${currentPrediction.risk_level}`} style={{ marginTop: '1rem' }}>
                                {currentPrediction.risk_level === 'ok' && '✓ Riesgo Bajo'}
                                {currentPrediction.risk_level === 'warning' && '⚠ Riesgo Moderado'}
                                {currentPrediction.risk_level === 'critical' && '🚨 Riesgo Alto'}
                            </div>
                        </div>
                    )}
                </div>

                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">📊 Tendencia (14 días)</h3>
                    </div>

                    {/* Gráfico simplificado con barras */}
                    <div style={{
                        display: 'flex',
                        alignItems: 'flex-end',
                        gap: '4px',
                        height: '200px',
                        padding: '1rem 0'
                    }}>
                        {predictions.slice(0, 14).reverse().map((p, i) => (
                            <div
                                key={i}
                                style={{
                                    flex: 1,
                                    height: `${p.probability * 100}%`,
                                    minHeight: '4px',
                                    background: `linear-gradient(to top, ${getRiskColor(p.probability)}, ${getRiskColor(p.probability)}88)`,
                                    borderRadius: '4px 4px 0 0',
                                    transition: 'height 0.3s ease',
                                }}
                                title={`${new Date(p.date).toLocaleDateString()}: ${(p.probability * 100).toFixed(1)}%`}
                            />
                        ))}
                    </div>

                    {/* Leyenda */}
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '0.75rem',
                        color: 'var(--text-muted)',
                        marginTop: '0.5rem'
                    }}>
                        <span>Hace 14 días</span>
                        <span>Hoy</span>
                    </div>

                    {/* Líneas de umbral */}
                    <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem', fontSize: '0.75rem' }}>
                        <span style={{ color: '#10b981' }}>● Bajo (&lt;35%)</span>
                        <span style={{ color: '#f59e0b' }}>● Moderado (35-55%)</span>
                        <span style={{ color: '#ef4444' }}>● Alto (&gt;55%)</span>
                    </div>
                </div>
            </div>

            {/* Estadísticas */}
            <div className="grid grid-4" style={{ marginBottom: '2rem' }}>
                <div className="card metric-card">
                    <div className="metric-value" style={{ color: 'var(--primary-light)' }}>
                        {predictions.length}
                    </div>
                    <div className="metric-label">Predicciones totales</div>
                </div>

                <div className="card metric-card">
                    <div className="metric-value success">
                        {(predictions.reduce((acc, p) => acc + p.probability, 0) / predictions.length * 100).toFixed(0)}%
                    </div>
                    <div className="metric-label">Promedio de riesgo</div>
                </div>

                <div className="card metric-card">
                    <div className="metric-value" style={{ color: 'var(--warning)' }}>
                        {predictions.filter(p => p.risk_level === 'warning').length}
                    </div>
                    <div className="metric-label">Alertas moderadas</div>
                </div>

                <div className="card metric-card">
                    <div className="metric-value danger">
                        {predictions.filter(p => p.probability >= 0.55).length}
                    </div>
                    <div className="metric-label">Alertas críticas</div>
                </div>
            </div>

            {/* Historial de predicciones */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">📜 Historial de Predicciones</h3>
                </div>

                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Fecha</th>
                            <th>Probabilidad</th>
                            <th>Horizonte</th>
                            <th>Nivel</th>
                        </tr>
                    </thead>
                    <tbody>
                        {predictions.slice(0, 10).map((p, i) => (
                            <tr key={i}>
                                <td>
                                    {new Date(p.date).toLocaleDateString('es-ES', {
                                        day: '2-digit',
                                        month: 'short',
                                        year: 'numeric',
                                    })}
                                </td>
                                <td>
                                    <span style={{
                                        color: getRiskColor(p.probability),
                                        fontWeight: 600
                                    }}>
                                        {(p.probability * 100).toFixed(1)}%
                                    </span>
                                </td>
                                <td>{p.horizon_days} días</td>
                                <td>
                                    <span className={`risk-indicator ${p.risk_level}`}>
                                        {p.risk_level === 'ok' && '✓ Bajo'}
                                        {p.risk_level === 'warning' && '⚠ Moderado'}
                                        {p.risk_level === 'critical' && '🚨 Alto'}
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Recomendaciones */}
            <div className="card" style={{ marginTop: '1.5rem' }}>
                <div className="card-header">
                    <h3 className="card-title">💡 Interpretación</h3>
                </div>
                <div style={{
                    padding: '1rem',
                    background: 'var(--bg-dark)',
                    borderRadius: '8px',
                    lineHeight: 1.8
                }}>
                    <p>
                        <strong>Tu modelo de predicción</strong> analiza patrones en tus datos de salud
                        para estimar la probabilidad de un brote en los próximos días.
                    </p>
                    <p style={{ marginTop: '1rem' }}>
                        📊 <strong>Factores considerados:</strong> actividad física, calidad del sueño,
                        patrones de comunicación, y datos biométricos de tus wearables.
                    </p>
                    <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>
                        ⚕️ <em>Este sistema es una herramienta de apoyo. Siempre consulta
                            con tu neurólogo para decisiones médicas.</em>
                    </p>
                </div>
            </div>
        </div>
    );
}
