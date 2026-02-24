import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { analysisApi } from '../api/client';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from 'recharts';
import {
    Activity,
    Cpu,
    GitBranch,
    Sliders,
    TrendingUp,
    AlertCircle
} from 'lucide-react';

export default function ModelAnalysisPage() {
    const [activeTab, setActiveTab] = useState<'overview' | 'performance' | 'features' | 'optimization'>('overview');

    const { data: training, isLoading: loadingTraining } = useQuery({
        queryKey: ['analysis-training'],
        queryFn: () => analysisApi.getTrainingResults().then(res => res.data),
    });

    const { data: ensemble, isLoading: loadingEnsemble } = useQuery({
        queryKey: ['analysis-ensemble'],
        queryFn: () => analysisApi.getEnsembleResults().then(res => res.data),
    });

    const { data: optuna, isLoading: loadingOptuna } = useQuery({
        queryKey: ['analysis-optuna'],
        queryFn: () => analysisApi.getOptunaResults().then(res => res.data),
    });

    console.log('Analysis Data State:', {
        training: !!training,
        ensemble: !!ensemble,
        optuna: !!optuna,
        loading: loadingTraining || loadingEnsemble || loadingOptuna
    });

    if (loadingTraining || loadingEnsemble || loadingOptuna) {
        return (
            <div className="loading-container fade-in">
                <div className="spinner"></div>
                <p>Cargando análisis profundo del modelo...</p>
            </div>
        );
    }

    if (!training || !ensemble || !optuna) {
        return (
            <div className="error-container fade-in">
                <AlertCircle size={48} className="text-danger" />
                <h2>No hay datos de análisis disponibles</h2>
                <p>Asegúrate de que el modelo haya sido entrenado recientemente.</p>
            </div>
        );
    }

    // Process data for charts
    const modelPerformanceData = training.models ? Object.entries(training.models).map(([name, metrics]: [string, any]) => ({
        name,
        AUROC: metrics.auroc,
        AUPRC: metrics.auprc,
    })) : [];

    const featureImportanceData = training.feature_importance ? training.feature_importance.slice(0, 15) : [];

    return (
        <div className="fade-in">
            <div className="page-header">
                <h1 className="page-title">Análisis del Modelo</h1>
                <p className="page-subtitle">Inspección profunda de métricas y comportamiento del modelo ML</p>
            </div>

            {/* Tabs */}
            <div className="tabs">
                <button
                    className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
                    onClick={() => setActiveTab('overview')}
                >
                    <TrendingUp size={16} />
                    Resumen
                </button>
                <button
                    className={`tab ${activeTab === 'performance' ? 'active' : ''}`}
                    onClick={() => setActiveTab('performance')}
                >
                    <Activity size={16} />
                    Rendimiento
                </button>
                <button
                    className={`tab ${activeTab === 'features' ? 'active' : ''}`}
                    onClick={() => setActiveTab('features')}
                >
                    <GitBranch size={16} />
                    Features
                </button>
                <button
                    className={`tab ${activeTab === 'optimization' ? 'active' : ''}`}
                    onClick={() => setActiveTab('optimization')}
                >
                    <Sliders size={16} />
                    Optimización
                </button>
            </div>

            {/* Content */}
            <div className="tab-content" style={{ marginTop: '1.5rem' }}>

                {/* OVERVIEW TAB */}
                {activeTab === 'overview' && (
                    <div className="grid grid-2 fade-in">
                        <div className="card">
                            <div className="card-header">
                                <h3 className="card-title">Mejor Modelo</h3>
                            </div>
                            <div style={{ textAlign: 'center', padding: '2rem' }}>
                                <Cpu size={48} className="text-primary" style={{ marginBottom: '1rem' }} />
                                <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--primary)' }}>
                                    {training.best_model}
                                </div>
                                <div className="badge primary" style={{ marginTop: '0.5rem' }}>
                                    En Producción
                                </div>
                            </div>
                        </div>

                        <div className="card">
                            <div className="card-header">
                                <h3 className="card-title">Métricas Clave</h3>
                            </div>
                            <div className="stats-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
                                <div className="stat-card">
                                    <div className="stat-value">{(training.models[training.best_model]?.auroc * 100).toFixed(1)}%</div>
                                    <div className="stat-label">AUROC</div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-value">{(training.models[training.best_model]?.auprc * 100).toFixed(1)}%</div>
                                    <div className="stat-label">AUPRC</div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-value">{training.samples}</div>
                                    <div className="stat-label">Muestras</div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-value">{training.features}</div>
                                    <div className="stat-label">Features</div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* PERFORMANCE TAB */}
                {activeTab === 'performance' && (
                    <div className="card fade-in">
                        <div className="card-header">
                            <h3 className="card-title">Comparativa de Modelos</h3>
                        </div>
                        <div style={{ height: '400px', width: '100%' }}>
                            <ResponsiveContainer>
                                <BarChart data={modelPerformanceData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                    <XAxis dataKey="name" stroke="var(--text-secondary)" />
                                    <YAxis stroke="var(--text-secondary)" domain={[0, 1]} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'var(--border-color)' }}
                                        itemStyle={{ color: 'var(--text-primary)' }}
                                    />
                                    <Legend />
                                    <Bar dataKey="AUROC" fill="var(--primary)" radius={[4, 4, 0, 0]} />
                                    <Bar dataKey="AUPRC" fill="var(--secondary)" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="card-body">
                            <h4>Detalle del Ensemble</h4>
                            <table className="table" style={{ marginTop: '1rem' }}>
                                <thead>
                                    <tr>
                                        <th>Método</th>
                                        <th>AUROC</th>
                                        <th>AUPRC</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {Object.entries(ensemble.results).map(([method, metrics]: [string, any]) => (
                                        <tr key={method}>
                                            <td style={{ textTransform: 'capitalize' }}>{method.replace('_', ' ')}</td>
                                            <td>{(metrics.auroc * 100).toFixed(2)}%</td>
                                            <td>{(metrics.auprc * 100).toFixed(2)}%</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* FEATURES TAB */}
                {activeTab === 'features' && (
                    <div className="card fade-in">
                        <div className="card-header">
                            <h3 className="card-title">Importancia de Features</h3>
                        </div>
                        <div style={{ height: '500px', width: '100%' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    data={featureImportanceData}
                                    layout="vertical"
                                    margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" horizontal={true} vertical={false} />
                                    <XAxis type="number" stroke="var(--text-secondary)" />
                                    <YAxis
                                        dataKey="feature"
                                        type="category"
                                        width={150}
                                        stroke="var(--text-secondary)"
                                        tick={{ fontSize: 12 }}
                                    />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'var(--border-color)' }}
                                        cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                    />
                                    <Bar dataKey="importance" fill="var(--success)" radius={[0, 4, 4, 0]} barSize={20} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="card-footer">
                            <p className="text-muted text-sm">
                                * Mostrando las top 15 features más influyentes en la predicción del modelo Random Forest.
                            </p>
                        </div>
                    </div>
                )}

                {/* OPTIMIZATION TAB */}
                {activeTab === 'optimization' && (
                    <div className="grid grid-2 fade-in">
                        {Object.entries(optuna).map(([model, data]: [string, any]) => (
                            <div className="card" key={model}>
                                <div className="card-header">
                                    <h3 className="card-title" style={{ textTransform: 'uppercase' }}>{model}</h3>
                                    <span className="badge success">
                                        AUROC: {(data.holdout_auroc * 100).toFixed(2)}%
                                    </span>
                                </div>
                                <div className="card-body">
                                    <h4>Mejores Hiperparámetros (Optuna)</h4>
                                    <ul className="info-list" style={{ marginTop: '1rem' }}>
                                        {Object.entries(data.params).map(([param, value]: [string, any]) => (
                                            <li key={param} style={{ justifyContent: 'space-between' }}>
                                                <span className="text-muted">{param}</span>
                                                <code className="code-snippet">
                                                    {typeof value === 'number' && !Number.isInteger(value)
                                                        ? value.toFixed(4)
                                                        : value}
                                                </code>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
