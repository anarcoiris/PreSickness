import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
    Save,
    RefreshCw,
    AlertCircle,
    Check,
} from 'lucide-react';

import { eventsApi, patientApi } from '../api/client';
import type { LabelSettings } from '../api/client';
import { Users, Trash2 } from 'lucide-react';

export default function SettingsPage() {
    const queryClient = useQueryClient();
    const [hasChanges, setHasChanges] = useState(false);

    const { data: settings, isLoading: isLoadingSettings, isError: isErrorSettings } = useQuery({
        queryKey: ['label-settings'],
        queryFn: () => eventsApi.getSettings().then(r => r.data),
    });

    const { data: retrainingStatus, isLoading: isLoadingRetraining } = useQuery({
        queryKey: ['retraining-status'],
        queryFn: () => eventsApi.getRetrainingStatus().then(r => r.data),
    });

    const { data: doctors, refetch: refetchDoctors } = useQuery({
        queryKey: ['my-doctors'],
        queryFn: async () => {
            try { return (await patientApi.listDoctors()).data; } catch { return []; }
        },
    });

    const handleRevokeDoctor = async (id: string) => {
        if (!confirm('¿Revocar acceso a este médico?')) return;
        try {
            await patientApi.revokeDoctor(id);
            refetchDoctors();
        } catch (e) { alert('Error al revocar acceso'); }
    };

    const [formData, setFormData] = useState<Partial<LabelSettings>>({});

    // Correcto: Usar useEffect para sincronizar props/data con estado interno
    useEffect(() => {
        if (settings) {
            setFormData(settings);
        }
    }, [settings]);

    const updateMutation = useMutation({
        mutationFn: (data: Partial<LabelSettings>) => eventsApi.updateSettings(data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['label-settings'] });
            setHasChanges(false);
        },
    });

    const retrainMutation = useMutation({
        mutationFn: () => eventsApi.triggerRetraining(),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['retraining-status'] });
            queryClient.invalidateQueries({ queryKey: ['events-stats'] });
        },
    });

    const handleChange = (field: keyof LabelSettings, value: any) => {
        setFormData(prev => ({ ...prev, [field]: value }));
        setHasChanges(true);
    };

    const handleSave = () => {
        updateMutation.mutate(formData);
    };

    if (isLoadingSettings || isLoadingRetraining) {
        return (
            <div className="loading">
                <div className="spinner" />
            </div>
        );
    }

    if (isErrorSettings) {
        return (
            <div className="error-state">
                <AlertCircle size={48} color="var(--color-error)" />
                <h2>Error al cargar configuración</h2>
                <p>No se pudo conectar con el servidor.</p>
                <button
                    className="btn btn-primary"
                    onClick={() => queryClient.invalidateQueries({ queryKey: ['label-settings'] })}
                >
                    Reintentar
                </button>
            </div>
        );
    }

    const currentSettings = { ...settings, ...formData };

    return (
        <div>
            <div className="page-header">
                <div>
                    <h1 className="page-title">Configuración de Labels</h1>
                    <p className="page-subtitle">
                        Personaliza cómo se generan los labels para el modelo de predicción
                    </p>
                </div>
                {hasChanges && (
                    <button className="btn btn-primary" onClick={handleSave} disabled={updateMutation.isPending}>
                        <Save size={18} />
                        {updateMutation.isPending ? 'Guardando...' : 'Guardar Cambios'}
                    </button>
                )}
            </div>

            {/* Retraining Status */}
            {retrainingStatus && retrainingStatus.requires_retraining && (
                <div className="alert-banner warning" style={{ marginBottom: 'var(--space-xl)' }}>
                    <AlertCircle size={20} />
                    <div style={{ flex: 1 }}>
                        <strong>Cambios pendientes</strong>
                        <p style={{ margin: 0, fontSize: '0.875rem', opacity: 0.8 }}>
                            Hay {retrainingStatus.pending_changes} cambio(s) que requieren regenerar los labels.
                        </p>
                    </div>
                    <button
                        className="btn btn-warning"
                        onClick={() => retrainMutation.mutate()}
                        disabled={retrainMutation.isPending}
                    >
                        <RefreshCw size={16} />
                        {retrainMutation.isPending ? 'Procesando...' : 'Reentrenar Ahora'}
                    </button>
                </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-xl)' }}>
                {/* Horizons */}
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">Horizontes de Predicción</h2>
                    </div>
                    <p style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-lg)' }}>
                        Define los horizontes temporales para los que se generarán labels.
                    </p>

                    <div className="form-group">
                        <label className="form-label">Horizontes activos (días)</label>
                        <div style={{ display: 'flex', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
                            {[7, 14, 21, 30].map(horizon => (
                                <label key={horizon} className="form-radio">
                                    <input
                                        type="checkbox"
                                        checked={currentSettings.horizons?.includes(horizon)}
                                        onChange={e => {
                                            const newHorizons = e.target.checked
                                                ? [...(currentSettings.horizons || []), horizon].sort((a, b) => a - b)
                                                : (currentSettings.horizons || []).filter(h => h !== horizon);
                                            handleChange('horizons', newHorizons);
                                        }}
                                    />
                                    <span>{horizon} días</span>
                                </label>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Event Types */}
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">Tipos de Eventos como Labels</h2>
                    </div>
                    <p style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-lg)' }}>
                        Qué tipos de eventos cuentan como "positivo" (label=1) para el modelo.
                    </p>

                    <div className="form-group">
                        {[
                            { value: 'confirmed_relapse', label: 'Brote Confirmado' },
                            { value: 'medication_start', label: 'Inicio de Medicación' },
                            { value: 'symptom_onset', label: 'Inicio de Síntomas' },
                            { value: 'hospital_visit', label: 'Visita al Hospital' },
                        ].map(type => (
                            <label key={type.value} className="form-radio" style={{ marginBottom: 'var(--space-sm)' }}>
                                <input
                                    type="checkbox"
                                    checked={currentSettings.label_event_types?.includes(type.value)}
                                    onChange={e => {
                                        const newTypes = e.target.checked
                                            ? [...(currentSettings.label_event_types || []), type.value]
                                            : (currentSettings.label_event_types || []).filter(t => t !== type.value);
                                        handleChange('label_event_types', newTypes);
                                    }}
                                />
                                <span>{type.label}</span>
                            </label>
                        ))}
                    </div>
                </div>

                {/* Censoring */}
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">Censura Temporal</h2>
                    </div>
                    <p style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-lg)' }}>
                        Días a ignorar al final del dataset (labels incompletos por falta de datos futuros).
                    </p>

                    <div className="form-group">
                        <label className="form-label">Días de censura</label>
                        <input
                            type="number"
                            className="form-input"
                            value={currentSettings.censor_days_before_end || 30}
                            onChange={e => handleChange('censor_days_before_end', parseInt(e.target.value) || 0)}
                            min={0}
                            max={90}
                        />
                        <small style={{ color: 'var(--color-text-muted)', marginTop: 'var(--space-sm)', display: 'block' }}>
                            Recomendado: igual al horizonte máximo (ej: 30 días)
                        </small>
                    </div>
                </div>

                {/* Auto Clusters */}
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">Clusters Automáticos</h2>
                    </div>
                    <p style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-lg)' }}>
                        Configuración para usar clusters auto-detectados como labels adicionales.
                    </p>

                    <div className="form-group">
                        <label className="form-radio">
                            <input
                                type="checkbox"
                                checked={currentSettings.use_auto_clusters || false}
                                onChange={e => handleChange('use_auto_clusters', e.target.checked)}
                            />
                            <span>Usar clusters auto-detectados como labels</span>
                        </label>
                    </div>

                    {currentSettings.use_auto_clusters && (
                        <div className="form-group">
                            <label className="form-label">Confianza mínima</label>
                            <input
                                type="range"
                                min={0.5}
                                max={1}
                                step={0.05}
                                value={currentSettings.auto_cluster_min_confidence || 0.8}
                                onChange={e => handleChange('auto_cluster_min_confidence', parseFloat(e.target.value))}
                                style={{ width: '100%' }}
                            />
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
                                <span>0.5 (más sensible)</span>
                                <span><strong>{((currentSettings.auto_cluster_min_confidence || 0.8) * 100).toFixed(0)}%</strong></span>
                                <span>1.0 (más estricto)</span>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Last Training Info */}
            {retrainingStatus && (
                <div className="card" style={{ marginTop: 'var(--space-xl)' }}>
                    <div className="card-header">
                        <h2 className="card-title">Estado del Modelo</h2>
                    </div>
                    <div style={{ display: 'flex', gap: 'var(--space-xl)' }}>
                        <div>
                            <div style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
                                Último entrenamiento
                            </div>
                            <div style={{ fontWeight: 600 }}>
                                {retrainingStatus.last_trained_at
                                    ? new Date(retrainingStatus.last_trained_at).toLocaleString('es-ES')
                                    : 'Nunca'}
                            </div>
                        </div>
                        <div>
                            <div style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
                                Cambios pendientes
                            </div>
                            <div style={{ fontWeight: 600, color: retrainingStatus.pending_changes > 0 ? 'var(--color-warning)' : 'var(--color-success)' }}>
                                {retrainingStatus.pending_changes}
                                {retrainingStatus.pending_changes === 0 && (
                                    <Check size={16} style={{ marginLeft: '0.5rem', verticalAlign: 'middle' }} />
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* My Doctors (for patients) */}
            <div className="card" style={{ marginTop: 'var(--space-xl)' }}>
                <div className="card-header" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Users size={20} />
                    <h2 className="card-title" style={{ margin: 0 }}>Mis Médicos</h2>
                </div>
                <div className="card-body">
                    <p style={{ color: 'var(--color-text-secondary)', marginBottom: '1rem' }}>
                        Profesionales sanitarios con acceso a tus datos de predicción.
                    </p>

                    {!doctors || doctors.length === 0 ? (
                        <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', textAlign: 'center', color: 'var(--color-text-muted)' }}>
                            No tienes médicos vinculados actualmente.
                        </div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            {doctors.map(doc => (
                                <div key={doc.doctor_id} style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    padding: '1rem',
                                    background: 'rgba(255,255,255,0.03)',
                                    borderRadius: '8px',
                                    border: '1px solid rgba(255,255,255,0.05)'
                                }}>
                                    <div>
                                        <div style={{ fontWeight: 'bold' }}>{doc.doctor_name}</div>
                                        <div style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
                                            {doc.doctor_email} • Desde {new Date(doc.granted_at).toLocaleDateString()}
                                        </div>
                                    </div>
                                    <button
                                        className="btn btn-sm btn-danger-outline"
                                        style={{
                                            background: 'transparent',
                                            border: '1px solid var(--color-danger)',
                                            color: 'var(--color-danger)',
                                            padding: '0.25rem 0.75rem',
                                            borderRadius: '4px',
                                            display: 'flex',
                                            gap: '0.5rem',
                                            alignItems: 'center',
                                            cursor: 'pointer'
                                        }}
                                        onClick={() => handleRevokeDoctor(doc.doctor_id)}
                                    >
                                        <Trash2 size={14} /> Revocar
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
