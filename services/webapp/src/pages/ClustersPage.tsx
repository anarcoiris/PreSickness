import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';
import {
    AlertTriangle,
    Check,
    X,
    TrendingUp,
} from 'lucide-react';

import { eventsApi } from '../api/client';
import type { Cluster, Event } from '../api/client';

interface ValidateModalProps {
    cluster: Cluster;
    onClose: () => void;
    onValidate: (data: Partial<Event>) => void;
    isLoading?: boolean;
}

function ValidateModal({ cluster, onClose, onValidate, isLoading }: ValidateModalProps) {
    const [formData, setFormData] = useState({
        event_date: cluster.peak_date,
        event_type: 'confirmed_relapse',
        severity: cluster.max_severity || 'moderate',
        notes: `Validado desde cluster detectado automáticamente (score: ${cluster.severity_score.toFixed(1)})`,
        medication_start_date: '',
    });

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onValidate({
            event_date: new Date(formData.event_date).toISOString(),
            event_type: formData.event_type as Event['event_type'],
            severity: formData.severity as Event['severity'] || undefined,
            notes: formData.notes,
            medication_start_date: formData.medication_start_date
                ? new Date(formData.medication_start_date).toISOString()
                : undefined,
        });
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h2 className="modal-title">Validar Cluster</h2>
                    <button className="modal-close" onClick={onClose}>
                        <X size={24} />
                    </button>
                </div>

                <div className="card" style={{ background: 'rgba(237, 137, 54, 0.1)', marginBottom: '1rem' }}>
                    <h4 style={{ marginBottom: '0.5rem' }}>Información del Cluster</h4>
                    <p style={{ margin: 0, color: 'var(--color-text-secondary)' }}>
                        Período: {format(new Date(cluster.start_date), 'dd/MM/yyyy')} - {format(new Date(cluster.end_date), 'dd/MM/yyyy')}
                    </p>
                    <p style={{ margin: 0, color: 'var(--color-text-secondary)' }}>
                        Señales detectadas: {cluster.total_signals} | Score: {cluster.severity_score.toFixed(1)}
                    </p>
                    {cluster.is_probable_relapse && (
                        <span className="badge pending" style={{ marginTop: '0.5rem' }}>
                            Probable Brote
                        </span>
                    )}
                </div>

                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label className="form-label">Fecha del evento *</label>
                        <input
                            type="date"
                            className="form-input"
                            value={formData.event_date}
                            onChange={e => setFormData({ ...formData, event_date: e.target.value })}
                            required
                        />
                        <small style={{ color: 'var(--color-text-muted)' }}>
                            Fecha pico sugerida: {format(new Date(cluster.peak_date), 'dd/MM/yyyy')}
                        </small>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Tipo de evento</label>
                        <select
                            className="form-select"
                            value={formData.event_type}
                            onChange={e => setFormData({ ...formData, event_type: e.target.value })}
                        >
                            <option value="confirmed_relapse">Brote Confirmado</option>
                            <option value="symptom_onset">Inicio de Síntomas</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Severidad</label>
                        <div className="form-radio-group">
                            {['mild', 'moderate', 'severe'].map(sev => (
                                <label key={sev} className="form-radio">
                                    <input
                                        type="radio"
                                        name="severity"
                                        value={sev}
                                        checked={formData.severity === sev}
                                        onChange={e => setFormData({ ...formData, severity: e.target.value })}
                                    />
                                    <span>{sev === 'mild' ? 'Leve' : sev === 'moderate' ? 'Moderado' : 'Severo'}</span>
                                </label>
                            ))}
                        </div>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Fecha de medicación (si aplica)</label>
                        <input
                            type="date"
                            className="form-input"
                            value={formData.medication_start_date}
                            onChange={e => setFormData({ ...formData, medication_start_date: e.target.value })}
                        />
                    </div>

                    <div className="form-group">
                        <label className="form-label">Notas</label>
                        <textarea
                            className="form-textarea"
                            value={formData.notes}
                            onChange={e => setFormData({ ...formData, notes: e.target.value })}
                        />
                    </div>

                    <div className="modal-footer">
                        <button type="button" className="btn btn-secondary" onClick={onClose}>
                            Cancelar
                        </button>
                        <button type="submit" className="btn btn-success" disabled={isLoading}>
                            {isLoading ? <div className="spinner-small" style={{ marginRight: '0.5rem' }} /> : <Check size={16} style={{ marginRight: '0.5rem' }} />}
                            {isLoading ? 'Validando...' : 'Confirmar Validación'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}

interface RejectModalProps {
    cluster: Cluster;
    onClose: () => void;
    onReject: (reason: string) => void;
    isLoading?: boolean;
}

function RejectModal({ onClose, onReject, isLoading }: RejectModalProps) {
    const [reason, setReason] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onReject(reason);
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h2 className="modal-title">Rechazar Cluster</h2>
                    <button className="modal-close" onClick={onClose}>
                        <X size={24} />
                    </button>
                </div>

                <p style={{ marginBottom: '1rem', color: 'var(--color-text-secondary)' }}>
                    ¿Estás seguro de que este cluster NO corresponde a un brote real?
                </p>

                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label className="form-label">Motivo del rechazo *</label>
                        <textarea
                            className="form-textarea"
                            value={reason}
                            onChange={e => setReason(e.target.value)}
                            placeholder="Ej: Falso positivo, coincide con vacaciones, actividad normal..."
                            required
                            minLength={5}
                        />
                    </div>

                    <div className="modal-footer">
                        <button type="button" className="btn btn-secondary" onClick={onClose}>
                            Cancelar
                        </button>
                        <button type="submit" className="btn btn-danger" disabled={isLoading || reason.length < 5}>
                            <X size={16} />
                            {isLoading ? 'Rechazando...' : 'Confirmar Rechazo'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}

export default function ClustersPage() {
    const queryClient = useQueryClient();
    const [validateCluster, setValidateCluster] = useState<Cluster | null>(null);
    const [rejectCluster, setRejectCluster] = useState<Cluster | null>(null);

    const { data: clusters = [], isLoading } = useQuery({
        queryKey: ['clusters'],
        queryFn: () => eventsApi.listClusters().then(r => r.data),
    });

    const validateMutation = useMutation({
        mutationFn: ({ id, data }: { id: string; data: Partial<Event> }) =>
            eventsApi.validateCluster(id, data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['clusters'] });
            queryClient.invalidateQueries({ queryKey: ['events'] });
            queryClient.invalidateQueries({ queryKey: ['retraining-status'] });
            setValidateCluster(null);
        },
        onError: (error: any) => {
            const message = error.response?.data?.detail || 'Error al validar el cluster';
            alert(`Error: ${message}`);
        },
    });

    const rejectMutation = useMutation({
        mutationFn: ({ id, reason }: { id: string; reason: string }) =>
            eventsApi.rejectCluster(id, reason),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['clusters'] });
            setRejectCluster(null);
        },
        onError: (error: any) => {
            const message = error.response?.data?.detail || 'Error al rechazar el cluster';
            alert(`Error: ${message}`);
        },
    });

    const pendingClusters = clusters.filter(c => c.status === 'pending');
    const processedClusters = clusters.filter(c => c.status !== 'pending');

    return (
        <div>
            <div className="page-header">
                <div>
                    <h1 className="page-title">Clusters Auto-Detectados</h1>
                    <p className="page-subtitle">
                        Revisa y valida los patrones detectados automáticamente por el sistema
                    </p>
                </div>
            </div>

            {/* Pending Clusters */}
            <div className="card" style={{ marginBottom: 'var(--space-xl)' }}>
                <div className="card-header">
                    <h2 className="card-title">
                        <AlertTriangle size={20} style={{ marginRight: '0.5rem', color: 'var(--color-warning)' }} />
                        Pendientes de Revisión ({pendingClusters.length})
                    </h2>
                </div>

                {isLoading ? (
                    <div className="loading">
                        <div className="spinner" />
                    </div>
                ) : pendingClusters.length === 0 ? (
                    <div className="empty-state">
                        <Check size={64} className="empty-state-icon" style={{ color: 'var(--color-success)' }} />
                        <h3>No hay clusters pendientes</h3>
                        <p>Todos los clusters detectados han sido procesados.</p>
                    </div>
                ) : (
                    <div className="event-list">
                        {pendingClusters.map(cluster => (
                            <div key={cluster.id} className="event-item">
                                <div
                                    className="event-icon"
                                    style={{
                                        background: cluster.is_probable_relapse
                                            ? 'rgba(245, 101, 101, 0.2)'
                                            : 'rgba(237, 137, 54, 0.2)',
                                        color: cluster.is_probable_relapse
                                            ? 'var(--color-danger)'
                                            : 'var(--color-warning)',
                                    }}
                                >
                                    <TrendingUp />
                                </div>

                                <div className="event-content">
                                    <div className="event-header">
                                        <h3 className="event-title">
                                            Cluster {format(new Date(cluster.start_date), 'dd/MM')} - {format(new Date(cluster.end_date), 'dd/MM/yyyy')}
                                        </h3>
                                        <span className="event-date">
                                            Pico: {format(new Date(cluster.peak_date), "d 'de' MMMM yyyy", { locale: es })}
                                        </span>
                                    </div>

                                    <div className="event-meta" style={{ marginTop: '0.5rem' }}>
                                        <span style={{ color: 'var(--color-text-secondary)' }}>
                                            📊 {cluster.total_signals} señales | {cluster.unique_types} tipos
                                        </span>
                                        <span style={{ color: 'var(--color-text-secondary)' }}>
                                            Score: {cluster.severity_score.toFixed(1)}
                                        </span>
                                        {cluster.is_probable_relapse && (
                                            <span className="badge pending">Probable Brote</span>
                                        )}
                                        {cluster.max_severity && (
                                            <span className={`badge ${cluster.max_severity}`}>
                                                {cluster.max_severity}
                                            </span>
                                        )}
                                    </div>
                                </div>

                                <div className="event-actions">
                                    <button
                                        className="btn btn-success"
                                        onClick={() => setValidateCluster(cluster)}
                                    >
                                        <Check size={16} />
                                        Validar
                                    </button>
                                    <button
                                        className="btn btn-secondary"
                                        onClick={() => setRejectCluster(cluster)}
                                    >
                                        <X size={16} />
                                        Rechazar
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Processed Clusters */}
            {processedClusters.length > 0 && (
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">Historial de Clusters Procesados</h2>
                    </div>

                    <div className="event-list">
                        {processedClusters.map(cluster => (
                            <div key={cluster.id} className="event-item" style={{ opacity: 0.7 }}>
                                <div
                                    className="event-icon"
                                    style={{
                                        background: cluster.status === 'validated'
                                            ? 'rgba(72, 187, 120, 0.2)'
                                            : 'rgba(245, 101, 101, 0.2)',
                                        color: cluster.status === 'validated'
                                            ? 'var(--color-success)'
                                            : 'var(--color-danger)',
                                    }}
                                >
                                    {cluster.status === 'validated' ? <Check /> : <X />}
                                </div>

                                <div className="event-content">
                                    <div className="event-header">
                                        <h3 className="event-title">
                                            {format(new Date(cluster.start_date), 'dd/MM')} - {format(new Date(cluster.end_date), 'dd/MM/yyyy')}
                                        </h3>
                                        <span className={`badge ${cluster.status}`}>
                                            {cluster.status === 'validated' ? 'Validado' : 'Rechazado'}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Modals */}
            {validateCluster && (
                <ValidateModal
                    cluster={validateCluster}
                    onClose={() => setValidateCluster(null)}
                    onValidate={(data) => validateMutation.mutate({ id: validateCluster.id, data })}
                    isLoading={validateMutation.isPending}
                />
            )}

            {rejectCluster && (
                <RejectModal
                    cluster={rejectCluster}
                    onClose={() => setRejectCluster(null)}
                    onReject={(reason) => rejectMutation.mutate({ id: rejectCluster.id, reason })}
                    isLoading={rejectMutation.isPending}
                />
            )}
        </div>
    );
}
