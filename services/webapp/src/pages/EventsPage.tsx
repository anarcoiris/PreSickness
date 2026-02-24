import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';
import {
    Plus,
    AlertCircle,
    Heart,
    Pill,
    Building2,
    Calendar,
    Edit,
    Trash2,
    RefreshCw,
    Upload,
    X,
} from 'lucide-react';

import { eventsApi } from '../api/client';
import type { Event, RetrainingStatus } from '../api/client';

const EVENT_TYPE_LABELS: Record<string, string> = {
    symptom_onset: 'Inicio de Síntomas',
    confirmed_relapse: 'Brote Confirmado',
    medication_start: 'Inicio de Medicación',
    hospital_visit: 'Visita al Hospital',
    doctor_appointment: 'Cita Médica',
};

const EVENT_TYPE_ICONS: Record<string, React.ReactNode> = {
    symptom_onset: <AlertCircle />,
    confirmed_relapse: <Heart />,
    medication_start: <Pill />,
    hospital_visit: <Building2 />,
    doctor_appointment: <Calendar />,
};

function EventIcon({ type }: { type: string }) {
    return (
        <div className={`event-icon ${type.replace('_', '-')}`}>
            {EVENT_TYPE_ICONS[type] || <AlertCircle />}
        </div>
    );
}

function RetrainingBanner({ status }: { status: RetrainingStatus }) {
    const queryClient = useQueryClient();

    const mutation = useMutation({
        mutationFn: () => eventsApi.triggerRetraining(),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['retraining-status'] });
            queryClient.invalidateQueries({ queryKey: ['events-stats'] });
        },
    });

    if (!status.requires_retraining) return null;

    return (
        <div className="alert-banner warning">
            <AlertCircle size={20} />
            <div style={{ flex: 1 }}>
                <strong>Reentrenamiento necesario</strong>
                <p style={{ margin: 0, fontSize: '0.875rem', opacity: 0.8 }}>
                    Hay {status.pending_changes} cambio(s) en eventos que requieren regenerar los labels del modelo.
                </p>
            </div>
            <button
                className="btn btn-warning"
                onClick={() => mutation.mutate()}
                disabled={mutation.isPending}
            >
                <RefreshCw size={16} className={mutation.isPending ? 'spinner' : ''} />
                {mutation.isPending ? 'Procesando...' : 'Reentrenar Modelo'}
            </button>
        </div>
    );
}

interface EventModalProps {
    event?: Event | null;
    onClose: () => void;
    onSave: (data: Partial<Event>) => void;
    isLoading?: boolean;
}

function EventModal({ event, onClose, onSave, isLoading }: EventModalProps) {
    const [formData, setFormData] = useState({
        event_date: event?.event_date?.split('T')[0] || new Date().toISOString().split('T')[0],
        event_type: event?.event_type || 'confirmed_relapse',
        severity: event?.severity || '',
        notes: event?.notes || '',
        medication_start_date: event?.medication_start_date?.split('T')[0] || '',
    });

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onSave({
            event_date: new Date(formData.event_date).toISOString(),
            event_type: formData.event_type as Event['event_type'],
            severity: formData.severity as Event['severity'] || undefined,
            notes: formData.notes || undefined,
            medication_start_date: formData.medication_start_date
                ? new Date(formData.medication_start_date).toISOString()
                : undefined,
        });
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h2 className="modal-title">
                        {event ? 'Editar Evento' : 'Nuevo Evento'}
                    </h2>
                    <button className="modal-close" onClick={onClose}>
                        <X size={24} />
                    </button>
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
                    </div>

                    <div className="form-group">
                        <label className="form-label">Tipo de evento *</label>
                        <select
                            className="form-select"
                            value={formData.event_type}
                            onChange={e => setFormData({ ...formData, event_type: e.target.value as Event['event_type'] })}
                        >
                            <option value="symptom_onset">Inicio de Síntomas</option>
                            <option value="confirmed_relapse">Brote Confirmado</option>
                            <option value="medication_start">Inicio de Medicación</option>
                            <option value="hospital_visit">Visita al Hospital</option>
                            <option value="doctor_appointment">Cita Médica</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Severidad</label>
                        <div className="form-radio-group">
                            <label className="form-radio">
                                <input
                                    type="radio"
                                    name="severity"
                                    value="mild"
                                    checked={formData.severity === 'mild'}
                                    onChange={e => setFormData({ ...formData, severity: e.target.value })}
                                />
                                <span>Leve</span>
                            </label>
                            <label className="form-radio">
                                <input
                                    type="radio"
                                    name="severity"
                                    value="moderate"
                                    checked={formData.severity === 'moderate'}
                                    onChange={e => setFormData({ ...formData, severity: e.target.value })}
                                />
                                <span>Moderado</span>
                            </label>
                            <label className="form-radio">
                                <input
                                    type="radio"
                                    name="severity"
                                    value="severe"
                                    checked={formData.severity === 'severe'}
                                    onChange={e => setFormData({ ...formData, severity: e.target.value })}
                                />
                                <span>Severo</span>
                            </label>
                        </div>
                    </div>

                    {(formData.event_type === 'confirmed_relapse' || formData.event_type === 'symptom_onset') && (
                        <div className="form-group">
                            <label className="form-label">Fecha de inicio de medicación</label>
                            <input
                                type="date"
                                className="form-input"
                                value={formData.medication_start_date}
                                onChange={e => setFormData({ ...formData, medication_start_date: e.target.value })}
                            />
                        </div>
                    )}

                    <div className="form-group">
                        <label className="form-label">Notas</label>
                        <textarea
                            className="form-textarea"
                            value={formData.notes}
                            onChange={e => setFormData({ ...formData, notes: e.target.value })}
                            placeholder="Describe el evento..."
                        />
                    </div>

                    <div className="modal-footer">
                        <button type="button" className="btn btn-secondary" onClick={onClose}>
                            Cancelar
                        </button>
                        <button type="submit" className="btn btn-primary" disabled={isLoading}>
                            {isLoading ? 'Guardando...' : 'Guardar'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}

interface ImportModalProps {
    onClose: () => void;
    onSuccess: () => void;
}

function ImportModal({ onClose, onSuccess }: ImportModalProps) {
    const [step, setStep] = useState<'upload' | 'preview'>('upload');
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setError(null);
        }
    };

    const handlePreview = async () => {
        if (!file) return;
        setIsLoading(true);
        setError(null);
        try {
            const res = await eventsApi.previewImport(file);
            setPreview(res.data);
            setStep('preview');
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Error leyendo el archivo');
        } finally {
            setIsLoading(false);
        }
    };

    const handleConfirm = async () => {
        if (!file) return;
        setIsLoading(true);
        setError(null);
        try {
            const res = await eventsApi.confirmImport(file);
            alert(`Importación completada: ${res.data.imported} ${res.data.type || 'registros'}`);
            onSuccess();
            onClose();
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Error durante la importación');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: '600px' }}>
                <div className="modal-header">
                    <h2 className="modal-title">Importar Datos</h2>
                    <button className="modal-close" onClick={onClose}>
                        <X size={24} />
                    </button>
                </div>

                <div className="modal-body">
                    {step === 'upload' ? (
                        <div className="upload-zone">
                            <p style={{ marginBottom: '1rem' }}>
                                Soporta <strong>CSV de Eventos</strong> y <strong>WhatsApp export (.txt)</strong>.
                            </p>

                            <input
                                type="file"
                                accept=".csv,.txt"
                                onChange={handleFileChange}
                                className="form-input"
                            />

                            {error && (
                                <div className="alert-banner warning" style={{ marginTop: '1rem' }}>
                                    <AlertCircle size={16} />
                                    <span>{error}</span>
                                </div>
                            )}

                            <div style={{ marginTop: 'var(--space-md)', fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
                                <p><strong>CSV Eventos:</strong> columnas <code>date</code>, <code>event_type</code> (opcionales: <code>severity</code>, <code>notes</code>).</p>
                                <p><strong>WhatsApp:</strong> Exportar chat sin archivos multimedia (.txt).</p>
                            </div>
                        </div>
                    ) : (
                        <div className="preview-zone">
                            {preview && (
                                <>
                                    <div className="stats-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
                                        <div className="stat-card">
                                            <div className="stat-value">{preview.valid_events}</div>
                                            <div className="stat-label">Registros Válidos</div>
                                        </div>
                                        <div className="stat-card" style={{ borderColor: preview.invalid_events > 0 ? 'var(--color-danger)' : undefined }}>
                                            <div className="stat-value" style={{ color: preview.invalid_events > 0 ? 'var(--color-danger)' : undefined }}>
                                                {preview.invalid_events}
                                            </div>
                                            <div className="stat-label">Errores / Ignorados</div>
                                        </div>
                                    </div>

                                    <h4 style={{ margin: '1rem 0 0.5rem' }}>Vista Previa</h4>
                                    <div style={{ background: 'var(--color-surface)', padding: '0.5rem', borderRadius: '4px', maxHeight: '200px', overflowY: 'auto', fontSize: '0.8rem', fontFamily: 'monospace' }}>
                                        {preview.preview.map((item: any, idx: number) => (
                                            <div key={idx} style={{ borderBottom: '1px solid var(--color-border)', padding: '4px 0' }}>
                                                {JSON.stringify(item)}
                                            </div>
                                        ))}
                                    </div>

                                    {preview.errors && preview.errors.length > 0 && (
                                        <div style={{ marginTop: '1rem' }}>
                                            <h4 style={{ color: 'var(--color-danger)' }}>Errores</h4>
                                            <ul style={{ fontSize: '0.8rem', color: 'var(--color-danger)' }}>
                                                {preview.errors.map((e: string, i: number) => <li key={i}>{e}</li>)}
                                            </ul>
                                        </div>
                                    )}
                                </>
                            )}
                            {error && (
                                <div className="alert-banner danger" style={{ marginTop: '1rem' }}>
                                    <AlertCircle size={16} />
                                    <span>{error}</span>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                <div className="modal-footer">
                    <button type="button" className="btn btn-secondary" onClick={onClose} disabled={isLoading}>
                        Cancelar
                    </button>
                    {step === 'upload' ? (
                        <button
                            type="button"
                            className="btn btn-primary"
                            onClick={handlePreview}
                            disabled={!file || isLoading}
                        >
                            {isLoading ? 'Analizando...' : 'Previsualizar'}
                        </button>
                    ) : (
                        <button
                            type="button"
                            className="btn btn-success"
                            onClick={handleConfirm}
                            disabled={isLoading || (preview?.valid_events === 0)}
                        >
                            <Upload size={16} />
                            {isLoading ? 'Importando...' : 'Confirmar Importación'}
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
}

export default function EventsPage() {
    const queryClient = useQueryClient();
    const [showModal, setShowModal] = useState(false);
    const [showImportModal, setShowImportModal] = useState(false);
    const [editingEvent, setEditingEvent] = useState<Event | null>(null);

    // Queries
    const { data: events = [], isLoading: loadingEvents } = useQuery({
        queryKey: ['events'],
        queryFn: () => eventsApi.list().then(r => r.data),
    });

    const { data: stats } = useQuery({
        queryKey: ['events-stats'],
        queryFn: () => eventsApi.getStats().then(r => r.data),
    });

    const { data: retrainingStatus } = useQuery({
        queryKey: ['retraining-status'],
        queryFn: () => eventsApi.getRetrainingStatus().then(r => r.data),
    });

    // Mutations
    const createMutation = useMutation({
        mutationFn: (data: Partial<Event>) => eventsApi.create(data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['events'] });
            queryClient.invalidateQueries({ queryKey: ['events-stats'] });
            queryClient.invalidateQueries({ queryKey: ['retraining-status'] });
            setShowModal(false);
        },
    });

    const updateMutation = useMutation({
        mutationFn: ({ id, data }: { id: string; data: Partial<Event> }) =>
            eventsApi.update(id, data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['events'] });
            queryClient.invalidateQueries({ queryKey: ['retraining-status'] });
            setShowModal(false);
            setEditingEvent(null);
        },
    });

    const deleteMutation = useMutation({
        mutationFn: (id: string) => eventsApi.delete(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['events'] });
            queryClient.invalidateQueries({ queryKey: ['events-stats'] });
            queryClient.invalidateQueries({ queryKey: ['retraining-status'] });
        },
    });

    const handleSave = (data: Partial<Event>) => {
        if (editingEvent) {
            updateMutation.mutate({ id: editingEvent.id, data });
        } else {
            createMutation.mutate(data);
        }
    };

    const handleEdit = (event: Event) => {
        setEditingEvent(event);
        setShowModal(true);
    };

    const handleDelete = (event: Event) => {
        if (confirm(`¿Eliminar evento del ${format(new Date(event.event_date), 'dd/MM/yyyy')}?`)) {
            deleteMutation.mutate(event.id);
        }
    };

    return (
        <div>
            <div className="page-header">
                <div>
                    <h1 className="page-title">Eventos Clínicos</h1>
                    <p className="page-subtitle">Gestiona tus eventos médicos y brotes</p>
                </div>
                <button className="btn btn-primary" onClick={() => setShowModal(true)}>
                    <Plus size={18} />
                    Nuevo Evento
                </button>
            </div>

            {retrainingStatus && <RetrainingBanner status={retrainingStatus} />}

            {/* Stats */}
            {stats && (
                <div className="stats-grid">
                    <div className="stat-card">
                        <div className="stat-value">{stats.total_events}</div>
                        <div className="stat-label">Total Eventos</div>
                    </div>
                    <div className="stat-card danger">
                        <div className="stat-value">{stats.confirmed_relapses}</div>
                        <div className="stat-label">Brotes Confirmados</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value">{stats.medication_starts}</div>
                        <div className="stat-label">Inicios Medicación</div>
                    </div>
                    <div className="stat-card warning">
                        <div className="stat-value">{stats.pending_clusters}</div>
                        <div className="stat-label">Clusters Pendientes</div>
                    </div>
                </div>
            )}

            {/* Event List */}
            <div className="card">
                <div className="card-header">
                    <h2 className="card-title">Historial de Eventos</h2>
                    <button
                        className="btn btn-secondary"
                        onClick={() => setShowImportModal(true)}
                    >
                        <Upload size={16} />
                        Importar Datos
                    </button>
                </div>

                {loadingEvents ? (
                    <div className="loading">
                        <div className="spinner" />
                    </div>
                ) : events.length === 0 ? (
                    <div className="empty-state">
                        <Calendar size={64} className="empty-state-icon" />
                        <h3>No hay eventos registrados</h3>
                        <p>Añade tu primer evento clínico para comenzar el seguimiento.</p>
                    </div>
                ) : (
                    <div className="event-list">
                        {events.map(event => (
                            <div key={event.id} className="event-item">
                                <EventIcon type={event.event_type} />

                                <div className="event-content">
                                    <div className="event-header">
                                        <h3 className="event-title">
                                            {EVENT_TYPE_LABELS[event.event_type]}
                                        </h3>
                                        <span className="event-date">
                                            {format(new Date(event.event_date), "d 'de' MMMM yyyy", { locale: es })}
                                        </span>
                                    </div>

                                    {event.notes && (
                                        <p className="event-notes">{event.notes}</p>
                                    )}

                                    <div className="event-meta">
                                        {event.severity && (
                                            <span className={`badge ${event.severity}`}>
                                                {event.severity === 'mild' ? 'Leve' :
                                                    event.severity === 'moderate' ? 'Moderado' : 'Severo'}
                                            </span>
                                        )}
                                        {event.medication_start_date && (
                                            <span style={{ color: 'var(--color-text-secondary)', fontSize: '0.875rem' }}>
                                                💊 Medicación: {format(new Date(event.medication_start_date), 'dd/MM/yyyy')}
                                            </span>
                                        )}
                                        {event.requires_retraining && (
                                            <span className="badge pending">Pendiente reentrenamiento</span>
                                        )}
                                    </div>
                                </div>

                                <div className="event-actions">
                                    <button
                                        className="btn btn-icon btn-secondary"
                                        onClick={() => handleEdit(event)}
                                        title="Editar"
                                    >
                                        <Edit size={16} />
                                    </button>
                                    <button
                                        className="btn btn-icon btn-secondary"
                                        onClick={() => handleDelete(event)}
                                        title="Eliminar"
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Modal Edit/Create */}
            {showModal && (
                <EventModal
                    event={editingEvent}
                    onClose={() => {
                        setShowModal(false);
                        setEditingEvent(null);
                    }}
                    onSave={handleSave}
                    isLoading={createMutation.isPending || updateMutation.isPending}
                />
            )}

            {/* Import Modal */}
            {showImportModal && (
                <ImportModal
                    onClose={() => setShowImportModal(false)}
                    onSuccess={() => {
                        queryClient.invalidateQueries({ queryKey: ['events'] });
                        queryClient.invalidateQueries({ queryKey: ['events-stats'] });
                        queryClient.invalidateQueries({ queryKey: ['retraining-status'] });
                    }}
                />
            )}
        </div>
    );
}
