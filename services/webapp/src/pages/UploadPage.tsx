import { useState, useRef, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { patientApi } from '../api/client';
import type { PatientUpload } from '../api/client';
import { Upload, File, CheckCircle, AlertCircle, Clock, Info, ExternalLink } from 'lucide-react';

export default function UploadPage() {
    const queryClient = useQueryClient();
    const [dragActive, setDragActive] = useState(false);
    const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const { data: uploads = [], isLoading: loadingUploads } = useQuery({
        queryKey: ['uploads'],
        queryFn: () => patientApi.listUploads().then(res => res.data),
    });

    const { data: messageStats } = useQuery({
        queryKey: ['messageStats'],
        queryFn: () => patientApi.getMessageStats().then(res => res.data),
        refetchInterval: 10000, // Refresh every 10s
    });

    const processMutation = useMutation({
        mutationFn: () => patientApi.triggerProcessing().then(res => res.data),
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: ['messageStats'] });
            setMessage({ type: 'success', text: `Procesados ${data.processed} mensajes` });
            setTimeout(() => setMessage(null), 5000);
        },
        onError: () => {
            setMessage({ type: 'error', text: 'Error al procesar mensajes' });
        }
    });

    const uploadMutation = useMutation({
        mutationFn: (file: File) => patientApi.uploadData(file).then(res => res.data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['uploads'] });
            setMessage({ type: 'success', text: '¡Archivo subido correctamente! Se procesará en breve.' });
            setTimeout(() => setMessage(null), 5000);
        },
        onError: (error: any) => {
            setMessage({
                type: 'error',
                text: error.response?.data?.detail || 'Error al subir el archivo. Inténtalo de nuevo.'
            });
        }
    });

    const handleUpload = async (file: File) => {
        if (!file) return;

        const allowedTypes = ['.csv', '.json', '.xlsx', '.txt'];
        const ext = '.' + file.name.split('.').pop()?.toLowerCase();

        if (!allowedTypes.includes(ext)) {
            setMessage({
                type: 'error',
                text: `Tipo de archivo no permitido. Formatos aceptados: ${allowedTypes.join(', ')}`
            });
            return;
        }

        uploadMutation.mutate(file);
    };

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleUpload(e.dataTransfer.files[0]);
        }
    }, []);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            handleUpload(e.target.files[0]);
        }
    };

    return (
        <div className="fade-in">
            <div className="page-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <Upload size={32} color="var(--primary)" />
                    <h1 className="page-title">Subir Datos</h1>
                </div>
                <p className="page-subtitle">
                    Importa tus registros de salud, chats de WhatsApp o datos de wearables para alimentar la IA.
                </p>
            </div>

            {message && (
                <div style={{
                    background: message.type === 'success'
                        ? 'rgba(16, 185, 129, 0.1)'
                        : 'rgba(239, 68, 68, 0.1)',
                    border: `1px solid ${message.type === 'success' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                    borderRadius: '12px',
                    padding: '1rem 1.5rem',
                    marginBottom: '2rem',
                    color: message.type === 'success' ? 'var(--success)' : 'var(--danger)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    fontWeight: 500
                }}>
                    {message.type === 'success' ? <CheckCircle size={20} /> : <AlertCircle size={20} />}
                    {message.text}
                </div>
            )}

            <div className="card" style={{ marginBottom: '2.5rem', padding: 0, overflow: 'hidden' }}>
                <div
                    className={`upload-zone ${dragActive ? 'dragging' : ''} ${uploadMutation.isPending ? 'uploading' : ''}`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => !uploadMutation.isPending && fileInputRef.current?.click()}
                    style={{
                        padding: '4rem 2rem',
                        cursor: uploadMutation.isPending ? 'not-allowed' : 'pointer',
                        textAlign: 'center',
                        transition: 'all 0.3s ease',
                        background: dragActive ? 'rgba(99, 179, 237, 0.05)' : 'transparent',
                        border: '2px dashed rgba(255,255,255,0.1)',
                        margin: '1rem',
                        borderRadius: '12px'
                    }}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".csv,.json,.xlsx,.txt"
                        onChange={handleFileChange}
                        style={{ display: 'none' }}
                    />

                    {uploadMutation.isPending ? (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                            <div className="spinner" />
                            <p style={{ fontWeight: 600, color: 'var(--primary)' }}>Subiendo archivo...</p>
                        </div>
                    ) : (
                        <>
                            <div style={{
                                width: '64px',
                                height: '64px',
                                background: 'rgba(99, 179, 237, 0.1)',
                                borderRadius: '50%',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                margin: '0 auto 1.5rem'
                            }}>
                                <Upload size={32} color="var(--primary)" />
                            </div>
                            <p style={{ fontSize: '1.25rem', marginBottom: '0.5rem', fontWeight: 600 }}>
                                Arrastra tu archivo aquí
                            </p>
                            <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                                O haz clic para seleccionar un archivo de tu explorador
                            </p>
                            <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                <span style={{ padding: '0.25rem 0.5rem', background: 'rgba(255,255,255,0.05)', borderRadius: '4px' }}>CSV</span>
                                <span style={{ padding: '0.25rem 0.5rem', background: 'rgba(255,255,255,0.05)', borderRadius: '4px' }}>JSON</span>
                                <span style={{ padding: '0.25rem 0.5rem', background: 'rgba(255,255,255,0.05)', borderRadius: '4px' }}>XLSX</span>
                                <span style={{ padding: '0.25rem 0.5rem', background: 'rgba(255,255,255,0.05)', borderRadius: '4px' }}>TXT (WhatsApp)</span>
                            </div>
                        </>
                    )}
                </div>
            </div>

            <div className="grid grid-3" style={{ marginBottom: '2.5rem' }}>
                <div className="card" style={{ padding: '1.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                        <div style={{ color: 'var(--primary-light)' }}><MessageSquare size={24} /></div>
                        <h3 className="card-title" style={{ fontSize: '1rem' }}>WhatsApp</h3>
                    </div>
                    {messageStats ? (
                        <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Mensajes cargados:</span>
                                <strong style={{ color: 'var(--text-primary)' }}>{messageStats.raw_messages.toLocaleString()}</strong>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Procesados (features):</span>
                                <strong style={{ color: 'var(--success)' }}>{messageStats.processed_datapoints.toLocaleString()}</strong>
                            </div>
                            {messageStats.raw_messages > messageStats.processed_datapoints && (
                                <button
                                    className="btn btn-primary"
                                    style={{ width: '100%', marginTop: '1rem', fontSize: '0.8rem' }}
                                    onClick={() => processMutation.mutate()}
                                    disabled={processMutation.isPending}
                                >
                                    {processMutation.isPending ? 'Procesando...' : `Procesar ${messageStats.raw_messages - messageStats.processed_datapoints} pendientes`}
                                </button>
                            )}
                        </div>
                    ) : (
                        <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                            Exporta tu chat de WhatsApp y sube el .txt para análisis.
                        </p>
                    )}
                </div>
                <div className="card" style={{ padding: '1.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                        <div style={{ color: 'var(--success)' }}><Activity size={24} /></div>
                        <h3 className="card-title" style={{ fontSize: '1rem' }}>Wearables</h3>
                    </div>
                    <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                        Importa datos de pasos, sueño y frecuencia cardíaca desde Fitbit, Apple Health o Google Fit.
                    </p>
                </div>
                <div className="card" style={{ padding: '1.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                        <div style={{ color: 'var(--secondary)' }}><File size={24} /></div>
                        <h3 className="card-title" style={{ fontSize: '1rem' }}>Registros Médicos</h3>
                    </div>
                    <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                        Carga archivos CSV con tus mediciones de síntomas diarios o registros de medicación.
                    </p>
                </div>
            </div>

            <div className="card">
                <div className="card-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Clock size={24} color="var(--text-muted)" />
                        <h3 className="card-title">Historial de Archivos</h3>
                    </div>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.875rem', background: 'rgba(255,255,255,0.05)', padding: '0.2rem 0.6rem', borderRadius: '4px' }}>
                        {uploads.length} total
                    </span>
                </div>

                {loadingUploads ? (
                    <div style={{ textAlign: 'center', padding: '3rem' }}>
                        <div className="spinner" style={{ margin: '0 auto' }} />
                    </div>
                ) : uploads.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--text-muted)' }}>
                        <div style={{ opacity: 0.3, marginBottom: '1rem' }}><File size={48} style={{ margin: '0 auto' }} /></div>
                        <p>No has subido ningún archivo todavía</p>
                        <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>Tus datos subidos aparecerán aquí.</p>
                    </div>
                ) : (
                    <div className="table-container">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Archivo</th>
                                    <th>Fecha de Subida</th>
                                    <th>Estado</th>
                                    <th>Acciones</th>
                                </tr>
                            </thead>
                            <tbody>
                                {uploads.map((upload: PatientUpload) => (
                                    <tr key={upload.id}>
                                        <td>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                                <div style={{ color: 'var(--primary-light)' }}>
                                                    <File size={20} />
                                                </div>
                                                <span style={{ fontWeight: 500 }}>{upload.filename}</span>
                                            </div>
                                        </td>
                                        <td style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                                            {new Date(upload.uploaded_at).toLocaleDateString('es-ES', {
                                                day: '2-digit',
                                                month: 'short',
                                                hour: '2-digit',
                                                minute: '2-digit'
                                            })}
                                        </td>
                                        <td>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                                <div style={{
                                                    width: '8px',
                                                    height: '8px',
                                                    borderRadius: '50%',
                                                    background: upload.processed ? 'var(--success)' : (upload.error_message ? 'var(--danger)' : 'var(--warning)')
                                                }} />
                                                <span style={{
                                                    fontSize: '0.875rem',
                                                    color: upload.processed ? 'var(--success)' : (upload.error_message ? 'var(--danger)' : 'var(--warning)')
                                                }}>
                                                    {upload.processed ? 'Procesado' : (upload.error_message ? 'Error' : 'Pendiente')}
                                                </span>
                                            </div>
                                        </td>
                                        <td>
                                            <button className="btn-icon" title="Ver detalles">
                                                <Info size={18} />
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            <div style={{ marginTop: '2rem', textAlign: 'center' }}>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                    <Info size={14} />
                    Tus datos se procesan localmente y de forma segura conforme a la RGPD.
                    <a href="#" style={{ color: 'var(--primary)', display: 'inline-flex', alignItems: 'center', gap: '0.2rem', textDecoration: 'none' }}>
                        Leer más <ExternalLink size={12} />
                    </a>
                </p>
            </div>
        </div>
    );
}

const MessageSquare = ({ size, color, className }: any) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color || "currentColor"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
);

const Activity = ({ size, color, className }: any) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color || "currentColor"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
);
