import { useState, useRef, useCallback } from 'react';
import { patientAPI } from '../api/client';

export default function Upload() {
    const [uploads, setUploads] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [dragActive, setDragActive] = useState(false);
    const [message, setMessage] = useState(null);
    const fileInputRef = useRef(null);

    // Cargar uploads al montar
    useState(() => {
        patientAPI.listUploads()
            .then(setUploads)
            .catch(console.error);
    }, []);

    const handleUpload = async (file) => {
        if (!file) return;

        const allowedTypes = ['.csv', '.json', '.xlsx'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();

        if (!allowedTypes.includes(ext)) {
            setMessage({ type: 'error', text: `Tipo de archivo no permitido. Usa: ${allowedTypes.join(', ')}` });
            return;
        }

        setUploading(true);
        setMessage(null);

        try {
            const result = await patientAPI.uploadData(file);
            setUploads(prev => [result, ...prev]);
            setMessage({ type: 'success', text: '¡Archivo subido correctamente!' });
        } catch (err) {
            setMessage({ type: 'error', text: err.response?.data?.detail || 'Error al subir archivo' });
        } finally {
            setUploading(false);
        }
    };

    const handleDrag = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleUpload(e.dataTransfer.files[0]);
        }
    }, []);

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            handleUpload(e.target.files[0]);
        }
    };

    return (
        <div className="fade-in">
            <div className="page-header">
                <h1 className="page-title">📤 Subir Datos</h1>
                <p className="page-subtitle">
                    Sube tus exportaciones de datos de salud para análisis
                </p>
            </div>

            {/* Mensaje de feedback */}
            {message && (
                <div style={{
                    background: message.type === 'success'
                        ? 'rgba(16, 185, 129, 0.1)'
                        : 'rgba(239, 68, 68, 0.1)',
                    border: `1px solid ${message.type === 'success' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                    borderRadius: '8px',
                    padding: '1rem',
                    marginBottom: '1.5rem',
                    color: message.type === 'success' ? 'var(--success)' : 'var(--danger)'
                }}>
                    {message.text}
                </div>
            )}

            {/* Zona de upload */}
            <div className="card" style={{ marginBottom: '2rem' }}>
                <div
                    className={`upload-zone ${dragActive ? 'dragging' : ''}`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".csv,.json,.xlsx"
                        onChange={handleFileChange}
                        style={{ display: 'none' }}
                    />

                    {uploading ? (
                        <>
                            <div className="spinner" style={{ margin: '0 auto 1rem' }} />
                            <p>Subiendo archivo...</p>
                        </>
                    ) : (
                        <>
                            <div className="upload-icon">📁</div>
                            <p style={{ fontSize: '1.125rem', marginBottom: '0.5rem' }}>
                                Arrastra tu archivo aquí o haz clic para seleccionar
                            </p>
                            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                                Formatos soportados: CSV, JSON, XLSX
                            </p>
                        </>
                    )}
                </div>
            </div>

            {/* Instrucciones */}
            <div className="card" style={{ marginBottom: '2rem' }}>
                <h3 className="card-title" style={{ marginBottom: '1rem' }}>
                    📋 ¿Qué datos puedo subir?
                </h3>
                <div className="grid grid-3">
                    <div style={{ padding: '1rem', background: 'var(--bg-dark)', borderRadius: '8px' }}>
                        <h4 style={{ color: 'var(--primary-light)', marginBottom: '0.5rem' }}>WhatsApp</h4>
                        <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                            Exporta tu chat y súbelo como archivo de texto
                        </p>
                    </div>
                    <div style={{ padding: '1rem', background: 'var(--bg-dark)', borderRadius: '8px' }}>
                        <h4 style={{ color: 'var(--primary-light)', marginBottom: '0.5rem' }}>Wearables</h4>
                        <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                            Datos de Fitbit, Apple Health, Google Fit en CSV
                        </p>
                    </div>
                    <div style={{ padding: '1rem', background: 'var(--bg-dark)', borderRadius: '8px' }}>
                        <h4 style={{ color: 'var(--primary-light)', marginBottom: '0.5rem' }}>Custom</h4>
                        <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                            Cualquier CSV con columnas de fecha y valores
                        </p>
                    </div>
                </div>
            </div>

            {/* Lista de uploads */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">📂 Archivos Subidos</h3>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                        {uploads.length} archivo{uploads.length !== 1 ? 's' : ''}
                    </span>
                </div>

                {uploads.length === 0 ? (
                    <div style={{
                        textAlign: 'center',
                        padding: '3rem',
                        color: 'var(--text-muted)'
                    }}>
                        <p>No has subido ningún archivo todavía</p>
                    </div>
                ) : (
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Archivo</th>
                                <th>Fecha</th>
                                <th>Estado</th>
                            </tr>
                        </thead>
                        <tbody>
                            {uploads.map((upload) => (
                                <tr key={upload.id}>
                                    <td style={{ fontWeight: 500 }}>
                                        📄 {upload.filename}
                                    </td>
                                    <td style={{ color: 'var(--text-secondary)' }}>
                                        {new Date(upload.uploaded_at).toLocaleDateString('es-ES', {
                                            day: '2-digit',
                                            month: 'short',
                                            year: 'numeric',
                                            hour: '2-digit',
                                            minute: '2-digit'
                                        })}
                                    </td>
                                    <td>
                                        <span className={`status-badge ${upload.processed ? 'online' : 'offline'}`}>
                                            {upload.processed ? 'Procesado' : 'Pendiente'}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
}
