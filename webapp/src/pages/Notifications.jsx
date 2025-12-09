import { useState, useEffect } from 'react';
import { systemAPI } from '../api/client';

export default function Notifications() {
    const [alerts, setAlerts] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadAlerts();
    }, []);

    const loadAlerts = async () => {
        try {
            const data = await systemAPI.getAlerts();
            setAlerts(data);
        } catch (err) {
            console.error('Error loading alerts:', err);
        } finally {
            setLoading(false);
        }
    };

    const getIcon = (type, level) => {
        if (level === 'critical') return '🚨';
        if (type === 'upload_reminder') return '📤';
        if (type === 'risk_increase') return '📈';
        return 'ℹ️';
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
                <h1 className="page-title">🔔 Notificaciones</h1>
                <p className="page-subtitle">
                    Alertas importantes y recordatorios
                </p>
            </div>

            <div className="card">
                {alerts.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                        <p>No tienes notificaciones nuevas</p>
                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
                        {alerts.map((alert, index) => (
                            <div
                                key={alert.id}
                                style={{
                                    display: 'flex',
                                    gap: '1rem',
                                    padding: '1.5rem',
                                    borderBottom: index < alerts.length - 1 ? '1px solid var(--border)' : 'none',
                                    background: alert.read ? 'transparent' : 'rgba(255, 255, 255, 0.02)',
                                    transition: 'background 0.2s',
                                }}
                            >
                                <div style={{ fontSize: '1.5rem', paddingTop: '0.25rem' }}>
                                    {getIcon(alert.alert_type, alert.alert_level)}
                                </div>

                                <div style={{ flex: 1 }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                        <h3 style={{
                                            fontSize: '1rem',
                                            fontWeight: 600,
                                            color: alert.read ? 'var(--text-primary)' : 'var(--primary-light)'
                                        }}>
                                            {alert.alert_level === 'critical' ? 'Alerta Crítica' :
                                                alert.alert_type === 'upload_reminder' ? 'Recordatorio' : 'Aviso'}
                                        </h3>
                                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                                            {new Date(alert.triggered_at).toLocaleDateString('es-ES', {
                                                day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'
                                            })}
                                        </span>
                                    </div>

                                    <p style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                                        {alert.message}
                                    </p>

                                    {!alert.read && (
                                        <div style={{ marginTop: '0.75rem' }}>
                                            <span className="status-badge online" style={{ fontSize: '0.7rem' }}>
                                                Nueva
                                            </span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
