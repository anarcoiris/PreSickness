import { useState, useEffect } from 'react';
import { patientAPI, authAPI } from '../api/client';
import { useNavigate } from 'react-router-dom';

export default function Profile() {
    const navigate = useNavigate();
    const [profile, setProfile] = useState(null);
    const [uploads, setUploads] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadProfile();
    }, []);

    const loadProfile = async () => {
        try {
            const [profileData, uploadsData] = await Promise.all([
                patientAPI.getProfile(),
                patientAPI.listUploads(),
            ]);
            setProfile(profileData);
            setUploads(uploadsData);
        } catch (err) {
            console.error('Error loading profile:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleLogout = () => {
        authAPI.logout();
        navigate('/login');
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
                <h1 className="page-title">👤 Mi Perfil</h1>
                <p className="page-subtitle">
                    Gestiona tu información y preferencias
                </p>
            </div>

            <div className="grid grid-2">
                {/* Info del usuario */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Información Personal</h3>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '2rem' }}>
                        <div style={{
                            width: '80px',
                            height: '80px',
                            borderRadius: '50%',
                            background: 'linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '2rem',
                        }}>
                            {profile?.name?.charAt(0).toUpperCase() || '?'}
                        </div>
                        <div>
                            <h2 style={{ marginBottom: '0.25rem' }}>{profile?.name}</h2>
                            <p style={{ color: 'var(--text-secondary)' }}>{profile?.email}</p>
                        </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            padding: '1rem',
                            background: 'var(--bg-dark)',
                            borderRadius: '8px'
                        }}>
                            <span style={{ color: 'var(--text-secondary)' }}>ID de Usuario</span>
                            <span style={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
                                {profile?.id}
                            </span>
                        </div>

                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            padding: '1rem',
                            background: 'var(--bg-dark)',
                            borderRadius: '8px'
                        }}>
                            <span style={{ color: 'var(--text-secondary)' }}>Miembro desde</span>
                            <span>
                                {profile?.created_at
                                    ? new Date(profile.created_at).toLocaleDateString('es-ES', {
                                        day: 'numeric',
                                        month: 'long',
                                        year: 'numeric'
                                    })
                                    : 'N/A'
                                }
                            </span>
                        </div>

                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            padding: '1rem',
                            background: 'var(--bg-dark)',
                            borderRadius: '8px'
                        }}>
                            <span style={{ color: 'var(--text-secondary)' }}>Archivos subidos</span>
                            <span>{uploads.length}</span>
                        </div>
                    </div>
                </div>

                {/* Acciones y configuración */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    <div className="card">
                        <div className="card-header">
                            <h3 className="card-title">⚙️ Configuración</h3>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                            <label style={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                padding: '0.75rem 1rem',
                                background: 'var(--bg-dark)',
                                borderRadius: '8px',
                                cursor: 'pointer'
                            }}>
                                <span>Notificaciones por email</span>
                                <input type="checkbox" defaultChecked />
                            </label>

                            <label style={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                padding: '0.75rem 1rem',
                                background: 'var(--bg-dark)',
                                borderRadius: '8px',
                                cursor: 'pointer'
                            }}>
                                <span>Alertas de riesgo</span>
                                <input type="checkbox" defaultChecked />
                            </label>

                            <label style={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                padding: '0.75rem 1rem',
                                background: 'var(--bg-dark)',
                                borderRadius: '8px',
                                cursor: 'pointer'
                            }}>
                                <span>Resumen semanal</span>
                                <input type="checkbox" />
                            </label>
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-header">
                            <h3 className="card-title">🔒 Seguridad</h3>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                            <button className="btn btn-secondary" style={{ justifyContent: 'flex-start' }}>
                                🔑 Cambiar contraseña
                            </button>
                            <button className="btn btn-secondary" style={{ justifyContent: 'flex-start' }}>
                                📱 Configurar 2FA
                            </button>
                            <button className="btn btn-secondary" style={{ justifyContent: 'flex-start' }}>
                                📦 Exportar mis datos
                            </button>
                        </div>
                    </div>

                    <button
                        className="btn btn-danger"
                        style={{ width: '100%' }}
                        onClick={handleLogout}
                    >
                        🚪 Cerrar Sesión
                    </button>
                </div>
            </div>
        </div>
    );
}
