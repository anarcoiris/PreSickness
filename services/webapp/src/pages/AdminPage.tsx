import { useState, useEffect } from 'react';
import { adminApi, systemApi, type UserResponse, type SystemIncident, type SystemMetrics } from '../api/client';
import {
    Users,
    ShieldAlert,
    Server,
    Activity,
    Database,
    RefreshCcw,
    CheckCircle,
    AlertTriangle,
    Mail,
    UserCircle,
    Check
} from 'lucide-react';

export default function AdminPage() {
    const [users, setUsers] = useState<UserResponse[]>([]);
    const [incidents, setIncidents] = useState<SystemIncident[]>([]);
    const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [activeTab, setActiveTab] = useState<'overview' | 'users' | 'incidents'>('overview');
    const [msg, setMsg] = useState<{ type: 'success' | 'error', text: string } | null>(null);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        setIsLoading(true);
        try {
            const [usersRes, incidentsRes, metricsRes] = await Promise.all([
                adminApi.listUsers(),
                adminApi.listIncidents(),
                systemApi.metrics()
            ]);
            setUsers(usersRes.data);
            setIncidents(incidentsRes.data);
            setMetrics(metricsRes.data);
        } catch (err) {
            console.error("Error loading admin data", err);
            setMsg({ type: 'error', text: 'Error al cargar datos administrativos. ¿Tienes permisos?' });
        } finally {
            setIsLoading(false);
        }
    };

    const handleUpdateRole = async (userId: string, newRole: string) => {
        try {
            await adminApi.updateUserRole(userId, newRole);
            setMsg({ type: 'success', text: `Rol actualizado a ${newRole}` });
            loadData();
        } catch (err) {
            setMsg({ type: 'error', text: 'No se pudo actualizar el rol' });
        }
    };

    const handleResolveIncident = async (id: string) => {
        try {
            await adminApi.resolveIncident(id);
            setMsg({ type: 'success', text: 'Incidencia marcada como resuelta' });
            loadData();
        } catch (err) {
            setMsg({ type: 'error', text: 'Error al resolver incidencia' });
        }
    };

    if (isLoading && !users.length) {
        return (
            <div className="loading-state">
                <div className="spinner"></div>
                <p>Cargando panel de administración...</p>
            </div>
        );
    }

    return (
        <div className="page-container admin-page fade-in">
            <header className="page-header">
                <div>
                    <h1 className="page-title">Panel de Control</h1>
                    <p className="page-subtitle">Administración global del sistema y usuarios</p>
                </div>
                <div className="flex gap-2">
                    <button className="btn btn-secondary" onClick={loadData}>
                        <RefreshCcw size={18} />
                        Actualizar
                    </button>
                </div>
            </header>

            {msg && (
                <div className={`alert ${msg.type === 'error' ? 'alert-danger' : 'alert-success'} mb-6`}>
                    {msg.text}
                </div>
            )}

            <div className="admin-tabs">
                <button
                    className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
                    onClick={() => setActiveTab('overview')}
                >
                    <Activity size={18} />
                    Resumen
                </button>
                <button
                    className={`tab-btn ${activeTab === 'users' ? 'active' : ''}`}
                    onClick={() => setActiveTab('users')}
                >
                    <Users size={18} />
                    Usuarios
                </button>
                <button
                    className={`tab-btn ${activeTab === 'incidents' ? 'active' : ''}`}
                    onClick={() => setActiveTab('incidents')}
                >
                    <ShieldAlert size={18} />
                    Incidencias {incidents.filter(i => !i.resolved).length > 0 && <span className="badge">{incidents.filter(i => !i.resolved).length}</span>}
                </button>
            </div>

            <div className="tab-content mt-6">
                {activeTab === 'overview' && metrics && (
                    <div className="overview-grid slide-in">
                        <div className="stats-cards">
                            <div className="stat-card">
                                <div className="stat-icon users"><Users size={24} /></div>
                                <div className="stat-value">{users.length}</div>
                                <div className="stat-label">Usuarios Totales</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-icon database"><Database size={24} /></div>
                                <div className="stat-value">{metrics.total_datapoints}</div>
                                <div className="stat-label">Datapoints Procesados</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-icon msgs"><Mail size={24} /></div>
                                <div className="stat-value">{metrics.total_messages}</div>
                                <div className="stat-label">Mensajes Recibidos</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-icon incidents"><ShieldAlert size={24} /></div>
                                <div className="stat-value">{incidents.filter(i => !i.resolved).length}</div>
                                <div className="stat-label">Incidencias Activas</div>
                            </div>
                        </div>

                        <div className="card mt-6">
                            <div className="card-header">
                                <h3><Server size={20} /> Estado de Servicios</h3>
                            </div>
                            <div className="card-body">
                                <div className="services-status-list">
                                    {Object.entries(metrics.services_status).map(([name, status]) => (
                                        <div key={name} className="service-status-item">
                                            <div className="service-name">{name.replace('_', ' ').toUpperCase()}</div>
                                            <div className={`service-badge ${status}`}>
                                                {status === 'ok' ? <CheckCircle size={14} /> : <AlertTriangle size={14} />}
                                                {status}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'users' && (
                    <div className="user-management-table card slide-in">
                        <table className="admin-table">
                            <thead>
                                <tr>
                                    <th>Usuario</th>
                                    <th>Email</th>
                                    <th>Rol</th>
                                    <th>Fecha Registro</th>
                                    <th>Acciones</th>
                                </tr>
                            </thead>
                            <tbody>
                                {users.map(user => (
                                    <tr key={user.id}>
                                        <td>
                                            <div className="flex items-center gap-3">
                                                <UserCircle size={20} className="text-muted" />
                                                <span className="font-semibold">{user.name}</span>
                                            </div>
                                        </td>
                                        <td>{user.email}</td>
                                        <td>
                                            <select
                                                value={user.role}
                                                onChange={(e) => handleUpdateRole(user.id, e.target.value)}
                                                className="role-select"
                                            >
                                                <option value="patient">Paciente</option>
                                                <option value="doctor">Médico</option>
                                                <option value="admin">Admin</option>
                                            </select>
                                        </td>
                                        <td>{new Date(user.created_at).toLocaleDateString()}</td>
                                        <td>
                                            <button className="btn-text" title="Ver detalles">Config.</button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}

                {activeTab === 'incidents' && (
                    <div className="incidents-list slide-in">
                        {incidents.length === 0 ? (
                            <div className="empty-state card">
                                <CheckCircle size={48} className="text-success" />
                                <h3>No hay incidencias activas</h3>
                                <p>El sistema está funcionando nominalmente.</p>
                            </div>
                        ) : (
                            <div className="incidents-stack">
                                {incidents.map(incident => (
                                    <div key={incident.id} className={`incident-card ${incident.severity} ${incident.resolved ? 'resolved' : ''}`}>
                                        <div className="incident-header">
                                            <div className="flex items-center gap-2">
                                                <span className={`severity-indicator ${incident.severity}`}></span>
                                                <span className="component-label">{incident.component}</span>
                                            </div>
                                            <span className="incident-time">{new Date(incident.created_at).toLocaleString()}</span>
                                        </div>
                                        <div className="incident-body">
                                            <p className="incident-message">{incident.message}</p>
                                            {incident.details && (
                                                <pre className="incident-details">{JSON.stringify(incident.details, null, 2)}</pre>
                                            )}
                                        </div>
                                        {!incident.resolved && (
                                            <div className="incident-footer">
                                                <button
                                                    className="btn btn-sm btn-success"
                                                    onClick={() => handleResolveIncident(incident.id)}
                                                >
                                                    <Check size={16} />
                                                    Marcar como resuelta
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>

            <style>{`
                .admin-tabs {
                    display: flex;
                    gap: 1rem;
                    border-bottom: 1px solid var(--color-border);
                    margin-bottom: 2rem;
                }
                .tab-btn {
                    padding: 0.75rem 1.5rem;
                    background: none;
                    border: none;
                    color: var(--color-text-secondary);
                    font-weight: 500;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    cursor: pointer;
                    position: relative;
                    transition: all 0.2s;
                }
                .tab-btn:hover { color: white; }
                .tab-btn.active { color: var(--color-accent); }
                .tab-btn.active::after {
                    content: '';
                    position: absolute;
                    bottom: -1px;
                    left: 0;
                    right: 0;
                    height: 2px;
                    background: var(--color-accent);
                }
                .badge {
                    background: var(--color-danger);
                    color: white;
                    font-size: 0.7rem;
                    padding: 0.1rem 0.4rem;
                    border-radius: 99px;
                    font-weight: 700;
                }
                
                .stats-cards {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1.5rem;
                }
                .stat-card {
                    background: var(--color-surface);
                    border: 1px solid var(--color-border);
                    border-radius: var(--radius-lg);
                    padding: 1.5rem;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                }
                .stat-icon {
                    width: 48px;
                    height: 48px;
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-bottom: 1rem;
                }
                .stat-icon.users { background: rgba(59, 130, 246, 0.1); color: #3b82f6; }
                .stat-icon.database { background: rgba(16, 185, 129, 0.1); color: #10b981; }
                .stat-icon.msgs { background: rgba(245, 158, 11, 0.1); color: #f59e0b; }
                .stat-icon.incidents { background: rgba(239, 68, 68, 0.1); color: #ef4444; }
                
                .stat-value { font-size: 1.75rem; font-weight: 700; color: white; }
                .stat-label { font-size: 0.875rem; color: var(--color-text-secondary); }
                
                .services-status-list {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }
                .service-status-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 0.75rem;
                    background: rgba(255,255,255,0.02);
                    border-radius: var(--radius-md);
                }
                .service-badge {
                    display: flex;
                    align-items: center;
                    gap: 0.4rem;
                    font-size: 0.75rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    padding: 0.25rem 0.6rem;
                    border-radius: 4px;
                }
                .service-badge.ok { color: var(--color-success); background: rgba(var(--color-success-rgb), 0.1); }
                .service-badge.unreachable { color: var(--color-danger); background: rgba(var(--color-danger-rgb), 0.1); }
                
                .admin-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .admin-table th {
                    text-align: left;
                    padding: 1rem;
                    border-bottom: 1px solid var(--color-border);
                    color: var(--color-text-secondary);
                    font-weight: 600;
                    font-size: 0.875rem;
                }
                .admin-table td {
                    padding: 1rem;
                    border-bottom: 1px solid var(--color-border);
                    font-size: 0.93rem;
                }
                .role-select {
                    background: var(--color-bg);
                    border: 1px solid var(--color-border);
                    color: white;
                    padding: 0.25rem 0.5rem;
                    border-radius: 4px;
                    font-size: 0.85rem;
                }
                
                .incidents-stack { display: flex; flex-direction: column; gap: 1rem; }
                .incident-card {
                    background: var(--color-surface);
                    border: 1px solid var(--color-border);
                    border-left: 4px solid var(--color-border);
                    border-radius: var(--radius-md);
                    padding: 1rem;
                    transition: all 0.2s;
                }
                .incident-card.critical { border-left-color: var(--color-danger); }
                .incident-card.error { border-left-color: #f87171; }
                .incident-card.warning { border-left-color: var(--color-warning); }
                .incident-card.resolved { opacity: 0.6; border-left-color: var(--color-success); }
                
                .incident-header { display: flex; justify-content: space-between; margin-bottom: 0.75rem; }
                .component-label { font-weight: 700; color: white; text-transform: uppercase; font-size: 0.75rem; }
                .incident-time { font-size: 0.75rem; color: var(--color-text-muted); }
                .incident-message { font-weight: 500; margin-bottom: 0.5rem; color: #e5e7eb; }
                .incident-details {
                    background: #111;
                    padding: 0.75rem;
                    border-radius: 4px;
                    font-size: 0.75rem;
                    font-family: monospace;
                    overflow-x: auto;
                    color: #9ca3af;
                }
                .incident-footer { margin-top: 1rem; display: flex; justify-content: flex-end; }
                
                .severity-indicator {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                }
                .severity-indicator.critical { background: var(--color-danger); box-shadow: 0 0 8px var(--color-danger); }
                .severity-indicator.error { background: #f87171; }
                .severity-indicator.warning { background: var(--color-warning); }
            `}</style>
        </div>
    );
}
