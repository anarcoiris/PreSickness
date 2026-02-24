import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { doctorApi, type DoctorPatient, patientContext } from '../api/client';
import { Users, UserPlus, Trash2, Mail, Calendar, Activity, AlertCircle, Check } from 'lucide-react';

export default function PatientsPage() {
    const [patients, setPatients] = useState<DoctorPatient[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isAdding, setIsAdding] = useState(false);
    const [newPatientEmail, setNewPatientEmail] = useState('');
    const [msg, setMsg] = useState<{ type: 'success' | 'error', text: string } | null>(null);
    const navigate = useNavigate();

    const handleViewDashboard = (patientId: string) => {
        patientContext.setPatient(patientId);
        navigate('/dashboard');
    };

    useEffect(() => {
        loadPatients();
    }, []);

    const loadPatients = async () => {
        try {
            const res = await doctorApi.listPatients();
            setPatients(res.data);
        } catch (err) {
            console.error("Error loading patients", err);
        } finally {
            setIsLoading(false);
        }
    };

    const handleAddPatient = async (e: React.FormEvent) => {
        e.preventDefault();
        setMsg(null);
        try {
            await doctorApi.addPatient(newPatientEmail);
            setMsg({ type: 'success', text: 'Paciente añadido correctamente' });
            setNewPatientEmail('');
            setIsAdding(false);
            loadPatients();
        } catch (err: any) {
            setMsg({ type: 'error', text: err.response?.data?.detail || 'Error al añadir paciente' });
        }
    };

    const handleRemovePatient = async (patientId: string) => {
        if (!confirm('¿Estás seguro de querer dejar de seguir a este paciente?')) return;
        try {
            await doctorApi.removePatient(patientId);
            loadPatients();
        } catch (err) {
            console.error("Error removing patient", err);
        }
    };

    return (
        <div className="page-container fade-in">
            <header className="page-header">
                <div>
                    <h1 className="page-title">Mis Pacientes</h1>
                    <p className="page-subtitle">Gestiona y monitoriza a tus pacientes asignados</p>
                </div>
                <button className="btn btn-primary" onClick={() => setIsAdding(true)}>
                    <UserPlus size={20} />
                    Añadir Paciente
                </button>
            </header>

            {msg && (
                <div className={`alert ${msg.type === 'error' ? 'alert-danger' : 'alert-success'}`}>
                    {msg.type === 'error' ? <AlertCircle size={20} /> : <Check size={20} />}
                    {msg.text}
                </div>
            )}

            {isAdding && (
                <div className="card mb-6 slide-in">
                    <div className="card-header">
                        <h3>Vincular Nuevo Paciente</h3>
                    </div>
                    <div className="card-body">
                        <form onSubmit={handleAddPatient} className="flex gap-4 items-end">
                            <div className="form-group flex-1">
                                <label className="form-label">Email del Paciente</label>
                                <div className="input-with-icon">
                                    <Mail size={18} className="input-icon" />
                                    <input
                                        type="email"
                                        className="form-input"
                                        placeholder="paciente@ejemplo.com"
                                        value={newPatientEmail}
                                        onChange={e => setNewPatientEmail(e.target.value)}
                                        required
                                    />
                                </div>
                            </div>
                            <div className="flex gap-2 mb-4">
                                <button type="button" className="btn btn-secondary" onClick={() => setIsAdding(false)}>
                                    Cancelar
                                </button>
                                <button type="submit" className="btn btn-primary">
                                    Vincular
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}

            {isLoading ? (
                <div className="loading-state">
                    <div className="spinner"></div>
                </div>
            ) : patients.length === 0 ? (
                <div className="empty-state">
                    <Users size={48} />
                    <h3>No tienes pacientes asignados</h3>
                    <p>Usa el botón "Añadir Paciente" para vincular pacientes existentes por su email.</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {patients.map(patient => (
                        <div key={patient.patient_id} className="card patient-card hover-lift">
                            <div className="card-header flex justify-between items-start">
                                <div className="flex items-center gap-3">
                                    <div className="avatar placeholder">
                                        {patient.patient_name.charAt(0).toUpperCase()}
                                    </div>
                                    <div>
                                        <h3 className="font-bold">{patient.patient_name}</h3>
                                        <span className="text-xs text-muted">{patient.patient_email}</span>
                                    </div>
                                </div>
                                <div className={`status-badge ${patient.status}`}>
                                    {patient.status}
                                </div>
                            </div>

                            <div className="card-body">
                                <div className="info-row">
                                    <Calendar size={16} />
                                    <span className="text-sm">Vinculado: {new Date(patient.granted_at).toLocaleDateString()}</span>
                                </div>
                                <div className="info-row">
                                    <Activity size={16} />
                                    <span className="text-sm">Acceso: {patient.access_level}</span>
                                </div>

                                <div className="mt-4 pt-4 border-t border-white/5 flex gap-2">
                                    <button
                                        className="btn btn-sm btn-secondary flex-1"
                                        onClick={() => handleViewDashboard(patient.patient_id)}
                                    >
                                        Ver Dashboard
                                    </button>
                                    <button
                                        className="btn btn-sm btn-icon btn-danger-outline"
                                        onClick={() => handleRemovePatient(patient.patient_id)}
                                        title="Dejar de seguir"
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            <style>{`
                .mb-6 { margin-bottom: 1.5rem; }
                .flex { display: flex; }
                .gap-4 { gap: 1rem; }
                .gap-2 { gap: 0.5rem; }
                .items-end { align-items: flex-end; }
                .flex-1 { flex: 1; }
                
                .avatar {
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    background: var(--color-accent);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    color: white;
                }
                
                .info-row {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    margin-bottom: 0.5rem;
                    color: var(--color-text-secondary);
                }
                
                .text-xs { font-size: 0.75rem; }
                .text-sm { font-size: 0.875rem; }
                .text-muted { color: var(--color-text-muted); }
                .font-bold { font-weight: 600; }
                
                .btn-sm { padding: 0.25rem 0.5rem; font-size: 0.875rem; }
                .btn-icon { padding: 0.5rem; }
                
                .btn-danger-outline {
                    background: transparent;
                    border: 1px solid var(--color-danger);
                    color: var(--color-danger);
                }
                .btn-danger-outline:hover {
                    background: var(--color-danger);
                    color: white;
                }
                
                .hover-lift { transition: transform 0.2s; }
                .hover-lift:hover { transform: translateY(-4px); }
                
                .status-badge {
                    padding: 0.25rem 0.5rem;
                    border-radius: 99px;
                    font-size: 0.7rem;
                    text-transform: uppercase;
                    font-weight: 700;
                }
                .status-badge.active { background: rgba(var(--color-success-rgb), 0.2); color: var(--color-success); }
                .status-badge.pending { background: rgba(var(--color-warning-rgb), 0.2); color: var(--color-warning); }
                
                .alert {
                    padding: 1rem;
                    border-radius: var(--radius-md);
                    margin-bottom: 1.5rem;
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                }
                .alert-success { background: rgba(72, 187, 120, 0.1); border: 1px solid var(--color-success); color: var(--color-success); }
                .alert-danger { background: rgba(245, 101, 101, 0.1); border: 1px solid var(--color-danger); color: var(--color-danger); }
            `}</style>
        </div>
    );
}
