import { BrowserRouter, Routes, Route, NavLink, Navigate, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { GoogleOAuthProvider } from '@react-oauth/google';
import {
  Calendar,
  FileText,
  Settings,
  AlertTriangle,
  LogOut,
  Brain,
  LayoutDashboard,
  BarChart2,
  Upload,
  ShieldCheck,
} from 'lucide-react';

import DashboardPage from './pages/DashboardPage';
import AnalyticsPage from './pages/AnalyticsPage';
import UploadPage from './pages/UploadPage';
import EventsPage from './pages/EventsPage';
import ClustersPage from './pages/ClustersPage';
import CalendarPage from './pages/CalendarPage';
import SettingsPage from './pages/SettingsPage';
import LoginPage from './pages/LoginPage';
import PatientsPage from './pages/PatientsPage';
import ModelAnalysisPage from './pages/ModelAnalysisPage';
import AdminPage from './pages/AdminPage';
import { CookieConsent } from './components/CookieConsent';

import './index.css';



const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60, // 1 min
      retry: 1,
    },
  },
});

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const token = localStorage.getItem('token');
  const location = useLocation();

  if (!token) {
    return <Navigate to="/" state={{ from: location }} replace />;
  }

  return <>{children}</>;
}

function Sidebar() {
  const role = localStorage.getItem('role') || 'patient';

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('role');
    window.location.href = '/';
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <Brain size={32} color="#63b3ed" />
        <h1>EM-Predictor</h1>
      </div>

      <nav className="sidebar-nav">
        {role === 'doctor' ? (
          <>
            <NavLink to="/doctor/dashboard" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <LayoutDashboard />
              <span>Mis Pacientes</span>
            </NavLink>
            <NavLink to="/settings" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Settings />
              <span>Configuración</span>
            </NavLink>
          </>
        ) : role === 'admin' ? (
          <>
            <NavLink to="/admin" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <ShieldCheck />
              <span>Admin Panel</span>
            </NavLink>
            <NavLink to="/analytics" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <BarChart2 />
              <span>Global Stats</span>
            </NavLink>
            <NavLink to="/settings" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Settings />
              <span>Configuración</span>
            </NavLink>
          </>
        ) : (
          <>
            <NavLink to="/dashboard" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <LayoutDashboard />
              <span>Dashboard</span>
            </NavLink>

            <NavLink to="/analytics" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <BarChart2 />
              <span>Analytics</span>
            </NavLink>

            <NavLink to="/analysis" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Brain />
              <span>Modelo ML</span>
            </NavLink>

            <NavLink to="/upload" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Upload />
              <span>Subir Datos</span>
            </NavLink>

            <div style={{ margin: '1rem 0', height: '1px', background: 'rgba(255,255,255,0.05)' }}></div>

            <NavLink to="/events" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <FileText />
              <span>Eventos</span>
            </NavLink>

            <NavLink to="/clusters" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <AlertTriangle />
              <span>Clusters</span>
            </NavLink>

            <NavLink to="/calendar" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Calendar />
              <span>Calendario</span>
            </NavLink>

            <NavLink to="/settings" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Settings />
              <span>Configuración</span>
            </NavLink>
          </>
        )}
      </nav>

      <div style={{ marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
        <button onClick={handleLogout} className="nav-link" style={{ background: 'none', border: 'none', width: '100%', cursor: 'pointer' }}>
          <LogOut />
          <span>Cerrar Sesión</span>
        </button>
      </div>
    </aside>
  );
}

function App() {
  const googleClientId = import.meta.env.VITE_GOOGLE_CLIENT_ID || '123456789-mock.apps.googleusercontent.com';

  return (
    <GoogleOAuthProvider clientId={googleClientId}>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Navigate to="/" replace />} />
            <Route path="/" element={localStorage.getItem('token') ? <Navigate to="/dashboard" replace /> : <LoginPage />} />

            <Route
              path="/*"
              element={
                <ProtectedRoute>
                  <div className="app">
                    <Sidebar />
                    <main className="main-content">
                      {localStorage.getItem('role') === 'doctor' && localStorage.getItem('selected_patient_id') && (
                        <div style={{
                          background: 'var(--color-warning)',
                          color: '#1a202c',
                          padding: '0.75rem 1.5rem',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          fontWeight: '600',
                          borderRadius: '0 0 var(--radius-md) var(--radius-md)',
                          marginBottom: '1rem'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <AlertTriangle size={18} />
                            <span>Estás viendo la vista del paciente</span>
                          </div>
                          <button
                            onClick={() => {
                              localStorage.removeItem('selected_patient_id');
                              window.location.href = '/doctor/dashboard';
                            }}
                            style={{
                              background: 'rgba(0,0,0,0.1)',
                              border: 'none',
                              padding: '0.25rem 0.75rem',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              fontWeight: 'bold',
                              fontSize: '0.875rem'
                            }}
                          >
                            Salir a Mi Lista
                          </button>
                        </div>
                      )}
                      <Routes>
                        <Route path="/" element={<Navigate to="/dashboard" replace />} />
                        <Route path="/dashboard" element={<DashboardPage />} />
                        <Route path="/analytics" element={<AnalyticsPage />} />
                        <Route path="/analysis" element={<ModelAnalysisPage />} />
                        <Route path="/upload" element={<UploadPage />} />
                        <Route path="/events" element={<EventsPage />} />
                        <Route path="/clusters" element={<ClustersPage />} />
                        <Route path="/calendar" element={<CalendarPage />} />
                        <Route path="/settings" element={<SettingsPage />} />
                        <Route path="/doctor/dashboard" element={<PatientsPage />} />
                        <Route path="/admin" element={<AdminPage />} />
                      </Routes>
                    </main>
                  </div>
                </ProtectedRoute>
              }
            />
          </Routes>
        </BrowserRouter>
        <CookieConsent />
      </QueryClientProvider>
    </GoogleOAuthProvider>
  );
}


export default App;


