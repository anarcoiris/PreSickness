import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGoogleLogin } from '@react-oauth/google';
import { authApi } from '../api/client';
import { Brain, Lock, Mail, AlertCircle, ChevronRight, Activity } from 'lucide-react';

export default function LoginPage() {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [role, setRole] = useState<'patient' | 'doctor'>('patient'); // Added role state
    const [isRegister, setIsRegister] = useState(false);
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();

    const handleGoogleLogin = useGoogleLogin({
        onSuccess: async (codeResponse) => {
            setError('');
            setIsLoading(true);
            try {
                // Send Authorization Code to Backend
                const res = await authApi.loginWithGoogle(codeResponse.code, role);
                localStorage.setItem('token', res.data.access_token);

                // Check profile to redirect correctly
                const profile = await authApi.getProfile();
                localStorage.setItem('role', profile.data.role); // Store role

                if (profile.data.role === 'doctor') {
                    navigate('/doctor/dashboard');
                } else if (profile.data.role === 'admin') {
                    navigate('/admin');
                } else {
                    navigate('/dashboard');
                }
            } catch (err: any) {
                console.error("Google Login Error", err);
                setError('Error en la autenticación con Google. Verifica la consola para más detalles.');
            } finally {
                setIsLoading(false);
            }
        },
        onError: () => {
            setError('Fallo en el inicio de sesión con Google.');
        },
        flow: 'auth-code',
        scope: "https://www.googleapis.com/auth/calendar https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email"
    });

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            if (isRegister) {
                await authApi.register(email, password, name, role); // Pass role
                // After register, login
                const loginRes = await authApi.login(email, password);
                localStorage.setItem('token', loginRes.data.access_token);
            } else {
                const res = await authApi.login(email, password);
                localStorage.setItem('token', res.data.access_token);
            }

            // Check profile to redirect correctly
            const profile = await authApi.getProfile();
            localStorage.setItem('role', profile.data.role); // Store role

            if (profile.data.role === 'doctor') {
                navigate('/doctor/dashboard');
            } else if (profile.data.role === 'admin') {
                navigate('/admin');
            } else {
                navigate('/dashboard');
            }
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Error en la operación. Por favor, inténtalo de nuevo.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="login-container">
            <div className="login-visual">
                <div className="visual-content">
                    <div className="visual-logo">
                        <Activity size={48} color="var(--color-accent-light)" />
                    </div>
                    <h1>Bienvenido a EM-Predictor</h1>
                    <p>
                        Inteligencia Artificial avanzada para la monitorización
                        y predicción proactiva de brotes en Esclerosis Múltiple.
                    </p>

                    <div className="feature-badges">
                        <div className="badge-item">
                            <div className="badge-dot" style={{ background: 'var(--color-success)' }}></div>
                            <span>Predicción TFT</span>
                        </div>
                        <div className="badge-item">
                            <div className="badge-dot" style={{ background: 'var(--color-warning)' }}></div>
                            <span>Análisis NLP</span>
                        </div>
                        <div className="badge-item">
                            <div className="badge-dot" style={{ background: 'var(--color-info)' }}></div>
                            <span>Wearables Sync</span>
                        </div>
                    </div>
                </div>
                <div className="visual-overlay"></div>
            </div>

            <div className="login-form-side">
                <div className="form-wrapper fade-in">
                    <div className="form-header">
                        <div className="mobile-logo">
                            <Brain size={32} color="var(--color-accent-light)" />
                        </div>
                        <h2>{isRegister ? 'Crear Cuenta' : 'Iniciar Sesión'}</h2>
                        <p>{isRegister ? 'Completa tus datos para registrarte' : 'Introduce tus credenciales para acceder al sistema'}</p>
                    </div>

                    {error && (
                        <div className="auth-error">
                            <AlertCircle size={18} />
                            <span>{error}</span>
                        </div>
                    )}

                    <form onSubmit={handleSubmit}>
                        {isRegister && (
                            <>
                                <div className="form-group role-selector">
                                    <label className="form-label">Soy:</label>
                                    <div className="role-options">
                                        <button
                                            type="button"
                                            className={`role-btn ${role === 'patient' ? 'active' : ''}`}
                                            onClick={() => setRole('patient')}
                                        >
                                            Paciente
                                        </button>
                                        <button
                                            type="button"
                                            className={`role-btn ${role === 'doctor' ? 'active' : ''}`}
                                            onClick={() => setRole('doctor')}
                                        >
                                            Médico
                                        </button>
                                    </div>
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Nombre Completo</label>
                                    <div className="input-with-icon">
                                        <Activity size={18} className="input-icon" />
                                        <input
                                            type="text"
                                            className="form-input"
                                            placeholder="Tu nombre"
                                            value={name}
                                            onChange={(e) => setName(e.target.value)}
                                            required={isRegister}
                                        />
                                    </div>
                                </div>
                            </>
                        )}

                        <div className="form-group">
                            <label className="form-label">Correo Electrónico</label>
                            <div className="input-with-icon">
                                <Mail size={18} className="input-icon" />
                                <input
                                    type="email"
                                    className="form-input"
                                    placeholder="ejemplo@medico.com"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    required
                                />
                            </div>
                        </div>

                        <div className="form-group">
                            <label className="form-label">Contraseña</label>
                            <div className="input-with-icon">
                                <Lock size={18} className="input-icon" />
                                <input
                                    type="password"
                                    className="form-input"
                                    placeholder="••••••••"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                />
                            </div>
                        </div>

                        {!isRegister && (
                            <div className="form-footer-actions">
                                <label className="checkbox-container">
                                    <input type="checkbox" />
                                    <span className="checkmark"></span>
                                    Recordar sesión
                                </label>
                                <a href="#" className="forgot-password">¿Olvidaste tu contraseña?</a>
                            </div>
                        )}

                        <button
                            type="submit"
                            className="btn btn-primary login-btn"
                            disabled={isLoading}
                        >
                            {isLoading ? (
                                <div className="spinner-small"></div>
                            ) : (
                                <>
                                    {isRegister ? 'Registrarse' : 'Acceder al Sistema'}
                                    <ChevronRight size={18} />
                                </>
                            )}
                        </button>
                    </form>

                    <div className="form-divider">
                        <span>O accede con</span>
                    </div>

                    <div className="social-login">
                        <button
                            className="btn btn-secondary"
                            onClick={() => handleGoogleLogin()}
                            disabled={isLoading}
                        >
                            <img src="https://www.svgrepo.com/show/355037/google.svg" alt="Google" width="20" />
                            Google
                        </button>
                    </div>

                    <p className="signup-prompt">
                        {isRegister ? '¿Ya tienes una cuenta?' : '¿No tienes una cuenta?'}
                        <a href="#" onClick={(e) => { e.preventDefault(); setIsRegister(!isRegister); }}>
                            {isRegister ? ' Inicia Sesión' : ' Regístrate aquí'}
                        </a>
                    </p>
                </div>
            </div>

            <style>{`
                .login-container {
                    display: flex;
                    min-height: 100vh;
                    background: var(--color-bg-primary);
                }

                .login-visual {
                    flex: 1.2;
                    background: url('https://images.unsplash.com/photo-1576091160550-21735994b1a3?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80');
                    background-size: cover;
                    background-position: center;
                    position: relative;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 4rem;
                }

                @media (max-width: 1024px) {
                    .login-visual { display: none; }
                }

                .visual-overlay {
                    position: absolute;
                    inset: 0;
                    background: linear-gradient(135deg, rgba(10, 22, 40, 0.95) 0%, rgba(26, 39, 68, 0.8) 100%);
                    z-index: 1;
                }

                .visual-content {
                    position: relative;
                    z-index: 2;
                    max-width: 500px;
                    color: white;
                }

                .visual-content h1 {
                    font-size: 3rem;
                    font-weight: 800;
                    margin: 1.5rem 0;
                    line-height: 1.1;
                }

                .visual-content p {
                    font-size: 1.125rem;
                    color: var(--color-text-secondary);
                    line-height: 1.6;
                    margin-bottom: 2.5rem;
                }

                .feature-badges {
                    display: flex;
                    gap: 1.5rem;
                }

                .badge-item {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    background: rgba(255, 255, 255, 0.05);
                    padding: 0.5rem 1rem;
                    border-radius: var(--radius-full);
                    font-size: 0.875rem;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }

                .badge-dot {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                }

                .login-form-side {
                    flex: 1;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 2rem;
                    background: var(--color-bg-primary);
                }

                .form-wrapper {
                    width: 100%;
                    max-width: 420px;
                }

                .form-header { margin-bottom: 2.5rem; }
                .form-header h2 { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem; }
                .form-header p { color: var(--color-text-secondary); }

                .mobile-logo { margin-bottom: 1.5rem; display: none; }
                @media (max-width: 1024px) { .mobile-logo { display: block; } }

                .input-with-icon { position: relative; }
                .input-icon {
                    position: absolute;
                    left: 1rem;
                    top: 50%;
                    transform: translateY(-50%);
                    color: var(--color-text-muted);
                }

                .input-with-icon .form-input { padding-left: 3rem; }

                .form-footer-actions {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 2rem;
                    font-size: 0.875rem;
                }

                .forgot-password { color: var(--color-accent-light); text-decoration: none; }
                .forgot-password:hover { text-decoration: underline; }

                .login-btn {
                    width: 100%;
                    padding: 1rem;
                    font-size: 1rem;
                    gap: 0.75rem;
                    margin-bottom: 2rem;
                }

                .form-divider {
                    position: relative;
                    text-align: center;
                    margin-bottom: 2rem;
                }

                .form-divider::before {
                    content: '';
                    position: absolute;
                    left: 0;
                    top: 50%;
                    width: 100%;
                    height: 1px;
                    background: rgba(255, 255, 255, 0.1);
                }

                .form-divider span {
                    position: relative;
                    background: var(--color-bg-primary);
                    padding: 0 1rem;
                    color: var(--color-text-muted);
                    font-size: 0.875rem;
                }

                .social-login button { width: 100%; gap: 1rem; font-weight: 600; }

                .signup-prompt {
                    margin-top: 2rem;
                    text-align: center;
                    color: var(--color-text-secondary);
                    font-size: 0.875rem;
                }

                .signup-prompt a { color: var(--color-accent-light); text-decoration: none; font-weight: 600; }

                .auth-error {
                    background: rgba(245, 101, 101, 0.1);
                    border: 1px solid var(--color-danger);
                    color: var(--color-danger);
                    padding: 0.75rem 1rem;
                    border-radius: var(--radius-md);
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    margin-bottom: 1.5rem;
                    font-size: 0.875rem;
                }

                .spinner-small {
                    width: 20px;
                    height: 20px;
                    border: 2px solid rgba(255,255,255,0.3);
                    border-top-color: white;
                    border-radius: 50%;
                    animation: spin 0.8s linear infinite;
                }

                /* Custom Checkbox */
                .checkbox-container {
                    display: block;
                    position: relative;
                    padding-left: 25px;
                    cursor: pointer;
                    user-select: none;
                }

                .checkbox-container input { position: absolute; opacity: 0; cursor: pointer; height: 0; width: 0; }
                .checkmark {
                    position: absolute;
                    top: 0;
                    left: 0;
                    height: 18px;
                    width: 18px;
                    background-color: rgba(255,255,255,0.05);
                    border: 1px solid rgba(255,255,255,0.2);
                    border-radius: 4px;
                }

                .checkbox-container:hover input ~ .checkmark { background-color: rgba(255,255,255,0.1); }
                .checkbox-container input:checked ~ .checkmark { background-color: var(--color-accent); border-color: var(--color-accent); }
                .checkmark:after {
                    content: "";
                    position: absolute;
                    display: none;
                    left: 6px;
                    top: 2px;
                    width: 4px;
                    height: 9px;
                    border: solid white;
                    border-width: 0 2px 2px 0;
                    transform: rotate(45deg);
                }
                .checkbox-container input:checked ~ .checkmark:after { display: block; }

                /* Role Selector */
                .role-selector { margin-bottom: 1.5rem; }
                .role-options {
                    display: flex;
                    gap: 1rem;
                    margin-top: 0.5rem;
                }
                .role-btn {
                    flex: 1;
                    padding: 0.75rem;
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: var(--radius-md);
                    color: var(--color-text-secondary);
                    cursor: pointer;
                    transition: all 0.2s;
                    font-weight: 500;
                }
                .role-btn:hover {
                    background: rgba(255, 255, 255, 0.1);
                    color: white;
                }
                .role-btn.active {
                    background: var(--color-accent);
                    color: white;
                    border-color: var(--color-accent);
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                }

            `}</style>
        </div>
    );
}
