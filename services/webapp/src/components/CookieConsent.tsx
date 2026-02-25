import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cookie, X, Settings, ShieldCheck } from 'lucide-react';

export const CookieConsent: React.FC = () => {
    const [isVisible, setIsVisible] = useState(false);
    const [showSettings, setShowSettings] = useState(false);
    const [preferences, setPreferences] = useState({
        essential: true,
        analytics: true,
        marketing: false
    });

    useEffect(() => {
        const consent = localStorage.getItem('cookie-consent');
        if (!consent) {
            setTimeout(() => setIsVisible(true), 1500);
        }
    }, []);

    const handleAcceptAll = () => {
        const allAccepted = { essential: true, analytics: true, marketing: true };
        setPreferences(allAccepted);
        localStorage.setItem('cookie-consent', JSON.stringify(allAccepted));
        setIsVisible(false);
    };

    const handleSavePreferences = () => {
        localStorage.setItem('cookie-consent', JSON.stringify(preferences));
        setIsVisible(false);
    };

    const handleDeclineNonEssential = () => {
        const minimal = { essential: true, analytics: false, marketing: false };
        setPreferences(minimal);
        localStorage.setItem('cookie-consent', JSON.stringify(minimal));
        setIsVisible(false);
    };

    return (
        <AnimatePresence>
            {isVisible && (
                <motion.div
                    initial={{ y: 100, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    exit={{ y: 100, opacity: 0 }}
                    className="fixed bottom-4 left-4 right-4 md:left-auto md:right-6 md:max-w-md z-[90] box-border"
                >
                    <div className="glass-card p-6 border shadow-2xl relative overflow-hidden group" style={{ background: 'var(--color-bg-secondary)', borderColor: 'var(--color-border)' }}>
                        <div className="absolute top-0 left-0 w-1 h-full bg-accent"></div>

                        {!showSettings ? (
                            <div className="space-y-4">
                                <div className="flex items-start gap-4">
                                    <div className="p-3 bg-accent/10 rounded-xl" style={{ backgroundColor: 'rgba(99, 179, 237, 0.1)' }}>
                                        <Cookie size={24} style={{ color: 'var(--color-accent)' }} />
                                    </div>
                                    <div className="flex-1">
                                        <h4 className="text-lg font-bold mb-1" style={{ color: 'var(--color-text-primary)' }}>Configuración de Cookies</h4>
                                        <p className="text-sm leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                                            Utilizamos cookies propias y de terceros para mejorar su experiencia, analizar el tráfico y personalizar contenido en EM-Predictor.
                                        </p>
                                    </div>
                                </div>

                                <div className="flex flex-col gap-2 pt-2">
                                    <button
                                        onClick={handleAcceptAll}
                                        className="btn btn-primary w-full py-2.5 text-sm"
                                    >
                                        Aceptar todas
                                    </button>
                                    <div className="grid grid-cols-2 gap-2">
                                        <button
                                            onClick={() => setShowSettings(true)}
                                            className="btn btn-secondary py-2 text-xs flex items-center justify-center gap-2"
                                        >
                                            <Settings size={14} /> Configurar
                                        </button>
                                        <button
                                            onClick={handleDeclineNonEssential}
                                            className="btn btn-secondary py-2 text-xs"
                                        >
                                            Solo esenciales
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-2">
                                        <ShieldCheck size={18} style={{ color: 'var(--color-accent)' }} />
                                        <h4 className="font-bold" style={{ color: 'var(--color-text-primary)' }}>Preferencias</h4>
                                    </div>
                                    <button
                                        onClick={() => setShowSettings(false)}
                                        className="p-1 rounded-full cursor-pointer"
                                        style={{ background: 'none', border: 'none', color: 'var(--color-text-secondary)' }}
                                    >
                                        <X size={18} />
                                    </button>
                                </div>

                                <div className="space-y-3">
                                    <div className="flex items-center justify-between p-3 rounded-lg border" style={{ backgroundColor: 'rgba(255,255,255,0.02)', borderColor: 'var(--color-border)' }}>
                                        <div>
                                            <p className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>Esenciales</p>
                                            <p className="text-[10px]" style={{ color: 'var(--color-text-secondary)' }}>Necesarias para el funcionamiento (auth).</p>
                                        </div>
                                        <div className="w-10 h-5 rounded-full relative" style={{ backgroundColor: 'var(--color-accent)', opacity: 0.5 }}>
                                            <div className="absolute right-1 top-1 w-3 h-3 bg-white rounded-full"></div>
                                        </div>
                                    </div>

                                    <div
                                        className="flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors"
                                        style={{ backgroundColor: 'rgba(255,255,255,0.02)', borderColor: 'var(--color-border)' }}
                                        onClick={() => setPreferences(prev => ({ ...prev, analytics: !prev.analytics }))}
                                    >
                                        <div>
                                            <p className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>Analíticas</p>
                                            <p className="text-[10px]" style={{ color: 'var(--color-text-secondary)' }}>Para entender cómo usas EM-Predictor.</p>
                                        </div>
                                        <div className={`w-10 h-5 rounded-full relative transition-colors`} style={{ backgroundColor: preferences.analytics ? 'var(--color-accent)' : '#4a5568' }}>
                                            <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${preferences.analytics ? 'right-1' : 'left-1'}`}></div>
                                        </div>
                                    </div>

                                    <div
                                        className="flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors"
                                        style={{ backgroundColor: 'rgba(255,255,255,0.02)', borderColor: 'var(--color-border)' }}
                                        onClick={() => setPreferences(prev => ({ ...prev, marketing: !prev.marketing }))}
                                    >
                                        <div>
                                            <p className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>Mejoras IA</p>
                                            <p className="text-[10px]" style={{ color: 'var(--color-text-secondary)' }}>Para entrenar modelos con telemetría anónima.</p>
                                        </div>
                                        <div className={`w-10 h-5 rounded-full relative transition-colors`} style={{ backgroundColor: preferences.marketing ? 'var(--color-accent)' : '#4a5568' }}>
                                            <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${preferences.marketing ? 'right-1' : 'left-1'}`}></div>
                                        </div>
                                    </div>
                                </div>

                                <button
                                    onClick={handleSavePreferences}
                                    className="btn btn-primary w-full py-2.5 text-sm mt-2"
                                >
                                    Guardar configuración
                                </button>
                            </div>
                        )}
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};
