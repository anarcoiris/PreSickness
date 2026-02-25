import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Shield } from 'lucide-react';

interface PrivacyPolicyProps {
    isOpen: boolean;
    onClose: () => void;
}

export const PrivacyPolicy: React.FC<PrivacyPolicyProps> = ({ isOpen, onClose }) => {
    return (
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                    />

                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        className="relative w-full max-w-2xl max-h-[80vh] overflow-hidden border rounded-2xl shadow-2xl flex flex-col"
                        style={{ backgroundColor: 'var(--color-bg-primary)', borderColor: 'var(--color-border)' }}
                    >
                        <div className="p-6 border-b flex justify-between items-center" style={{ backgroundColor: 'rgba(255,255,255,0.02)', borderColor: 'var(--color-border)' }}>
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-accent/10 rounded-lg" style={{ backgroundColor: 'rgba(99, 179, 237, 0.1)' }}>
                                    <Shield size={20} style={{ color: 'var(--color-accent)' }} />
                                </div>
                                <h2 className="text-xl font-bold" style={{ color: 'var(--color-text-primary)' }}>Política de Privacidad y Tratamiento de Datos</h2>
                            </div>
                            <button
                                onClick={onClose}
                                className="p-2 hover:bg-white/10 rounded-full transition-colors cursor-pointer"
                                style={{ background: 'none', border: 'none', color: 'var(--color-text-secondary)' }}
                            >
                                <X size={20} />
                            </button>
                        </div>

                        <div className="p-8 overflow-y-auto space-y-6" style={{ color: 'var(--color-text-secondary)' }}>
                            <section>
                                <h3 className="font-bold mb-3" style={{ color: 'var(--color-text-primary)' }}>1. Responsabilidad y Finalidad del Tratamiento</h3>
                                <p>
                                    Sistema <strong>EM-Predictor</strong> actúa como responsable del tratamiento de los datos clínicos, biométricos y de comportamiento del paciente.
                                    Para ejercer sus derechos RGPD, puede contactarnos a través del panel de su médico o en <span style={{ color: 'var(--color-accent)' }}>privacy@prebrote.ddns.net</span>.
                                </p>
                            </section>

                            <section>
                                <h3 className="font-bold mb-3" style={{ color: 'var(--color-text-primary)' }}>2. Datos que recopilamos</h3>
                                <p>A través de nuestros canales de mensajería (WhatsApp/Telegram), interfaz web y dispositivos IoT recopilamos:</p>
                                <ul className="list-disc ml-5 mt-2 space-y-1">
                                    <li>Datos personales básicos (Nombre, correo electrónico, IDs de dispositivos)</li>
                                    <li>Textos, audios y mensajes destinados a inferencia semántica y análisis del estado de salud mental y cognitivo.</li>
                                    <li>Registros de telemetría provenientes de dispositivos vestibles (movimiento, ritmo cardíaco).</li>
                                    <li>Eventos clínicos anotados por el usuario o su cuerpo médico.</li>
                                </ul>
                            </section>

                            <section>
                                <h3 className="font-bold mb-3" style={{ color: 'var(--color-text-primary)' }}>3. Uso de la Inteligencia Artificial</h3>
                                <p>Los datos procesados se introducen en canales de Machine Learning (Transformers, NLP y Time Series) con dos fines:</p>
                                <ul className="list-disc ml-5 mt-2 space-y-1">
                                    <li>Predecir y alertar de manera temprana sobre posibles brotes (esclerosis, fatiga, etc).</li>
                                    <li>Re-entrenar modelos predictivos (si confió la opción en la configuración de Cookies), asegurando la total anonimización de la cuenta de usuario antes del entrenamiento.</li>
                                </ul>
                            </section>

                            <section>
                                <h3 className="font-bold mb-3" style={{ color: 'var(--color-text-primary)' }}>4. Base legal</h3>
                                <p>
                                    El tratamiento de datos de salud requiere su consentimiento explícito como principal base jurídica de este tratamiento continuo,
                                    ya que implica decisiones automatizadas de IA relativas a variables biomédicas.
                                </p>
                            </section>

                            <section>
                                <h3 className="font-bold mb-3" style={{ color: 'var(--color-text-primary)' }}>5. Seguridad y Aislamiento (On-Premise)</h3>
                                <p>
                                    El procesamiento LLM de textos y audios se realiza de forma local y <em>offline</em>, donde EM-Predictor no expone
                                    la información sensible a servicios cognitivos externos no auditados por nuestro ecosistema (On-Premise).
                                    Se implementan cifrados HMAC y JWT para las conexiones API de clientes o ingestiones Kafka.
                                </p>
                            </section>

                            <section>
                                <div className="mt-4 p-4 border rounded-md" style={{ borderColor: 'var(--color-border)', backgroundColor: 'rgba(99,179,237,0.05)' }}>
                                    Recordamos que las inferencias y riesgos provistos por EM-Predictor son asistencia clínica y <strong>no reemplazan</strong> el diagnóstico médico cualificado.
                                </div>
                            </section>
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
    );
};
