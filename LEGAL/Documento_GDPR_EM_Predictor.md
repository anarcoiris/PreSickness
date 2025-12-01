# 📄 Documento de cumplimiento GDPR / Privacidad — Proyecto EM-Predictor

## 1. Data Protection Impact Assessment (DPIA) — borrador

### 1.1. Información general

**Título del proyecto:** EM-Predictor — Sistema de predicción de brotes de Esclerosis Múltiple mediante ML sobre datos lingüísticos / clínicos anonimizados.  
**Responsable del tratamiento:** [Nombre de tu entidad o empresa]  
**Encargado(s) del tratamiento:** Backend, equipo ML, infraestructura de datos, almacenamiento, servicios de alerta.  
**Finalidad del tratamiento:** Desarrollo e investigación de un sistema predictivo de salud (brotes de EM), monitoreo / seguimiento longitudinal de pacientes, generación de alertas clínicas, investigación médica, análisis estadístico agregado.  
**Base jurídica:** Consentimiento explícito e informado de los pacientes (interesados), conforme al Reglamento (UE) 2016/679 (GDPR) + Ley Orgánica de Protección de Datos y Garantía de Derechos Digitales (LOPDGDD).

### 1.2. Descripción del tratamiento de datos

**Recogida de datos:** texto libre (diarios, autoevaluaciones, cuestionarios), posiblemente audio, metadatos (timestamp, idioma, contexto, metadatos clínicos), datos demográficos.

**Procesamiento / Transformaciones:** extracción de características lingüísticas mediante embeddings / NLP, cálculo de features temporales (ventanas: 1d, 3d, 7d, 14d, 30d), agregaciones, normalización, anonimización / pseudonimización de IDs (hash irreversibles + salado), cifrado de datos sensibles.

**Almacenamiento:** base de datos (TimescaleDB/PostgreSQL + Redis + almacenamiento S3-compatible con cifrado en reposo), logs, historial de features, modelos entrenados, predicciones, alertas.

**Acceso:** solo personal autorizado (roles limitados), registro de accesos, acceso mediante credenciales seguras, TLS + cifrado en tránsito.

**Conservación / Retención:** definir periodo de retención tras final del estudio o tras revocación del consentimiento — por ejemplo: datos anonimizados indefinidamente para análisis agregado; datos identificables sólo mientras dure el estudio + 5 años de retención para trazabilidad.

**Transferencias / Diseminación:** en principio sin difusión pública de datos identificables; resultados agregados o anonimizados compartibles con partners clínicos / publicaciones.

### 1.3. Evaluación de riesgos y medidas de mitigación

| Riesgo identificado | Consecuencias potenciales | Medidas de mitigación / minimización |
|--------------------|---------------------------|--------------------------------------|
| Reidentificación de sujetos (texto, metadatos) | Pérdida de anonimato, vulneración de privacidad | Pseudonimización irreversible; no almacenar texto plano; sólo embeddings / features cifradas; hashing de IDs; separación de datos identificativos y clínicos; control de acceso. |
| Fuga de datos en reposo o en tránsito | Exposición de datos sensibles | Cifrado at-rest (AES-256), cifrado in-transit (TLS 1.3), uso de infraestructura segura, encriptación de backups, uso de Vault/KMS para secretos. |
| Acceso no autorizado / interno | Mala praxis, negligencia, fuga de datos | Políticas de roles mínimos (least privilege), registro de accesos, auditorías periódicas, controles de acceso, logs inmutables. |
| Uso indebido de datos para fines distintos | Violación del consentimiento, sanciones legales | Consentimiento claro y explícito; limitación del uso a fines definidos; registro de finalidades; prohibición de reuso sin nuevo consentimiento. |
| Incidente / Brecha de seguridad | Daño reputacional, sanciones, pérdida de confianza | Plan de respuesta a incidentes, notificación rápida a la AEPD, mitigación, cifrado, backups cifrados, auditorías de seguridad. |

### 1.4. Conclusión preliminar & recomendaciones

El tratamiento de datos planificado es de **alto riesgo** (datos sensibles de salud, perfilado ML, monitoreo longitudinal).  
La DPIA es **obligatoria** bajo GDPR. Las medidas propuestas (pseudonimización, cifrado, separación de roles, auditoría, consentimiento explícito) **mitigan el riesgo significativamente**.

**Recomendaciones adicionales:**  
- Designar un DPO externo/interno y registrarlo ante la AEPD si aplica.  
- Documentar flujos de datos completos (data flow diagrams).  
- Establecer políticas de borrado / anonimización final al terminar el estudio.  
- Definir procedimientos de auditoría, logging y respuesta a incidentes.

---

## 2. Consentimiento Informado — borrador (versión paciente/usuario)

**Título:** Documento de consentimiento informado para participación en estudio EM-Predictor  
**Responsable:** [Nombre de la organización / hospital / entidad promotora]

Estimado/a participante,

Le solicitamos su participación en el estudio de investigación denominado **“EM-Predictor: predicción de brotes de esclerosis múltiple mediante modelado de lenguaje y datos clínicos”**. Antes de aceptar, por favor lea con atención la siguiente información.

### 2.1. Qué datos se recopilan

- Texto libre (diarios personales, autoevaluaciones, cuestionarios, posible grabación de audio)  
- Metadatos (fecha, hora, idioma, contexto, posibles datos demográficos o clínicos)  
- Datos clínicos relacionados con su diagnóstico de EM y su historial médico.

### 2.2. Finalidad del tratamiento

- Desarrollar un modelo de predicción de brotes de EM con antelación.  
- Monitoreo longitudinal del progreso de la enfermedad.  
- Investigación clínica y análisis agregados.  
- Generación de alertas clínicas para neurólogos / equipo médico.  
- Publicación de resultados científicos en formato anonimizado y agregado.

### 2.3. Cómo se protegen sus datos

- Sus datos serán **pseudonimizados**.  
- No se almacenará texto plano: solo embeddings, features numéricas o datos anonimizados.  
- Cifrado en reposo y en tránsito (TLS + cifrado en base de datos).  
- Acceso restringido únicamente al personal autorizado.  
- Puede revocar su consentimiento en cualquier momento.

### 2.4. Derechos del participante

Derechos reconocidos por RGPD/LOPDGDD: acceso, rectificación, supresión, limitación, oposición, portabilidad, retirada del consentimiento, derecho al olvido.

### 2.5. Voluntariedad y retirada

La participación es completamente voluntaria. Puede retirarse en cualquier momento sin repercusiones negativas. Si decide revocar su consentimiento, sus datos serán eliminados o anonimizados.

### 2.6. Contacto y responsable del estudio

- **Responsable:** [Nombre / entidad / hospital / empresa]  
- **Contacto DPO:** [Nombre / email / teléfono]

### Consentimiento explícito
☐ He leído y comprendido la información proporcionada.  
☐ Acepto voluntariamente participar en este estudio bajo los términos descritos.  
☐ Consiento el tratamiento de mis datos conforme a lo expuesto.  
☐ Entiendo que puedo revocar mi consentimiento en cualquier momento.  

Fecha: ________     Firma: ___________________

---

## 3. Checklist de cumplimiento GDPR / “Privacy-by-Design”

- [ ] Realizar DPIA antes del inicio del tratamiento  
- [ ] Designar DPO (interno o externo)  
- [ ] Documentar flujos de datos  
- [ ] Implementar pseudonimización / hashing irreversible de IDs  
- [ ] No almacenar texto plano  
- [ ] Cifrado at-rest (AES-256) + in-transit (TLS 1.3)  
- [ ] Control de accesos / roles, least privilege, logs de auditoría  
- [ ] Procedimiento para revocación de consentimiento y borrado/anónimización  
- [ ] Retención mínima y justificada  
- [ ] Políticas de backup, recuperación y plan de respuesta a incidentes  
- [ ] Transparencia con participantes  
- [ ] Consentimiento informado claro y documentado  
- [ ] Contratos con proveedores conforme GDPR/LOPDGDD  
- [ ] Registro de actividades de tratamiento (RAT)

---

## 4. Posibles consultoras / proveedores DPO en España

| Consultora | Servicios | Ubicación / Contacto |
|------------|-----------|----------------------|
| **DataPro Legal** | RGPD, auditoría, DPO externo | https://dataprolegal.com — Madrid |
| **DATAX** | DPO externo certificado, EIPD | https://datax.es — Barcelona (93 754 06 88) |
| **Auratech Legal Solutions** | DPO, auditorías, asesoría RGPD | https://auratechlegal.es — Madrid |
| **PrivaLex Partners** | DPO externo, supervisión continua | https://privalex.es — Barcelona / online |
| **LegalDPO** | RGPD integral y defensa jurídica | https://legaldpo.es — España |

⚠️ Recomiendo contactar al menos 2–3 para solicitar presupuesto y disponibilidad.

