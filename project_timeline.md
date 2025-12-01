# Plan Ejecutivo de Desarrollo - EM Predictor Prototype
## Timeline, Presupuesto y Acciones Críticas

---

## 📊 RESUMEN EJECUTIVO

**Objetivo:** Validar hipótesis clínica de que señales lingüísticas y contextuales pueden anticipar brotes de EM con 7-30 días de antelación.

**Alcance MVP:**
- Backend completo con ML pipeline (Python/PyTorch)
- Cliente Android simplificado (generador sintético)
- 50-150 pacientes piloto durante 6 meses
- Infraestructura cloud-agnóstica (Docker/K8s)
- Cumplimiento GDPR/HIPAA desde diseño

**Duración:** 16 semanas (4 meses) hasta piloto controlado
**Presupuesto estimado:** €90.000 - €150.000
**Equipo mínimo:** 6-8 personas

---

## 🗓️ TIMELINE DETALLADO

### FASE 0: Preparación Legal y Técnica (Semanas 1-4)
**Responsables Principales:** Agent Legal (Lex), Agent Architect (Archie)


#### Semana 1-2: Marco Legal
**Lead:** Agent Legal (Lex) | **Support:** Agent PM (Chief)

- [ ] Contratar DPO o consultor legal especializado en health tech
- [ ] Elaborar DPIA (Data Protection Impact Assessment)
- [ ] Diseñar plantillas de consentimiento informado
- [ ] Contactar 2-3 centros neurológicos para partnership
- [ ] Definir protocolo de investigación con neurólogos

**Entregables:**
- ✅ DPIA completo
- ✅ Consentimiento informado (v1.0)
- ✅ Protocolo clínico (endpoint: EDSS change ≥1.0 o evento clínico confirmado)
- ✅ MoU con centro neurológico partner

**Riesgos:**
- ⚠️ Retraso en aprobación ética (mitigación: iniciar en paralelo con desarrollo backend)
- ⚠️ Falta de partner clínico (mitigación: buscar 3+ opciones)

#### Semana 3-4: Arquitectura Técnica
**Lead:** Agent Architect (Archie) | **Support:** Agent Backend (Backus)

- [ ] Finalizar decisiones de stack tecnológico
- [ ] Diseñar schema de base de datos
- [ ] Configurar repositorios Git (monorepo o multi-repo)
- [ ] Setup infra base: servidores dev/staging
- [ ] Documentar API contracts (OpenAPI spec)

**Entregables:**
- ✅ ADR (Architecture Decision Records)
- ✅ Schema DB (v1.0)
- ✅ API spec (OpenAPI 3.0)
- ✅ Infra as Code (Terraform/Pulumi)

---

### FASE 1: Backend Core (Semanas 5-10)
**Responsables Principales:** Agent Backend (Backus), Agent Architect (Archie), Agent ML (Brain)


#### Semana 5-6: Infraestructura Base
**Lead:** Agent Architect (Archie) | **Support:** Agent Backend (Backus)

**Objetivo:** Sistema de ingesta segura funcional

**Tareas:**
- [ ] Implementar API Gateway (FastAPI) con autenticación
- [ ] Setup PostgreSQL + TimescaleDB + Redis
- [ ] Implementar crypto utilities (Fernet + HMAC)
- [ ] Configurar MinIO para almacenamiento
- [ ] Setup Kafka/Redpanda para event streaming
- [ ] Implementar health checks y logging

**Entregables:**
- ✅ API Gateway deployable
- ✅ DB schema aplicado
- ✅ Docker Compose funcional
- ✅ Tests unitarios (>70% coverage)

**Hitos de validación:**
- ✅ Ingesta de 1000 datapoints sintéticos/min
- ✅ Latencia p99 < 500ms
- ✅ Cifrado end-to-end verificado

#### Semana 7-8: Feature Extraction
**Lead:** Agent Backend (Backus) | **Support:** Agent ML (Brain)

**Objetivo:** Pipeline NLP y feature engineering

**Tareas:**
- [ ] Implementar extractor de features lingüísticas
- [ ] Integrar modelos de embeddings (sentence-transformers)
- [ ] Implementar cálculo de ventanas temporales (1d, 3d, 7d, 14d, 30d)
- [ ] Setup feature store (Redis + Postgres)
- [ ] Worker asíncrono para procesamiento batch
- [ ] Tests con datasets sintéticos

**Entregables:**
- ✅ Feature extraction service
- ✅ Feature store funcional
- ✅ Pipeline de agregación temporal
- ✅ Documentación de features

**Validación:**
- ✅ Procesar 10k datapoints en <5 min
- ✅ Features almacenados con <1s latencia

#### Semana 9-10: ML Training Pipeline
**Lead:** Agent ML (Brain) | **Support:** Agent Backend (Backus)

**Objetivo:** Entrenar primer modelo TFT

**Tareas:**
- [ ] Implementar dataset builder (features → labels)
- [ ] Configurar TFT (pytorch-forecasting)
- [ ] Setup MLflow para tracking
- [ ] Implementar time-series cross-validation
- [ ] Entrenar modelo baseline (LSTM/Prophet)
- [ ] Entrenar TFT en datos sintéticos
- [ ] Evaluar métricas (AUROC, AUPRC, calibration)

**Entregables:**
- ✅ Training pipeline completo
- ✅ Modelo TFT entrenado (v0.1)
- ✅ Experimentos en MLflow
- ✅ Notebook de análisis

**Targets:**
- 🎯 AUROC > 0.65 en validación (datos sintéticos)
- 🎯 Calibration Brier score < 0.25

---

### FASE 2: Serving e Inferencia (Semanas 11-12)
**Responsables Principales:** Agent ML (Brain), Agent Architect (Archie)


#### Semana 11-12: ML Inference Service
**Lead:** Agent ML (Brain) | **Support:** Agent Architect (Archie)

**Objetivo:** Servir predicciones en tiempo real

**Tareas:**
- [ ] Implementar inference service (ONNX Runtime)
- [ ] Cargar modelo desde MLflow
- [ ] API para predicciones on-demand
- [ ] Batch prediction job (diario)
- [ ] Almacenar predicciones en DB
- [ ] Alert manager básico (umbral simple)
- [ ] Tests de carga

**Entregables:**
- ✅ Inference API (gRPC o REST)
- ✅ Alert service
- ✅ Batch prediction pipeline
- ✅ Load tests (500 req/s)

**Targets:**
- 🎯 Latencia inferencia < 100ms (p95)
- 🎯 Throughput > 1000 predictions/min

---

### FASE 3: Cliente y Dashboard (Semanas 13-14)
**Responsables Principales:** Agent Frontend (Droid), Agent PM (Chief)


#### Semana 13: Cliente Android Mock
**Lead:** Agent Frontend (Droid) | **Support:** Agent Backend (Backus)

**Objetivo:** App Android para generar datos sintéticos

**Tareas:**
- [ ] Implementar UI básico (Jetpack Compose)
- [ ] Generador de datos sintéticos
- [ ] Conexión con API backend
- [ ] WorkManager para envío periódico
- [ ] Encriptación local
- [ ] Tests de integración

**Entregables:**
- ✅ APK funcional
- ✅ Generador sintético
- ✅ Docs de usuario

#### Semana 14: Dashboard Clínico
**Lead:** Agent Frontend (Droid) | **Support:** Agent PM (Chief)

**Objetivo:** UI para médicos

**Tareas:**
- [ ] Dashboard con Streamlit o React
- [ ] Visualización de risk scores
- [ ] Timeline de predicciones
- [ ] Alertas pendientes
- [ ] Export de reportes (PDF)
- [ ] Sistema de roles (RBAC)

**Entregables:**
- ✅ Dashboard web
- ✅ Sistema de autenticación
- ✅ Docs para clínicos

---

### FASE 4: Integración y Testing (Semanas 15-16)
**Responsables Principales:** Agent QA (Guard), Agent PM (Chief)


#### Semana 15: Testing End-to-End
**Lead:** Agent QA (Guard) | **Support:** All Agents

**Tareas:**
- [ ] Tests de integración completos
- [ ] Load testing (JMeter/Locust)
- [ ] Security audit (OWASP Top 10)
- [ ] Penetration testing básico
- [ ] Performance profiling
- [ ] Documentación final

#### Semana 16: Pre-Piloto
**Lead:** Agent PM (Chief) | **Support:** Agent QA (Guard), Agent Legal (Lex)

**Tareas:**
- [ ] Deploy en entorno staging
- [ ] Alpha test con 5-10 usuarios sintéticos
- [ ] Validación con equipo clínico
- [ ] Ajustes finales
- [ ] Preparación protocolo piloto
- [ ] Training para clínicos

**Entregables:**
- ✅ Sistema deployado en staging
- ✅ Informe de testing
- ✅ Protocolo piloto aprobado
- ✅ Go/No-Go para piloto real

---

## 💰 PRESUPUESTO ESTIMADO (16 semanas)

### Personal (€70k-€120k)
| Rol | FTE | Duración | Costo |
|-----|-----|----------|-------|
| Product/Clinical Lead | 0.5 | 4 meses | €20k-€30k |
| Data Scientist/ML Eng | 1.0 | 4 meses | €20k-€35k |
| Backend Engineer | 1.0 | 4 meses | €18k-€30k |
| Android Engineer | 0.5 | 2 meses | €8k-€12k |
| DevOps Engineer | 0.5 | 4 meses | €10k-€15k |
| Legal/DPO Consultant | 0.2 | 2 meses | €4k-€8k |

### Infraestructura (€8k-€15k)
- Cloud hosting (AWS/GCP/Azure): €2k-€5k
- Dev/staging environments: €1k-€2k
- MLflow + storage: €1k-€2k
- Monitoring tools: €500-€1k
- Licenses (GitHub, tools): €500-€1k
- Domain, SSL certs: €200
- Contingency (20%): €2k-€4k

### Legal y Compliance (€5k-€10k)
- Legal consultation: €3k-€5k
- DPO services: €2k-€3k
- Insurance (liability): €500-€2k

### Clínico (€5k-€10k)
- Partner hospital fees: €2k-€5k
- IRB/Ethics committee: €1k-€2k
- Clinical consultation: €2k-€3k

### Contingencia (10%): €8k-€15k

**Total: €90k-€150k**

---

## ⚠️ RIESGOS CRÍTICOS Y MITIGACIONES

### Alto Impacto

**1. No obtener aprobación ética a tiempo**
- **Probabilidad:** Media
- **Impacto:** Alto (bloquea piloto)
- **Mitigación:** 
  - Iniciar proceso en Semana 1
  - Buscar 3+ comités en paralelo
  - Preparar documentación exhaustiva
  - Contingencia: iniciar con datos históricos anonimizados

**2. Partner clínico se retira**
- **Probabilidad:** Baja-Media
- **Impacto:** Alto
- **Mitigación:**
  - Tener 2+ partners comprometidos
  - MoU firmados con penalizaciones
  - Red de contactos amplia (congresos, asociaciones EM)

**3. Modelo no alcanza performance mínima**
- **Probabilidad:** Media
- **Impacto:** Alto (invalida hipótesis)
- **Mitigación:**
  - Definir umbrales realistas desde inicio (AUROC > 0.65)
  - Entrenar múltiples arquitecturas (TFT, LSTM, ensemble)
  - Consultar literatura (benchmarks similares)
  - Iteración rápida (weekly model updates)

### Medio Impacto

**4. Complejidad técnica subestimada**
- **Probabilidad:** Media-Alta
- **Impacto:** Medio (retraso)
- **Mitigación:**
  - Buffer de 20% en timeline
  - Arquitectura modular (fallos aislados)
  - Code reviews obligatorios
  - Pair programming en componentes críticos

**5. Problemas de privacidad/seguridad**
- **Probabilidad:** Baja
- **Impacto:** Crítico
- **Mitigación:**
  - Security by design desde día 1
  - Auditoría externa en Semana 15
  - Penetration testing
  - Bug bounty (post-launch)

---

## ✅ CHECKLIST DE ACCIONES INMEDIATAS (Semana 1)

### Lunes
- [ ] **(All)** Reunión kickoff con equipo completo
- [ ] **(Chief)** Definir roles y responsabilidades (RACI matrix)
- [ ] **(Archie)** Setup repositorios Git
- [ ] **(Chief)** Contratar consultor legal/DPO


### Martes-Miércoles
- [ ] Primera versión DPIA
- [ ] Contactar 3 centros neurológicos (email + llamada)
- [ ] Setup herramientas (Jira/Linear, Slack, Notion)
- [ ] Configurar CI/CD básico (GitHub Actions)

### Jueves-Viernes
- [ ] Definir protocolo clínico (borrador)
- [ ] Diseñar consentimiento informado (v0.1)
- [ ] Arquitectura técnica (ADR)
- [ ] Estimación detallada de tareas (sprint planning)

### Entregable Semana 1
📄 **Documento de Proyecto** con:
- Equipo y roles
- Timeline detallado (Gantt chart)
- DPIA borrador
- Protocolo clínico v0.1
- Arquitectura técnica
- Presupuesto aprobado

---

## 📈 MÉTRICAS DE ÉXITO (Piloto - Mes 6-12)

### Técnicas
- ✅ Uptime > 99.5%
- ✅ Latencia p95 < 200ms
- ✅ 0 incidentes de seguridad
- ✅ AUROC > 0.70 en datos reales

### Clínicas
- ✅ Sensibilidad > 70% para brotes confirmados
- ✅ FP rate < 20% (alertas falsas)
- ✅ Lead time promedio > 10 días
- ✅ 80%+ pacientes completan 6 meses

### Operativas
- ✅ 50+ pacientes enrolled
- ✅ Adherencia > 70% (datos enviados regularmente)
- ✅ Satisfacción clínicos > 4/5
- ✅ Data retention compliance 100%

---

## 🚀 PRÓXIMOS PASOS DESPUÉS DEL PILOTO

### Si resultados positivos (AUROC > 0.70, sensibilidad > 65%):

1. **Publicación científica** (3-6 meses)
   - Paper en journal de neurología/digital health
   - Presentación en congreso (AAN, ECTRIMS)

2. **Ensayo clínico controlado** (12-18 meses)
   - RCT con grupo control
   - Endpoints: reducción hospitalizaciones, QoL, costos
   - Tamaño: 200-500 pacientes

3. **Certificación regulatoria** (12-24 meses)
   - CE marking (MDR Class IIa)
   - FDA 510(k) o De Novo pathway
   - Dossier completo: performance, risk management, clinical evaluation

4. **Comercialización** (18+ meses)
   - B2B2C: venta a hospitales/aseguradoras
   - Freemium model para pacientes
   - Integración con EHR (Epic, Cerner)

### Si resultados negativos o mixtos:

1. **Análisis post-mortem**
   - ¿Qué features fueron predictivas?
   - ¿Qué falló? (datos, modelo, protocolo)

2. **Pivot**
   - Probar otras patologías (Parkinson, depresión)
   - Cambiar target: fatiga vs brote
   - Tool de monitoreo (no predictivo)

---

## 📚 RECURSOS Y HERRAMIENTAS RECOMENDADAS

### Desarrollo
- **Backend:** Python 3.11+, FastAPI, asyncio
- **ML:** PyTorch 2.x, pytorch-forecasting, scikit-learn
- **DB:** PostgreSQL 15 + TimescaleDB, Redis 7
- **Storage:** MinIO (S3-compatible)
- **Queue:** Redpanda (Kafka-compatible)
- **MLOps:** MLflow, BentoML, DVC
- **Monitoring:** Prometheus, Grafana, Sentry

### Cliente
- **Android:** Kotlin, Jetpack Compose, WorkManager, Room
- **Crypto:** Tink, EncryptedSharedPreferences
- **Testing:** JUnit, Espresso

### Infra
- **IaC:** Terraform or Pulumi
- **CI/CD:** GitHub Actions, ArgoCD
- **K8s:** K3s (edge) or EKS/GKE (cloud)
- **Secrets:** Hashicorp Vault or AWS KMS

### Compliance
- **Privacy:** OneTrust, TrustArc
- **Audit:** ELK Stack, immutable logs
- **Auth:** Keycloak, Auth0

---

## 🎓 APRENDIZAJES Y BEST PRACTICES

### Do's ✅
1. **Privacy by design:** No enviar texto crudo nunca
2. **Start simple:** Baseline models primero (Prophet, LSTM)
3. **Clinical validation:** Involucrar neurólogos desde día 1
4. **Explainability:** SHAP, attention weights para confianza clínica
5. **Iteración rápida:** Weekly model updates, A/B testing
6. **Documentation:** Todo debe estar documentado (código, decisiones, protocolos)

### Don'ts ❌
1. **No gold plating:** Evitar over-engineering en MVP
2. **No magic bullets:** TFT no garantiza éxito, probar ensembles
3. **No olvidar UX:** App difícil = baja adherencia = piloto fallido
4. **No subestimar legal:** GDPR violations son carísimas
5. **No black box:** Modelos deben ser explicables para adopción clínica

---

## 📞 CONTACTOS CLAVE

### Partners Potenciales (España)
- Hospital Clínic Barcelona - Servicio Neurología
- Hospital Gregorio Marañón Madrid - Unidad EM
- Hospital Vall d'Hebron - Neuroimmunología
- Fundación Esclerosis Múltiple (FEM)

### Regulatorio
- AEMPS (Agencia Española de Medicamentos)
- AEPD (Agencia Española Protección de Datos)

### Asociaciones
- ECTRIMS (European Committee for Treatment and Research in MS)
- Multiple Sclerosis International Federation

---

## 📄 CONCLUSIÓN

Este plan establece una ruta clara para desarrollar y validar un prototipo funcional de predicción de brotes de EM en **16 semanas**.

**Factores críticos de éxito:**
1. ✅ Aprobación ética temprana
2. ✅ Partner clínico comprometido
3. ✅ Equipo técnico sólido
4. ✅ Arquitectura de privacidad robusta
5. ✅ Expectativas realistas de performance

**Next step:** Aprobar presupuesto y comenzar Fase 0 (Legal + Arquitectura).

**¿Preguntas? → Contactar Product Lead**
