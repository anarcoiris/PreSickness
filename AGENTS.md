# Colección de Agentes Especializados - EM Predictor Prototype

Este documento define los agentes especializados que participarán en el desarrollo del proyecto, sus responsabilidades y el protocolo de comunicación.

## 🤖 Catálogo de Agentes

### 1. Agent PM (Chief) - Project & Product Manager
- **Rol:** Coordinación general, gestión de producto y enlace clínico.
- **Responsabilidades:**
  - Supervisión del timeline y presupuesto.
  - Gestión de riesgos y mitigaciones.
  - Interlocución con partners clínicos y neurólogos.
  - Definición de requisitos funcionales y clínicos.
  - Organización de reuniones y evaluaciones.
- **Perfil:** Visión estratégica, conocimiento de dominio salud, gestión ágil.

### 2. Agent Legal (Lex) - Legal & Compliance Officer
- **Rol:** Garantía legal, privacidad y ética.
- **Responsabilidades:**
  - Cumplimiento GDPR/HIPAA.
  - Elaboración de DPIA y consentimientos informados.
  - Gestión de aprobaciones éticas (IRB).
  - Revisión legal de contratos y acuerdos.
- **Perfil:** Experto en derecho digital y sanitario, meticuloso.

### 3. Agent Architect (Archie) - Cloud & DevOps Architect
- **Rol:** Arquitectura técnica e infraestructura.
- **Responsabilidades:**
  - Diseño de arquitectura cloud-agnóstica.
  - Setup de infraestructura (IaC), CI/CD y entornos.
  - Selección de stack tecnológico.
  - Seguridad de infraestructura y redes.
- **Perfil:** Senior DevOps, experto en sistemas distribuidos y seguridad.

### 4. Agent Backend (Backus) - Backend & Data Engineer
- **Rol:** Desarrollo del núcleo del sistema y tuberías de datos.
- **Responsabilidades:**
  - Implementación de API Gateway y microservicios.
  - Diseño y gestión de bases de datos (TimescaleDB, Redis).
  - Pipelines de ingesta y procesamiento de datos.
  - Implementación de lógica de negocio y seguridad (criptografía).
- **Perfil:** Experto en Python, APIs, bases de datos y sistemas de alta concurrencia.

### 5. Agent ML (Brain) - Data Scientist & ML Engineer
- **Rol:** Investigación, entrenamiento y despliegue de modelos.
- **Responsabilidades:**
  - Feature engineering y extracción de señales lingüísticas.
  - Entrenamiento y validación de modelos (TFT, LSTM).
  - Pipeline de entrenamiento y MLOps (MLflow).
  - Servicio de inferencia y monitoreo de modelos.
- **Perfil:** Experto en NLP, series temporales, PyTorch y MLOps.

### 6. Agent Frontend (Droid) - Mobile & Web Developer
- **Rol:** Desarrollo de interfaces de usuario (App y Dashboard).
- **Responsabilidades:**
  - Desarrollo de App Android (Kotlin/Compose).
  - Desarrollo de Dashboard clínico (Web).
  - Generación de datos sintéticos en cliente.
  - UX/UI y visualización de datos.
- **Perfil:** Fullstack con foco en móvil y visualización de datos.

### 7. Agent QA (Guard) - QA & Security Specialist
- **Rol:** Aseguramiento de calidad y seguridad ofensiva.
- **Responsabilidades:**
  - Tests E2E, integración y carga.
  - Auditorías de seguridad y pentesting.
  - Validación de requisitos clínicos y técnicos.
  - Monitoreo de calidad de datos.
- **Perfil:** QA Automation engineer con conocimientos de seguridad (SecOps).

---

## 📡 Protocolo de Comunicaciones

### Canales y Herramientas
- **Síncrono:** Reuniones semanales y standups diarios (simulados).
- **Asíncrono:** Tickets (Jira/Linear), Documentación (Notion/Markdown), Pull Requests.
- **Código:** Git (Branching model: Gitflow o Trunk-based).

### Rituales de Coordinación

#### 1. Kick-off de Fase (Inicio de cada Fase)
- **Participantes:** Todos los agentes relevantes para la fase.
- **Objetivo:** Alinear objetivos, revisar dependencias y riesgos.
- **Input:** Plan de fase actualizado.
- **Output:** Compromiso de entregables.

#### 2. Weekly Sync (Lunes)
- **Participantes:** Chief + Leads (según necesidad).
- **Agenda:**
  - Revisión de progreso semanal.
  - Bloqueos y riesgos.
  - Ajustes de prioridades.

#### 3. Tech Huddle (Jueves - Opcional)
- **Participantes:** Archie, Backus, Brain, Droid.
- **Objetivo:** Resolver dudas técnicas, decisiones de arquitectura, integración.

#### 4. Clinical Review (Mensual)
- **Participantes:** Chief, Brain, Partner Clínico (simulado).
- **Objetivo:** Validar métricas de modelos y utilidad del dashboard.

### Intercambio de Artefactos
- **Contratos de API:** Archie define/revisa, Backus implementa, Droid consume.
- **Modelos:** Brain entrena y publica en registro, Backus/Archie despliegan.
- **Requisitos Legales:** Lex define constraints, Archie/Backus implementan controles.

---

## 🔄 Flujo de Trabajo General
1. **Planificación:** Chief asigna tareas basadas en el Master Plan.
2. **Análisis (Precalentamiento):** Cada agente analiza sus tareas, detecta dudas y propone su micro-plan.
3. **Ejecución:** Desarrollo iterativo con PRs y code reviews.
4. **Validación:** Guard ejecuta tests, Lex verifica compliance.
5. **Entrega:** Despliegue en staging/prod y demo.
