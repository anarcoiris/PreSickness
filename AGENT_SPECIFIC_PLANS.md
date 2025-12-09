# Planes Específicos y Precalentamiento de Agentes

Este documento contiene los planes detallados ("precalentamientos") generados por cada agente especializado tras revisar su asignación en el Plan Maestro.

---

## 🎩 Agent PM (Chief) - Plan de Arranque

**Estado:** 🟢 Listo para Kick-off
**Enfoque:** Mitigación de riesgos iniciales y alineación de equipo.

### 📋 Micro-Plan Semana 1
1. **Reunión Kick-off:**
   - Agenda: Presentación del equipo, revisión de objetivos, Q&A.
   - Entregable: Acta de reunión y compromiso de equipo.
2. **Gestión de Riesgos:**
   - Prioridad 1: Contactar partners clínicos. Tengo 3 emails preparados para enviar el Martes.
   - Prioridad 2: Validar presupuesto para contratación de Legal.
3. **Herramientas:**
   - Configurar tablero en Linear/Jira con swimlanes por agente.
   - Crear canal de Slack #general y específicos por fase.

**❓ Dudas/Bloqueos:**
- Necesito confirmación del presupuesto final para aprobar la contratación del consultor legal externo.

---

## ⚖️ Agent Legal (Lex) - Evaluación de Compliance

**Estado:** 🟡 Esperando aprobación de presupuesto
**Enfoque:** Blindaje legal desde el diseño (Privacy by Design).

### 📋 Micro-Plan Semana 1-2
1. **DPIA (Evaluación de Impacto):**
   - Identificar flujo de datos sensibles (síntomas, medicación).
   - Definir medidas de mitigación (pseudonimización).
2. **Consentimiento Informado:**
   - Redactar v1.0 para pacientes (lenguaje claro, GDPR compliant).
   - Cláusula específica para uso de datos en entrenamiento de IA.
3. **Investigación:**
   - Revisar regulación actual sobre "Software as a Medical Device" (SaMD) para anticipar Fase 3.

**📝 Requisitos para otros agentes:**
- **@Archie:** Necesito diagrama de flujo de datos para el DPIA.
- **@Backus:** Confirmar qué datos se guardan en texto plano (espero que ninguno).

---

## 🏗️ Agent Architect (Archie) - Blueprint Técnico

**Estado:** 🟢 Diseñando infraestructura
**Enfoque:** Simplicidad, seguridad y escalabilidad horizontal.

### 📋 Micro-Plan Semana 3-4
1. **Repositorio:**
   - Decisión: Monorepo (Nx o Turborepo) para facilitar integración Backend-Frontend-ML.
   - Estructura: `/apps/backend`, `/apps/android`, `/libs/shared`, `/infra`.
2. **Infraestructura (IaC):**
   - Tool: Terraform.
   - Provider: AWS (por madurez en servicios HIPAA compliance) o Hetzner (si presupuesto es ajustado, pero requiere más config manual). Asumiré AWS por defecto.
3. **CI/CD:**
   - GitHub Actions.
   - Pipelines: Linting -> Unit Tests -> Build Docker -> Push Registry.

**🔧 Decisiones Técnicas Preliminares:**
- **Container Orchestration:** Docker Compose para dev, K8s (EKS) para prod.
- **Secret Management:** AWS Secrets Manager.

---

## ⚙️ Agent Backend (Backus) - Diseño de Core

**Estado:** 🟢 Prototipando API
**Enfoque:** Performance y seguridad.

### 📋 Micro-Plan Semana 5-6
1. **API Gateway:**
   - Framework: FastAPI (Python) por su velocidad y soporte asíncrono.
   - Auth: OAuth2 con JWT.
2. **Base de Datos:**
   - TimescaleDB (sobre Postgres) es perfecta para series temporales de sensores/síntomas.
   - Redis para caché y cola de tareas rápidas.
3. **Seguridad:**
   - Implementar librería compartida de encriptación (Fernet) para datos sensibles en reposo.

**❓ Preguntas para Brain:**
- ¿Qué formato de datos necesitas para el entrenamiento? (CSV, Parquet, JSON?)
- ¿Frecuencia de ingesta de datos? (Real-time vs Batch)

**Progreso 02/12**
- API Gateway implementado en `services/api-gateway` con autenticación + Kafka.
- Worker de extracción y ventanas funcionando (`services/feature-extractor`).

**Próximas 48h**
- Instrumentar pruebas E2E de ingesta → ventanas.
- Preparar contratos para servicio de inferencia y alert manager.

---

## 🧠 Agent ML (Brain) - Estrategia de Modelado

**Estado:** 🟡 Investigando SOTA
**Enfoque:** Baseline robusto antes de complejidad.

### 📋 Micro-Plan Semana 7-10
1. **Baseline:**
   - Implementar modelo simple (Logistic Regression o Random Forest) sobre features manuales para tener un benchmark.
2. **TFT (Temporal Fusion Transformer):**
   - Es el target, pero complejo. Empezaré con `pytorch-forecasting`.
3. **Datos Sintéticos:**
   - Necesito generar datos que simulen brotes. Crearé un script generador basado en distribuciones estadísticas conocidas de EM.

**📝 Requisitos para Backus:**
- Necesito acceso directo a una réplica de lectura de la DB o un dump diario en S3/MinIO.
- Los logs de texto deben estar pre-procesados (limpieza básica) si es posible.

**Progreso 02/12**
- Script `train_tft.py` refactorizado con configuración declarativa y tracking MLflow.

**Próximas 48h**
- Ejecutar primer experimento completo y registrar métricas (AUROC, AUPRC).
- Diseñar servicio `ml-inference` y definir serialización de modelos (TorchScript/ONNX).

---

## 📱 Agent Frontend (Droid) - UX/UI Concept

**Estado:** 🟢 Bocetando
**Enfoque:** Accesibilidad y facilidad de uso (pacientes con posibles dificultades motoras/visuales).

### 📋 Micro-Plan Semana 13-14
1. **Tech Stack:**
   - Android: Kotlin + Jetpack Compose (Moderno, declarativo).
   - Web: React + TailwindCSS (Rápido desarrollo).
2. **Prototipo:**
   - Pantalla 1: "Check-in diario" (Emoji slider + campo de texto opcional).
   - Pantalla 2: "Mi historial" (Gráfica simple).
3. **Accesibilidad:**
   - Botones grandes, alto contraste, soporte para voz (speech-to-text).

**❓ Preguntas para Chief:**
- ¿Tenemos logo/branding? Si no, usaré un placeholder limpio.

---

## 🛡️ Agent QA (Guard) - Estrategia de Calidad

**Estado:** 🟢 Preparando entorno
**Enfoque:** Shift-left testing.

### 📋 Micro-Plan General
1. **Estrategia de Pruebas:**
   - Unitarias: Responsabilidad de cada dev (Coverage > 70%).
   - Integración: API tests con Pytest.
   - E2E: Playwright para dashboard, Maestro para Android.
2. **Seguridad:**
   - Configurar SonarQube en CI/CD para análisis estático.
   - Planificar pentest manual para Semana 15.

**⚠️ Alerta:**
- Necesito datos de prueba anonimizados lo antes posible para los tests de integración.
