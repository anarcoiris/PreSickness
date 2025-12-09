# Plataforma EM-Predictor

## Resumen

Se ha implementado una plataforma completa para predicción de esclerosis múltiple con:
- **CLI interactivo** para gestión de servicios
- **Backend API** con autenticación JWT, Alertas y Analytics
- **Frontend React** con diseño premium y 5 módulos principales

---

## Componentes Creados

### 1. CLI (`cli.py`)
CLI con Rich para gestión de servicios.
- `python cli.py status` - Ver estado (Docker + Local)
- `python cli.py start [all|infra|backend|webapp]`
- `python cli.py ngrok` - Exponer plataforma a internet

### 2. Backend API
**Ruta:** `services/unified_app/main.py`
**Funcionalidades:**
- **Autenticación**: JWT (Login/Register)
- **Gestión de Pacientes**: Perfiles y configuración
- **Uploads**: Carga de datos de salud (CSV/JSON/XLSX)
- **Alertas**: Endpoint `/api/alerts` para notificaciones
- **Métricas**: Endpoint `/api/metrics` para estado del sistema
- **Proxy ML**: Redirección inteligente a servicios de inferencia

**Persistencia:**
- Soporte para **PostgreSQL** con `database.py`
- Fallback automático a memoria (mocks) si no hay DB disponible

### 3. Frontend React
**Ruta:** `webapp/`
**Módulos:**
- 📊 **Dashboard**: Resumen y predicción rápida
- 📈 **Analytics**: Gráfico de tendencias e historial de riesgo
- 🔔 **Notificaciones**: Centro de alertas y recordatorios
- 📤 **Upload**: Carga de datos con drag & drop
- 👤 **Perfil**: Gestión de usuario y seguridad

**Tecnología:** Vite, React Router v6, Axios, CSS Modules (Diseño Premium)

---

## Cómo Ejecutar (Flujo Completo)

### 1. Iniciar Infraestructura (Docker)
Para habilitar persistencia y ML real:
```bash
python cli.py start infra
```
*Esto iniciará PostgreSQL, Redis, ML Inference, etc.*

### 2. Reiniciar Backend (si es necesario)
Si el backend ya corre, se reconectará automáticamente o puedes reiniciarlo:
```bash
cd services/unified_app
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### 3. Frontend
```bash
cd webapp
npm run dev
```

### 4. Exponer con ngrok
```bash
ngrok http 8080
```

---

## URLs Locales

| Servicio | URL |
|----------|-----|
| API Backend | http://localhost:8080 |
| Frontend React | http://localhost:5173 |
| Swagger UI | http://localhost:8080/docs |

---

## Estado Actual
- **Modo Prototipo**: Funcional sin Docker (usa memoria y mocks)
- **Modo Producción**: Activar Docker para persistencia real y ML real.
