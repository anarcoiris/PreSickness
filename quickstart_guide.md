# 🚀 EM Predictor - Quickstart Guide

Guía para iniciar el desarrollo del prototipo en **menos de 30 minutos**.

> **Última actualización:** 02/12/2025  
> **Versión:** 0.2.0

---

## 📦 Estructura del Proyecto

```
em-predictor/
├── services/
│   ├── api-gateway/        # FastAPI - Ingesta de datos
│   ├── feature-extractor/  # Worker de extracción de features
│   ├── ml-inference/       # Servicio de predicción
│   ├── alert-manager/      # Gestión de alertas
│   └── dashboard/          # UI clínica (Streamlit)
├── MiniLLM/                # Modelo de lenguaje local (opcional)
├── monitoring/             # Prometheus + Grafana
├── docs/                   # Documentación
├── docker-compose.yml      # Orquestación de servicios
├── db_schema.sql           # Schema de base de datos
└── train_tft.py            # Pipeline de entrenamiento ML
```

---

## 📋 Pre-requisitos

### Software necesario:
```bash
# Check versions
python --version  # 3.11+
docker --version  # 20.10+
docker-compose --version  # 2.0+
git --version
node --version  # 18+ (para dashboard)
```

### Instalar dependencias:
```bash
# macOS
brew install python@3.11 docker docker-compose postgresql redis

# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv docker.io docker-compose postgresql-client redis-tools

# Verificar Docker está corriendo
docker ps
```

---

## 🏗️ Setup del Proyecto (Primera Vez)

### 1. Clonar estructura del proyecto

```bash
# Crear estructura de directorios
mkdir -p em-predictor/{services,ml,data,docs,tests}
cd em-predictor

# Inicializar Git
git init
git checkout -b main

# Crear .gitignore
cat > .gitignore << 'EOF'
*.pyc
__pycache__/
.env
*.log
data/raw/
data/processed/
.DS_Store
*.sqlite
.pytest_cache/
.venv/
venv/
node_modules/
*.pem
*.key
checkpoints/
mlruns/
EOF
```

### 2. Crear entorno virtual Python

```bash
# Crear venv
python3.11 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias base
cat > requirements.txt << 'EOF'
# Web Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Database
asyncpg==0.29.0
psycopg2-binary==2.9.9
redis==5.0.1

# ML Core
torch==2.1.2
pytorch-lightning==2.1.3
pytorch-forecasting==1.0.0
scikit-learn==1.4.0
numpy==1.26.3
pandas==2.1.4

# NLP
transformers==4.36.2
sentence-transformers==2.2.2
textblob==0.17.1

# Audio
librosa==0.10.1
soundfile==0.12.1

# MLOps
mlflow==2.9.2
bentoml==1.1.11

# Crypto
cryptography==41.0.7

# Utils
python-dotenv==1.0.0
python-multipart==0.0.6
aiofiles==23.2.1
httpx==0.26.0

# Testing
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0

# Monitoring
prometheus-client==0.19.0
EOF

pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
# Crear .env
cat > .env << 'EOF'
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=empredictor
DB_USER=emuser
DB_PASSWORD=changeme_strong_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=changeme_redis_password

# MinIO (S3)
MINIO_HOST=localhost
MINIO_PORT=9000
MINIO_USER=minioadmin
MINIO_PASSWORD=minioadmin123_strong

# Kafka
KAFKA_BROKERS=localhost:19092

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Encryption
ENCRYPTION_KEY=generate_with_python_fernet  # Ver abajo

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
EOF

# Generar encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" >> .env
```

### 4. Crear estructura de servicios

```bash
# API Gateway
mkdir -p services/api-gateway
cat > services/api-gateway/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

# Feature Extractor
mkdir -p services/feature-extractor
cat > services/feature-extractor/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "worker.py"]
EOF

# ML Inference
mkdir -p services/ml-inference
# Dashboard
mkdir -p services/dashboard
# Alert Manager
mkdir -p services/alert-manager
```

---

## 🐳 Levantar Infraestructura

### 1. Copiar docker-compose.yml (del artifact anterior)

```bash
# Crear docker-compose.yml en la raíz
# (copiar contenido del artifact "docker_compose")
```

### 2. Inicializar base de datos

```bash
# Crear directorio SQL
mkdir -p sql

# Copiar schema.sql (del artifact anterior)
# Guardar en sql/schema.sql
```

### 3. Levantar servicios

```bash
# Levantar solo infra (sin servicios custom)
docker-compose up -d postgres redis minio redpanda mlflow

# Verificar que todo está corriendo
docker-compose ps

# Ver logs
docker-compose logs -f postgres
```

### 4. Inicializar base de datos

```bash
# Esperar a que Postgres esté listo
sleep 10

# Aplicar schema
docker-compose exec postgres psql -U emuser -d empredictor -f /docker-entrypoint-initdb.d/01-schema.sql

# Verificar tablas
docker-compose exec postgres psql -U emuser -d empredictor -c "\dt"
```

---

## 🧪 Pruebas Rápidas

### Test 1: API Gateway

```bash
# Copiar api_gateway.py al directorio
cp <artifact_api_gateway> services/api-gateway/main.py

# Instalar dependencias locales
pip install fastapi uvicorn asyncpg redis cryptography

# Ejecutar localmente (para desarrollo rápido)
cd services/api-gateway
uvicorn main:app --reload --port 8000

# En otra terminal, probar health check
curl http://localhost:8000/v1/health
```

### Test 2: Feature Extractor

```bash
# Copiar feature_extractor.py
cp <artifact_feature_extractor> services/feature-extractor/worker.py

# Crear script de prueba
cat > services/feature-extractor/test_extraction.py << 'EOF'
import asyncio
from worker import FeatureExtractor

async def test():
    config = {
        'redis_url': 'redis://localhost:6379',
        'db_dsn': 'postgresql://emuser:changeme@localhost/empredictor'
    }
    
    extractor = FeatureExtractor(config)
    await extractor.initialize()
    
    # Test text features
    text = "I'm feeling tired today. My legs feel heavy and I'm having trouble concentrating."
    features = extractor.extract_text_features(text)
    print("Text features:", features)
    
    # Test embedding
    embedding = extractor.compute_embedding(text)
    print(f"Embedding shape: {embedding.shape}")
    
    await extractor.close()

asyncio.run(test())
EOF

python services/feature-extractor/test_extraction.py
```

### Test 3: Synthetic Data Generator

```bash
# Crear script de prueba
cat > test_synthetic.py << 'EOF'
import numpy as np
import json
from datetime import datetime

def generate_synthetic_datapoint():
    """Genera un datapoint sintético"""
    return {
        'user_id_hash': 'test_' + '0' * 60,
        'timestamp': datetime.utcnow().isoformat(),
        'embedding': np.random.randn(768).tolist(),
        'numeric_features': {
            'sentiment_score': float(np.random.uniform(-0.3, 0.3)),
            'avg_sentence_len': float(np.random.uniform(8, 16)),
            'type_token_ratio': float(np.random.uniform(0.6, 0.8)),
            'num_messages': int(np.random.randint(5, 20)),
            'steps': int(np.random.randint(3000, 10000)),
            'sleep_hours': float(np.random.uniform(5, 8))
        }
    }

# Generar 10 datapoints
for i in range(10):
    dp = generate_synthetic_datapoint()
    print(json.dumps(dp, indent=2))
    print("---")
EOF

python test_synthetic.py
```

---

## 🎯 Hitos de Validación (Primera Semana)

### Día 1: Infraestructura
- [ ] Docker Compose levanta todos los servicios
- [ ] Postgres acepta conexiones
- [ ] Redis responde a ping
- [ ] MLflow UI accesible (http://localhost:5000)

### Día 2-3: API Gateway
- [ ] Health check funciona
- [ ] Puede ingerir datapoint sintético
- [ ] Datos se guardan en Postgres
- [ ] Logs aparecen correctamente

### Día 4-5: Feature Extraction
- [ ] Extrae features de texto
- [ ] Genera embeddings
- [ ] Calcula ventanas temporales
- [ ] Almacena en feature store

---

## 🐛 Troubleshooting Común

### Postgres no inicia
```bash
# Ver logs
docker-compose logs postgres

# Resetear volumen
docker-compose down -v
docker-compose up -d postgres
```

### Redis connection refused
```bash
# Verificar que Redis está corriendo
docker-compose ps redis

# Probar conexión
redis-cli -h localhost ping
```

### Python dependencies fail
```bash
# Recrear venv
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Docker out of memory
```bash
# Aumentar memoria en Docker Desktop (Mac/Windows)
# Settings > Resources > Memory > 8GB

# Linux: verificar límites
docker info | grep -i memory
```

---

## 📊 Acceso a UIs

Una vez todo está corriendo:

| Servicio | URL | Credenciales |
|----------|-----|--------------|
| **API Gateway (Docs)** | http://localhost:8000/docs | — |
| **ML Inference (Docs)** | http://localhost:8001/docs | — |
| **Alert Manager (Docs)** | http://localhost:8002/docs | — |
| **Dashboard Clínico** | http://localhost:8501 | — |
| **MLflow** | http://localhost:5000 | — |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin123 |
| **Grafana** | http://localhost:3000 | admin / admin |
| **Redpanda Console** | http://localhost:8080 | — |

---

## 🚀 Comandos Rápidos

### Levantar todo el stack

```bash
# Solo infraestructura (DB, Redis, Kafka, MLflow)
docker-compose up -d postgres redis minio redpanda mlflow

# Stack completo (incluye servicios)
docker-compose up -d

# Ver logs de un servicio
docker-compose logs -f api_gateway
```

### Entrenar modelo

```bash
# Activar entorno virtual
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Ejecutar entrenamiento
python train_tft.py

# Con parámetros personalizados (via .env o variables)
DB_DSN=postgresql://... MLFLOW_URI=http://... python train_tft.py
```

### Dashboard local

```bash
cd services/dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎓 Próximos Pasos

### Completados ✅
- [x] API Gateway con autenticación y Kafka
- [x] Feature Extractor con ventanas temporales
- [x] Pipeline de entrenamiento TFT
- [x] Servicio de inferencia ML
- [x] Alert Manager con notificaciones
- [x] Dashboard clínico básico

### Pendientes 🔜
- [ ] Integrar MiniLLM para features avanzadas (ver `docs/MINILLM_INTEGRATION_PLAN.md`)
- [ ] Tests E2E del flujo completo
- [ ] Configurar CI/CD con GitHub Actions
- [ ] Deploy en staging

**Documentación adicional:**
- `docs/MINILLM_INTEGRATION_PLAN.md` - Plan de integración del LLM local
- `AGENT_SPECIFIC_PLANS.md` - Planes detallados por agente
- `project_timeline.md` - Timeline y presupuesto

---

## 💬 ¿Problemas?

1. Revisa logs: `docker-compose logs -f <service>`
2. Verifica .env: `cat .env`
3. Consulta arquitectura: `docs/architecture.md`
4. Abre issue en GitHub

**Good luck! 🚀**
