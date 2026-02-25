@echo off
TITLE EM-Predictor System Launcher
echo ===================================================
echo   EM-Predictor System Launcher
echo [1/5] Starting Unified Infrastructure (Postgres, Redis, Ollama, Kafka, MinIO, MLflow)...
docker-compose -f ..\telegram-archiver\docker-compose.unified.yml up -d
IF %ERRORLEVEL% NEQ 0 (
    echo Error starting infrastructure. Please check Docker Desktop.
    pause
    exit /b
)

echo Waiting for infrastructure and LLM to initialize...
timeout /t 5

echo [2/5] Starting ML Inference (Port 8001)...
start "ML Inference" cmd /k "cd services\ml-inference && .venv\Scripts\activate 2>nul || echo Virtual env not found & python main.py"

echo [4/5] Starting Unified Backend (Port 8010)...
start "Unified Backend" cmd /k "cd services\unified_app && .venv\Scripts\activate 2>nul || echo Virtual env not found & python main.py --port 8010"

echo [5/5] Starting Webapp Frontend (Port 5173)...
start "Frontend Webapp" cmd /k "cd services\webapp && npm run dev -- --host 0.0.0.0"

echo ===================================================
echo   System Started!
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8010/docs
echo   MLflow:   http://localhost:5000
echo   Ollama:   http://localhost:11434
echo ===================================================
echo Press any key to exit launcher (services will keep running)
pause
