@echo off
TITLE PreSickness Launcher
echo ===================================================
echo   PreSickness System Launcher
echo ===================================================

echo [1/6] Starting Core Databases (Postgres, Redis)...
docker start em_postgres em_redis em_nlp_agent
IF %ERRORLEVEL% NEQ 0 (
    echo Error starting databases. Please check Docker Desktop.
    pause
    exit /b
)

echo [2/6] Starting MLflow...
docker start mlflow
IF %ERRORLEVEL% NEQ 0 (
    echo MLflow container not found. Creating it...
    docker run -d -p 5000:5000 --name mlflow -v "%cd%/mlruns:/mlruns" python:3.10-slim bash -c "pip install mlflow && mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db"
)
timeout /t 5

echo [3/6] Starting NLP Agent (Port 8002)...
start "NLP Agent" cmd /k "cd services\nlp-agent && python main.py"

echo [4/6] Starting ML Inference (Port 8001)...
start "ML Inference" cmd /k "cd services\ml-inference && python main.py"

echo [5/6] Starting Unified Backend (Port 8080)...
start "Unified Backend" cmd /k "cd services\unified_app && python main.py --port 8080"

echo [6/6] Starting Webapp Frontend...
start "Frontend Webapp" cmd /k "cd services\webapp && npm run dev -- --host 0.0.0.0"

echo ===================================================
echo   System Started!
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8080/docs
echo   MLflow:   http://localhost:5000
echo ===================================================
echo Press any key to exit launcher (services will keep running)
pause
