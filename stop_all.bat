@echo off
TITLE PreSickness Stopper
echo Stopping Python services...
taskkill /F /IM python.exe /T
echo Stopping Node.js frontend...
taskkill /F /IM node.exe /T
echo Stopping Docker containers...
docker stop em_postgres em_redis mlflow
echo All services stopped.
pause
