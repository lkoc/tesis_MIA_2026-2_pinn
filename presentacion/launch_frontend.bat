@echo off
setlocal
cd /d "%~dp0.."

echo [PINN_V4] Verificando si Streamlit ya esta activo en http://localhost:8501 ...
netstat -ano | findstr ":8501 " | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo [PINN_V4] Iniciando Front-End Streamlit en nueva ventana...
    start "PINN_V4 Front-End (Streamlit)" cmd /k "streamlit run frontend\app.py --server.port 8501"
    timeout /t 3 /nobreak >nul
) else (
    echo [PINN_V4] Streamlit ya estaba escuchando en el puerto 8501.
)

start "" "http://localhost:8501"
endlocal
