@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_REAL=C:\Users\armga\AppData\Local\Programs\Python\Python313\python.exe"

echo Ejecutando barrido de rasgos Deep ONMF ajustados...
"%PYTHON_REAL%" ".\05_barrido_rasgos_deep_onmf_ajustados.py" %*

if errorlevel 1 (
  echo.
  echo El barrido Deep ONMF ha terminado con error.
  exit /b %errorlevel%
)

echo.
echo Barrido terminado. Revisa ..\resultados.
endlocal
