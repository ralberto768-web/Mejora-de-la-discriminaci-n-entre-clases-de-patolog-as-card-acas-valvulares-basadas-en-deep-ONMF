@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_REAL=C:\Users\armga\AppData\Local\Programs\Python\Python313\python.exe"

echo Ejecutando Deep ONMF ajustado sin eliminar audios...
"%PYTHON_REAL%" ".\01_ejecutar_deep_onmf_ajustado.py" %*

if errorlevel 1 (
  echo.
  echo La ejecucion Deep ONMF ajustada ha terminado con error.
  exit /b %errorlevel%
)

echo.
echo Revisa la carpeta resultados de Programacion objetivo.
endlocal
