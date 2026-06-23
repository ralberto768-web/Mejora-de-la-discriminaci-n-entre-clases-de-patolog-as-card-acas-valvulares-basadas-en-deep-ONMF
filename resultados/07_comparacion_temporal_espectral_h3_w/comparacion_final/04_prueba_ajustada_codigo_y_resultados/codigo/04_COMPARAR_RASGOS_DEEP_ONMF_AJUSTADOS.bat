@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_REAL=C:\Users\armga\AppData\Local\Programs\Python\Python313\python.exe"

echo Comparando los mejores rasgos Deep ONMF ajustados con la Figura 11...
"%PYTHON_REAL%" ".\03_comparar_con_rasgos_deep_onmf_ajustados.py" %*

if errorlevel 1 (
  echo.
  echo La comparacion Deep ONMF ajustada ha terminado con error.
  exit /b %errorlevel%
)

echo.
echo Comparacion Deep ONMF ajustada terminada.
endlocal
