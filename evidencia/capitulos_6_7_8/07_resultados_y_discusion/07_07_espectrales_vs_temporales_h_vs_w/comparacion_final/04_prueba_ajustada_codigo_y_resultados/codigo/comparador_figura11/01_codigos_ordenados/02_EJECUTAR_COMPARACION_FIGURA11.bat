@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON_REAL=C:\Users\armga\AppData\Local\Programs\Python\Python313\python.exe"

echo Ejecutando la comparacion final de la Figura 11...
"%PYTHON_REAL%" ".\01_codigos_ordenados\01_comparar_figura11.py" %*

if errorlevel 1 (
  echo.
  echo La comparacion ha terminado con error.
  exit /b %errorlevel%
)

echo.
echo Extrayendo la referencia de la Figura 11 desde el PDF objetivo...
"%PYTHON_REAL%" ".\01_codigos_ordenados\03_extraer_referencia_figura11_pdf.py"

if errorlevel 1 (
  echo.
  echo La extraccion de la referencia PDF ha terminado con error.
  exit /b %errorlevel%
)

echo.
echo Comparacion terminada. Revisa la carpeta 02_resultados.
endlocal
