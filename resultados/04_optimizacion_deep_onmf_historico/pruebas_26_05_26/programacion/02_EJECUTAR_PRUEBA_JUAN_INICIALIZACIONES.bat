@echo off
cd /d "%~dp0"
set "PYTHON313=C:\Users\armga\AppData\Local\Programs\Python\Python313\python.exe"
if exist "%PYTHON313%" (
    "%PYTHON313%" 01_ejecutar_prueba_inicializaciones.py
) else (
    python 01_ejecutar_prueba_inicializaciones.py
)
pause
