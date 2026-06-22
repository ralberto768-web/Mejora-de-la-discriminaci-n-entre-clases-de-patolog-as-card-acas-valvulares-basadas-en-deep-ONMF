@echo off
setlocal
set "REPO_DIR=%~dp0"
set "MODO=%~1"
if "%MODO%"=="" set "MODO=todo"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%REPO_DIR%run_all.ps1" -Modo "%MODO%"
