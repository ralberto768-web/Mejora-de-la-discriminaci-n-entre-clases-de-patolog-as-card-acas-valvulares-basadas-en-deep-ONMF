# Reproducibilidad

## Objetivo

Permitir que otra persona pueda comprobar la evidencia incluida y, si dispone de los datos externos, repetir el flujo experimental.

## Flujo recomendado

1. Crear entorno Python.
2. Instalar dependencias.
3. Verificar estructura del repositorio.
4. Revisar resultados esperados.
5. Colocar datos externos si se quiere ejecutar desde cero.
6. Ejecutar los scripts originales de cada bloque experimental.

## Comandos

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\comprobar_entorno.py
python scripts\verificar_repositorio.py --modo rapido
python scripts\resumen_resultados.py
```

En Windows se puede ejecutar todo con:

```powershell
.\run_all.bat todo
```

## Repetibilidad numerica

Los resultados de aprendizaje automatico pueden variar ligeramente si cambian:

- version de Python/librerias;
- CPU/GPU;
- semillas aleatorias;
- configuracion determinista de PyTorch;
- particiones k-fold.

Por eso el repositorio conserva particiones, configuraciones, tablas finales, figuras y manifiestos. Para comparar una nueva ejecucion se deben contrastar sus resultados con `docs/RESULTADOS_ESPERADOS.md` y los CSV incluidos.
