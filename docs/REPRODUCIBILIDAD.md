# Reproducibilidad

Este repositorio está preparado para dos usos distintos:

1. revisar y verificar la evidencia ya generada;
2. repetir los experimentos si se dispone de los datos externos originales.

## Verificación sin datos externos

Sin descargar bases de datos se puede comprobar:

- que el repositorio contiene los documentos principales;
- que los manifiestos existen;
- que los resultados clave están presentes;
- que las tablas y figuras usadas por la memoria están incluidas;
- que los archivos grandes están registrados mediante Git LFS.

Comando recomendado en Windows:

```powershell
.\run_all.bat todo
```

Comandos separados:

```powershell
python scripts\comprobar_entorno.py
python scripts\verificar_repositorio.py --modo rapido
python scripts\resumen_resultados.py
```

## Reproducción con datos externos

Para repetir los experimentos desde cero se necesitan los audios o bases de datos originales. Estos datos no se suben directamente al repositorio por tamaño y posibles restricciones de distribución.

Estructura esperada:

```text
datos_externos/
  Yaseen/
  AWGN/
```

Después de colocar los datos, se deben usar los scripts y configuraciones conservados dentro de `evidencia/capitulos_6_7_8/`.

## Entorno Python

Crear entorno:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Comprobar dependencias:

```powershell
python scripts\comprobar_entorno.py
```

## Control de variabilidad

Los resultados de aprendizaje automático pueden variar ligeramente si cambian:

- versión de Python;
- versión de NumPy, scikit-learn, PyTorch u otras librerías;
- CPU/GPU;
- semillas aleatorias;
- configuración determinista de PyTorch;
- particiones k-fold.

Por eso este repositorio conserva configuraciones, particiones, tablas finales, figuras, matrices de confusión y manifiestos. Una nueva ejecución debe compararse contra `docs/RESULTADOS_ESPERADOS.md` y los CSV incluidos.

## Git LFS

Antes de clonar o descargar completamente los archivos grandes:

```powershell
git lfs install
git lfs pull
```

Sin Git LFS, algunos `.npy`, `.npz`, `.pdf` u otros ficheros pesados pueden quedar como punteros y no como archivos completos.

