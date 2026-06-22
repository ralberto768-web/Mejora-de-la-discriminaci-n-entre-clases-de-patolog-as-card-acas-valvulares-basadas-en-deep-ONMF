# Guia para el tribunal

Este repositorio permite tres niveles de comprobacion.

## Nivel 1: revisar evidencia ya generada

Abrir:

- `documento_global/DOCUMENTO_EVALUACION_CAPITULOS_6_7_8.pdf`
- `evidencia/capitulos_6_7_8/00_README_GENERAL.md`
- `docs/MAPA_INDICE_TFG.md`

Este nivel no requiere ejecutar codigo.

## Nivel 2: verificar integridad del paquete

Ejecutar:

```powershell
python scripts\verificar_repositorio.py --modo rapido
```

Para comprobacion con hashes de archivos pequenos y de control:

```powershell
python scripts\verificar_repositorio.py --modo completo
```

Los archivos grandes se controlan por presencia, tamano y Git LFS. La lista esta en `github/ARCHIVOS_GRANDES_GIT_LFS.csv`.

En Windows, si PowerShell bloquea scripts, ejecutar:

```powershell
.\run_all.bat todo
```

## Nivel 3: reproducir resultados

1. Instalar dependencias con `requirements.txt`.
2. Colocar bases de datos en `datos_externos`.
3. Revisar `docs/REPRODUCIBILIDAD.md`.
4. Ejecutar los scripts originales indicados por cada bloque de evidencia.

La validacion principal del repositorio no depende de una unica ejecucion larga: separa codigo, datos externos, resultados esperados y verificacion de hashes.
