# Mejora de la discriminacion entre clases de patologias cardiacas valvulares basadas en Deep-ONMF

Este repositorio contiene el codigo, las configuraciones, los resultados y la documentacion tecnica de un Trabajo Fin de Grado sobre clasificacion automatica de sonidos cardiacos. El trabajo estudia representaciones temporales obtenidas con Deep-ONMF y su uso junto con clasificadores de aprendizaje automatico para discriminar clases de patologias cardiacas valvulares.

El repositorio esta preparado para que un lector externo pueda entender que se ha hecho, que resultados se han obtenido y que archivos necesita revisar o ejecutar, sin depender de carpetas privadas ni de documentos locales del autor.

## Que incluye

- `metodologia/`: codigo, configuraciones y material tecnico del flujo Deep-ONMF y del sistema de clasificacion.
- `resultados/`: resultados organizados por tipo de experimento: bases de datos, metricas, validacion cruzada, optimizacion, escenario sin ruido, escenario con ruido AWGN y comparacion entre representaciones `H3` y `W`.
- `conclusiones/`: evidencias usadas para cerrar objetivos, limitaciones y lineas futuras.
- `informe_general/`: informe PDF y Markdown con una lectura integrada del trabajo y de los resultados principales.
- `docs/`: guias de lectura, reproducibilidad, datos externos y resultados esperados.
- `scripts/`: comprobaciones automaticas de entorno, integridad del repositorio y resumen de evidencias.
- `datos_externos/`: plantilla para colocar bases de datos que no se distribuyen directamente en GitHub.
- `github/`: manifiestos del repositorio, lista de archivos grandes y estado de la entrega.
- `verificacion/`: manifiestos y resumen de integridad de la evidencia experimental incluida.

## Lectura recomendada

Para entender el contenido sin ejecutar nada:

1. `docs/LECTURA_RAPIDA.md`
2. `informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf`
3. `docs/GUIA_ESTRUCTURA_REPOSITORIO.md`
4. `docs/RESULTADOS_ESPERADOS.md`

Para comprobar que la descarga esta completa:

```powershell
.\run_all.bat todo
```

Para preparar una reproduccion completa de los experimentos:

```text
docs/DATOS_EXTERNOS.md
docs/REPRODUCIBILIDAD.md
```

La reproduccion completa requiere disponer de las bases de datos externas de sonidos cardiacos. Los audios fuente no se incluyen directamente en el repositorio por tamano y por posibles restricciones de licencia.

## Flujo experimental

El trabajo sigue este esquema general:

1. Preparacion de senales de fonocardiograma.
2. Extraccion de caracteristicas temporales mediante Deep-ONMF.
3. Obtencion de matrices internas del modelo, principalmente `H3` y `W`.
4. Clasificacion mediante UjaNet y comparacion con representaciones clasicas.
5. Evaluacion con validacion cruzada k-fold.
6. Analisis en escenario sin ruido y con ruido AWGN.
7. Comparacion entre informacion temporal y espectral, incluyendo el contraste `H3` frente a `W`.
8. Discusion de coste, dimension de entrada, robustez y limitaciones.

## Estructura de resultados

| Ruta | Contenido |
|---|---|
| `resultados/01_bases_de_datos_y_escenarios/` | Evidencia sobre bases de datos y escenarios de evaluacion. |
| `resultados/02_metricas_de_evaluacion/` | Definiciones, tablas y salidas relacionadas con metricas. |
| `resultados/03_validacion_cruzada_kfold/` | Particiones, validaciones y resultados de k-fold. |
| `resultados/04_optimizacion_deep_onmf/` | Configuraciones optimizadas por capas y dimensiones. |
| `resultados/04_optimizacion_deep_onmf_historico/` | Resultados historicos complementarios de la busqueda Deep-ONMF. |
| `resultados/05_escenario_sin_ruido/` | Evaluacion sobre datos limpios o escenario optimo. |
| `resultados/06_escenario_ruidoso_awgn/` | Evaluacion con ruido AWGN y distintos niveles SNR. |
| `resultados/07_comparacion_temporal_espectral_h3_w/` | Comparacion entre representaciones temporales y espectrales, incluyendo `H3` y `W`. |
| `resultados/08_discusion_resultados/` | Material para interpretar los resultados principales. |

## Comprobacion rapida

En Windows:

```powershell
.\run_all.bat todo
```

Ese comando comprueba dependencias basicas, verifica archivos obligatorios y resume evidencias clave. Tambien se puede ejecutar por partes:

```powershell
python scripts\comprobar_entorno.py
python scripts\verificar_repositorio.py --modo rapido
python scripts\resumen_resultados.py
```

## Git LFS

El repositorio usa Git LFS para ficheros grandes como `.npy`, `.npz`, `.pt`, `.zip`, `.pdf` y resultados pesados. Despues de clonar:

```powershell
git lfs install
git lfs pull
```

La lista de archivos grandes esta en `github/ARCHIVOS_GRANDES_GIT_LFS.csv`.

## Estado de la entrega

- Paquete preparado para GitHub con estructura publica y nombres descriptivos.
- Codigo, resultados, figuras, tablas, configuraciones y manifiestos incluidos.
- Documentacion redactada para lectura autonoma por parte de un tribunal o lector externo.
- Comprobacion automatica disponible mediante `run_all.bat todo`.
