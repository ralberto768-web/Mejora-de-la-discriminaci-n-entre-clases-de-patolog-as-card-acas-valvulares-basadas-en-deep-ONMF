# Mejora de la discriminación entre clases de patologías cardíacas valvulares basadas en deep-ONMF

Este repositorio contiene la implementación, los resultados experimentales y la evidencia documental de un Trabajo Fin de Grado centrado en el análisis automático de sonidos cardíacos.

El objetivo del trabajo es estudiar si las características temporales obtenidas mediante Deep-ONMF pueden mejorar la discriminación entre clases de patologías cardíacas valvulares. Para ello se extraen representaciones a partir de señales de fonocardiograma, se comparan matrices internas del modelo como `H`/`H3` y `W`, y se evalúan clasificadores como UjaNet y otros modelos clásicos en escenarios sin ruido y con ruido AWGN.

## Qué problema aborda

Las valvulopatías son alteraciones de las válvulas cardíacas que pueden modificar el sonido del corazón. El trabajo parte de grabaciones de sonidos cardíacos y aplica procesado de señal y aprendizaje automático para clasificar diferentes clases patológicas.

La idea principal es transformar cada audio en representaciones útiles para clasificación. En este repositorio se conserva el flujo completo usado para:

- extraer características temporales con Deep-ONMF;
- generar matrices `W`, `H` y especialmente `H3`;
- comparar características temporales frente a representaciones espectrales clásicas;
- evaluar el modelo en escenario real y escenario ruidoso con AWGN;
- guardar tablas, métricas, figuras, matrices de confusión y manifiestos de verificación.

## Cómo leer este repositorio

Si solo quieres entender qué se ha hecho, empieza por:

1. `docs/LECTURA_RAPIDA.md`
2. `documento_global/DOCUMENTO_EVALUACION_CAPITULOS_6_7_8.pdf`
3. `docs/MAPA_INDICE_TFG.md`
4. `docs/RESULTADOS_ESPERADOS.md`

Si quieres comprobar que los archivos incluidos no están corruptos:

```powershell
.\run_all.bat todo
```

Si quieres intentar reproducir los experimentos desde cero, lee primero:

```text
docs/DATOS_EXTERNOS.md
docs/REPRODUCIBILIDAD.md
```

La reproducción completa requiere disponer de las bases de datos externas de sonidos cardíacos. Los audios fuente no se incluyen directamente en el repositorio por tamaño y posibles restricciones de licencia.

## Estructura del repositorio

| Ruta | Qué contiene | Para qué sirve |
|---|---|---|
| `documento_global/` | PDF y Markdown con una visión integrada de materiales, métodos, resultados, discusión y conclusiones. | Lectura rápida para revisar el trabajo sin navegar por todas las carpetas. |
| `evidencia/capitulos_6_7_8/06_materiales_y_metodos/` | Código y configuraciones relacionados con Deep-ONMF, extracción de `W`/`H` y arquitectura UjaNet. | Explicar el método propuesto y el sistema de clasificación. |
| `evidencia/capitulos_6_7_8/07_resultados_y_discusion/` | Resultados, métricas, tablas, figuras y experimentos organizados por apartado. | Revisar la optimización, el escenario real, el escenario ruidoso y la comparación `H` frente a `W`. |
| `evidencia/capitulos_6_7_8/08_conclusiones_y_lineas_futuras/` | Material de cierre, documentos finales y evidencias para conclusiones. | Relacionar resultados con objetivos y líneas futuras. |
| `evidencia/capitulos_6_7_8/09_manifiestos_verificacion/` | Manifiestos con rutas, tamaños y hashes. | Auditar que la copia de evidencia coincide con los archivos generados. |
| `docs/` | Guías para lectores externos, tribunal, datos externos, reproducibilidad e índice del TFG. | Entender y evaluar el repositorio sin depender de explicaciones externas. |
| `scripts/` | Comprobaciones automáticas de entorno, integridad y resumen de resultados. | Verificar que el paquete descargado está completo. |
| `datos_externos/` | Plantilla para colocar bases de datos que no se suben al repositorio. | Preparar una reproducción completa con datos originales. |
| `github/` | Manifiestos del repositorio, lista de archivos grandes y notas de subida. | Control de Git/Git LFS y trazabilidad de la entrega. |

## Bloques experimentales principales

Dentro de `evidencia/capitulos_6_7_8/07_resultados_y_discusion/` se conservan los resultados separados así:

- `07_01_bases_de_datos`: descripción y auditoría de los escenarios usados.
- `07_02_metricas`: métricas de clasificación y separabilidad.
- `07_03_metodologia_evaluacion_kfold`: particiones y validación cruzada k-fold.
- `07_04_optimizacion_deep_onmf`: búsqueda de configuraciones por capas y dimensiones.
- `07_05_escenario_real`: comparación en datos sin ruido.
- `07_06_escenario_ruidoso_awgn`: comparación con ruido AWGN y distintos niveles SNR.
- `07_07_espectrales_vs_temporales_h_vs_w`: comparación entre características espectrales y temporales, incluyendo `H3` frente a `W`.
- `07_08_discusion`: material para interpretar los resultados principales.

## Comprobación rápida

En Windows:

```powershell
.\run_all.bat todo
```

Ese comando:

1. comprueba que las dependencias principales están disponibles;
2. verifica que los archivos obligatorios existen y coinciden con el manifiesto;
3. imprime un resumen de resultados y evidencias clave.

También se puede ejecutar manualmente:

```powershell
python scripts\comprobar_entorno.py
python scripts\verificar_repositorio.py --modo rapido
python scripts\resumen_resultados.py
```

## Reproducción completa

Este repositorio permite revisar los resultados ya generados y preparar una reproducción de los experimentos. Para repetir todo desde cero hacen falta:

- Python y las dependencias de `requirements.txt`;
- las bases de datos externas colocadas en `datos_externos/`;
- las particiones, configuraciones y scripts incluidos en `evidencia/`;
- Git LFS para descargar correctamente los archivos grandes.

Consulta `docs/REPRODUCIBILIDAD.md` antes de lanzar ejecuciones largas.

## Archivos grandes y Git LFS

El repositorio usa Git LFS para ficheros grandes como `.npy`, `.npz`, `.pt`, `.zip` y documentos pesados. Después de clonar:

```powershell
git lfs install
git lfs pull
```

La lista de archivos grandes está en:

```text
github/ARCHIVOS_GRANDES_GIT_LFS.csv
```

## Estado de la entrega

- Archivos de evidencia comparados fuente/destino: `66.830`
- Entradas del manifiesto del repositorio: `66.852`
- Tamaño aproximado del paquete: `8.13 GB`
- Objetos Git LFS subidos: `886`
- Commit inicial de entrega: `aec5198`

