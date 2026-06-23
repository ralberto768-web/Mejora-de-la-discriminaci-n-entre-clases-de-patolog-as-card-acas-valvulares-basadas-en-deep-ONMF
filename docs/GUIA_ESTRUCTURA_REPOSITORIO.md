# Guia de estructura del repositorio

Esta guia explica que contiene cada bloque del repositorio y para que se usa. Los nombres estan pensados para lectura publica: cada carpeta describe su funcion tecnica sin depender de rutas locales ni de documentos privados.

## Bloques principales

| Carpeta | Funcion |
|---|---|
| `metodologia/` | Material tecnico del metodo: Deep-ONMF, extraccion de matrices `W` y `H`, configuraciones y arquitectura de clasificacion. |
| `resultados/` | Evidencia experimental organizada por bases de datos, metricas, validacion, optimizacion, escenarios y comparaciones. |
| `conclusiones/` | Material de cierre: objetivos alcanzados, limitaciones y lineas futuras. |
| `informe_general/` | Informe PDF/Markdown que integra metodologia, resultados y discusion en un unico documento. |
| `docs/` | Guias de lectura para un evaluador externo. |
| `scripts/` | Comprobaciones automaticas y resumen de archivos clave. |
| `datos_externos/` | Carpeta preparada para colocar bases de datos externas no distribuidas en GitHub. |
| `verificacion/` | Manifiestos de integridad de evidencia experimental. |
| `github/` | Estado del paquete, manifiesto del repositorio y archivos gestionados con Git LFS. |

## Resultados experimentales

| Carpeta | Contenido esperado |
|---|---|
| `resultados/01_bases_de_datos_y_escenarios/` | Descripcion y auditoria de bases de datos, escenarios limpios y escenarios con AWGN. |
| `resultados/02_metricas_de_evaluacion/` | Accuracy, precision, recall, F1-score, matrices de confusion y metricas de separabilidad cuando aparecen. |
| `resultados/03_validacion_cruzada_kfold/` | Particiones y salidas de validacion cruzada. |
| `resultados/04_optimizacion_deep_onmf/` | Mejores configuraciones Deep-ONMF por numero de capas y dimensiones. |
| `resultados/04_optimizacion_deep_onmf_historico/` | Ejecuciones historicas complementarias conservadas para trazabilidad. |
| `resultados/05_escenario_sin_ruido/` | Resultados de clasificacion en condiciones limpias. |
| `resultados/06_escenario_ruidoso_awgn/` | Resultados bajo ruido AWGN y diferentes niveles SNR. |
| `resultados/07_comparacion_temporal_espectral_h3_w/` | Comparacion entre caracteristicas temporales y espectrales, con enfasis en `H3` frente a `W`. |
| `resultados/08_discusion_resultados/` | Graficas, tablas y notas utiles para interpretar los resultados. |

## Lectura tecnica sugerida

1. Revisar `metodologia/README.md` para entender el flujo de extraccion y clasificacion.
2. Revisar `resultados/README.md` y despues cada subcarpeta de resultados.
3. Consultar el informe `informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf` para ver una narrativa unificada.
4. Ejecutar `run_all.bat todo` para comprobar que la descarga conserva los archivos principales.
