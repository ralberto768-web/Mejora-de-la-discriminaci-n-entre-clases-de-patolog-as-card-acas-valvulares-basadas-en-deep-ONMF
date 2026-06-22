# Documento de evaluacion capitulos 6, 7 y 8

Este documento corrige la salida anterior y organiza punto a punto que material corresponde a cada parte del indice.

| Punto | Contenido | Carpeta | Evidencia | Como evaluarlo |
|---|---|---|---|---|
| 6. Materiales y métodos | Modelo Deep-ONMF y sistema completo de clasificación con H3 o W. | `06_materiales_y_metodos` | Codigos Deep-ONMF, pipeline, UjaNet, configuraciones y scripts ULTIMA. | Debe ser visual, paso a paso y no mezclar optimizacion posterior como si fuera el metodo base. |
| 6.1 Extracción temporal Deep-ONMF | Desde PCG/matriz tiempo-frecuencia hasta extracción de W y H. | `06_materiales_y_metodos/01_metodo_deep_onmf_w_h` | audio.py, onmf.py, pipeline.py, representaciones.py, configuraciones. | Explicar W como bases/filtros y H como activaciones temporales; centrar la memoria en H3/W. |
| 6.2 Arquitectura CNN/UjaNet | Arquitectura UjaNet para clasificar valvulopatías usando H3 o W. | `06_materiales_y_metodos/02_arquitectura_ujanet_h3_w` | clasificadores.py, evaluacion.py, entrada_ujanet, historiales y modelos. | Decir que después se optimizan capas/dimensiones, pero el esquema base usa H3/W. |
| 7.1 Bases de datos | Escenario Yaseen sin ruido y escenario ruidoso AWGN/SNR. | `07_resultados_y_discusion/07_01_bases_de_datos` | auditorias, particiones, validacion_protocolo y datasets ruidosos sin WAV. | Describir escenarios sin saturar con archivos fuente. |
| 7.2 Métricas | Accuracy, sensitivity, specificity, precision, score y separabilidad. | `07_resultados_y_discusion/07_02_metricas` | metricas/*.csv, comparacion_estadistica_w_h3.csv, metricas_separabilidad.csv. | Separar métricas de clasificación de métricas visuales/separabilidad. |
| 7.3 Metodología k-fold | Validación cruzada k-fold, particiones y protocolo. | `07_resultados_y_discusion/07_03_metodologia_evaluacion_kfold` | particiones_5fold.csv, configuracion_usada.json, clasificadores/*/fold_*. | Dejar claro qué se entrena y evalúa en cada fold. |
| 7.4 Optimización Deep-ONMF | Capas/dimensiones, mejores configuraciones 1,2,3,4 capas y variación de parámetros. | `07_resultados_y_discusion/07_04_optimizacion_deep_onmf` | Programacion objetivo/resultados, pruebas Juan, FINAL/RESULTADOS, ULTIMA 01/03. | No saturar: usar tablas resumen y remitir completos a anexos. |
| 7.5 Escenario real | Representaciones clásicas y óptimos H3 en Yaseen sin ruido. | `07_resultados_y_discusion/07_05_escenario_real` | resultados_validacion_modelo_optimo, punto3_validacion, informes y matrices. | Comparar sin afirmar que Deep-ONMF gana todo si no lo sostienen las métricas. |
| 7.6 Escenario ruidoso | Resultados con SNR/AWGN. | `07_resultados_y_discusion/07_06_escenario_ruidoso_awgn` | ULTIMA 02/04/06, informe AWGN. | Analizar robustez y variación con ruido. |
| 7.7 Espectrales vs temporales | Superioridad de H frente a W con nubes H3/W y escenarios ruidosos. | `07_resultados_y_discusion/07_07_espectrales_vs_temporales_h_vs_w` | figuras H/W, prueba NMF/ONMF, Matriz W, comparacion final. | No explicar el enfoque mejorado en la memoria: tratar H3 mejorado como normal. |
| 7.8 Discusión | Interpretar menor dimensión/coste temporal y rendimiento frente a clásicas. | `07_resultados_y_discusion/07_08_discusion` | informes modelo óptimo, figuras resumen, comparativas. | Discusión = interpretación, no repetir tablas. |
| 8. Conclusiones y líneas futuras | Cerrar objetivos, limitaciones y futuro. | `08_conclusiones_y_lineas_futuras` | documentos finales, matriz de relación y anexos. | Debe cerrar 6 y 7, sin meter resultados nuevos. |

## Resumen de copia

- Archivos copiados/verificados: 66771/66771
- Errores: 0
- Manifiesto: `Codigos y resultados para la memoria\CAPITULOS_6_7_8_20260622_1344\09_manifiestos_verificacion\MANIFIESTO_ARCHIVOS.csv`
