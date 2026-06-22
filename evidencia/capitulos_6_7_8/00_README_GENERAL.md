# Paquete correcto para capitulos 6, 7 y 8

Este paquete corrige la salida anterior. El indice real usado aqui es:

- 6. Materiales y metodos
- 7. Resultados y discusion
- 8. Conclusiones y lineas futuras

## Relacion punto-carpeta

| Punto | Carpeta | Que contiene |
|---|---|---|
| 6. Materiales y métodos | `06_materiales_y_metodos` | Modelo Deep-ONMF y sistema completo de clasificación con H3 o W. |
| 6.1 Extracción temporal Deep-ONMF | `06_materiales_y_metodos/01_metodo_deep_onmf_w_h` | Desde PCG/matriz tiempo-frecuencia hasta extracción de W y H. |
| 6.2 Arquitectura CNN/UjaNet | `06_materiales_y_metodos/02_arquitectura_ujanet_h3_w` | Arquitectura UjaNet para clasificar valvulopatías usando H3 o W. |
| 7.1 Bases de datos | `07_resultados_y_discusion/07_01_bases_de_datos` | Escenario Yaseen sin ruido y escenario ruidoso AWGN/SNR. |
| 7.2 Métricas | `07_resultados_y_discusion/07_02_metricas` | Accuracy, sensitivity, specificity, precision, score y separabilidad. |
| 7.3 Metodología k-fold | `07_resultados_y_discusion/07_03_metodologia_evaluacion_kfold` | Validación cruzada k-fold, particiones y protocolo. |
| 7.4 Optimización Deep-ONMF | `07_resultados_y_discusion/07_04_optimizacion_deep_onmf` | Capas/dimensiones, mejores configuraciones 1,2,3,4 capas y variación de parámetros. |
| 7.5 Escenario real | `07_resultados_y_discusion/07_05_escenario_real` | Representaciones clásicas y óptimos H3 en Yaseen sin ruido. |
| 7.6 Escenario ruidoso | `07_resultados_y_discusion/07_06_escenario_ruidoso_awgn` | Resultados con SNR/AWGN. |
| 7.7 Espectrales vs temporales | `07_resultados_y_discusion/07_07_espectrales_vs_temporales_h_vs_w` | Superioridad de H frente a W con nubes H3/W y escenarios ruidosos. |
| 7.8 Discusión | `07_resultados_y_discusion/07_08_discusion` | Interpretar menor dimensión/coste temporal y rendimiento frente a clásicas. |
| 8. Conclusiones y líneas futuras | `08_conclusiones_y_lineas_futuras` | Cerrar objetivos, limitaciones y futuro. |

## Advertencias importantes

- En 7.7 no se debe explicar en la memoria en que consiste el enfoque mejorado; se presenta como el enfoque normal.
- En resultados no se debe saturar: los CSV completos quedan aqui y en anexos.
- La comparacion H/W debe separar nubes de puntos, rendimiento y ruido.
- La discusion debe interpretar, no repetir tablas.
