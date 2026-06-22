# Resultados esperados

Este repositorio no se limita a guardar código. También conserva las salidas experimentales generadas durante el trabajo: métricas, matrices de confusión, figuras, tablas, configuraciones y documentos de evaluación.

## Qué se debe poder revisar

Un lector debe poder localizar evidencias para:

- la optimización del modelo Deep-ONMF por número de capas y dimensiones;
- la comparación entre escenario real y escenario ruidoso;
- los resultados con ruido AWGN y distintos niveles SNR;
- la comparación entre características temporales (`H3`) y matriz `W`;
- la comparación con representaciones clásicas;
- la validación cruzada k-fold;
- las matrices de confusión y métricas finales.

## Dónde están las evidencias principales

| Bloque | Ruta |
|---|---|
| Resumen global de resultados | `documento_global/DOCUMENTO_EVALUACION_CAPITULOS_6_7_8.pdf` |
| Manifiesto general de archivos | `evidencia/capitulos_6_7_8/09_manifiestos_verificacion/MANIFIESTO_ARCHIVOS.csv` |
| Optimización Deep-ONMF | `evidencia/capitulos_6_7_8/07_resultados_y_discusion/07_04_optimizacion_deep_onmf/` |
| Escenario sin ruido | `evidencia/capitulos_6_7_8/07_resultados_y_discusion/07_05_escenario_real/` |
| Escenario con ruido AWGN | `evidencia/capitulos_6_7_8/07_resultados_y_discusion/07_06_escenario_ruidoso_awgn/` |
| Comparación H3 frente a W | `evidencia/capitulos_6_7_8/07_resultados_y_discusion/07_07_espectrales_vs_temporales_h_vs_w/` |

## Comprobación automática

Para imprimir un resumen de resultados clave:

```powershell
python scripts\resumen_resultados.py
```

Para comprobar estructura y manifiesto:

```powershell
python scripts\verificar_repositorio.py --modo rapido
```

## Interpretación correcta

Los resultados deben interpretarse separando:

- rendimiento de clasificación;
- separación visual o geométrica de las características;
- robustez frente a ruido;
- coste computacional y dimensionalidad de las representaciones.

No debe asumirse que Deep-ONMF gana en todos los casos. El objetivo del repositorio es conservar la evidencia completa para que esa interpretación se pueda comprobar.

