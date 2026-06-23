# Informe general de resultados Deep-ONMF

## 1. Proposito del repositorio

Este informe resume el material incluido en el repositorio para evaluar un sistema de clasificacion de sonidos cardiacos basado en Deep-ONMF. El objetivo es revisar de forma ordenada el metodo, las evidencias experimentales y las conclusiones tecnicas sin depender de rutas locales ni de documentos privados.

## 2. Metodo Deep-ONMF y clasificacion

El flujo experimental parte de senales de fonocardiograma. A partir de ellas se extraen caracteristicas temporales mediante Deep-ONMF, obteniendo matrices internas como `W`, `H` y `H3`. La representacion `H3` se usa como entrada temporal principal para el sistema de clasificacion, mientras que `W` se conserva para comparaciones de separabilidad y rendimiento.

El repositorio conserva codigo, configuraciones y resultados intermedios en `metodologia/`. Esta carpeta debe revisarse para entender como se generan las representaciones y como se conectan con la arquitectura de clasificacion.

## 3. Bases de datos y escenarios

La evaluacion distingue dos condiciones principales. La primera corresponde a un escenario sin ruido, usado como referencia de funcionamiento. La segunda introduce ruido AWGN con distintos niveles SNR para estudiar robustez. La evidencia asociada esta organizada en `resultados/01_bases_de_datos_y_escenarios/`, `resultados/05_escenario_sin_ruido/` y `resultados/06_escenario_ruidoso_awgn/`.

Los audios fuente completos no se distribuyen en GitHub. La reproduccion desde cero requiere colocarlos en `datos_externos/` siguiendo la guia correspondiente.

## 4. Metricas y metodologia de evaluacion

Las salidas experimentales incluyen metricas de clasificacion como accuracy, precision, recall y F1-score, junto con matrices de confusion y tablas resumen cuando estan disponibles. La evaluacion se organiza con validacion cruzada k-fold para reducir dependencia de una unica particion.

Las carpetas relevantes son `resultados/02_metricas_de_evaluacion/` y `resultados/03_validacion_cruzada_kfold/`.

## 5. Optimizacion Deep-ONMF

La optimizacion estudia el efecto del numero de capas y de las dimensiones internas del modelo. La evidencia principal se encuentra en `resultados/04_optimizacion_deep_onmf/`. Las ejecuciones historicas complementarias se conservan en `resultados/04_optimizacion_deep_onmf_historico/` para trazabilidad.

En la revision de resultados conviene separar dos aspectos: por un lado, las configuraciones objetivamente evaluadas; por otro, la interpretacion sobre que configuracion es mas defendible para el sistema final.

## 6. Resultados en escenario sin ruido

El escenario sin ruido permite comparar el comportamiento del enfoque temporal Deep-ONMF frente a representaciones clasicas. La evidencia se guarda en `resultados/05_escenario_sin_ruido/` e incluye tablas, figuras y salidas de clasificacion cuando estan disponibles.

Este bloque sirve como referencia para valorar si la representacion temporal conserva informacion discriminativa suficiente con menor dimension de entrada.

## 7. Resultados en escenario ruidoso

El escenario con AWGN analiza la estabilidad del sistema ante degradacion artificial de la senal. La carpeta `resultados/06_escenario_ruidoso_awgn/` agrupa resultados por nivel de ruido y permite comparar el comportamiento relativo de las representaciones.

La interpretacion debe centrarse en tendencias de robustez y no solo en un valor puntual de una metrica.

## 8. Comparacion temporal-espectral y H3-W

La carpeta `resultados/07_comparacion_temporal_espectral_h3_w/` contiene la evidencia usada para comparar caracteristicas temporales frente a espectrales y para contrastar el uso de `H3` frente a `W`.

Esta comparacion tiene dos lecturas complementarias: representabilidad visual mediante proyecciones o nubes de puntos, y rendimiento cuantitativo mediante metricas de clasificacion. Ambas lecturas deben mantenerse separadas para evitar conclusiones excesivas.

## 9. Discusion tecnica

Los resultados deben interpretarse considerando rendimiento, dimension de entrada, coste computacional y robustez. Aunque las representaciones temporales no siempre superen a las representaciones clasicas en todas las metricas, pueden aportar una entrada de menor dimension y un coste potencialmente menor.

La carpeta `resultados/08_discusion_resultados/` conserva material util para esta lectura.

## 10. Conclusiones y lineas futuras

El repositorio permite defender un flujo completo de extraccion temporal, clasificacion y evaluacion bajo condiciones limpias y ruidosas. Las lineas futuras pasan por ampliar bases de datos, refinar la optimizacion de capas y dimensiones, estudiar otras arquitecturas de clasificacion y mejorar la validacion externa.

El material de cierre esta en `conclusiones/`.

## 11. Verificacion

La integridad del paquete puede comprobarse con:

```powershell
.\run_all.bat todo
```

Los manifiestos estan en `github/` y `verificacion/`. El objetivo de esas comprobaciones no es reentrenar modelos, sino confirmar que el repositorio conserva los archivos obligatorios y las evidencias principales.
