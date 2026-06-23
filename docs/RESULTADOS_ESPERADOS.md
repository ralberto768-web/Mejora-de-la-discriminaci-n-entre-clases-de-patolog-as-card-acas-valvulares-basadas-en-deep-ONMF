# Resultados esperados

Este documento indica que deberia encontrar un lector al revisar cada bloque experimental.

## Optimizacion Deep-ONMF

Ruta principal: `resultados/04_optimizacion_deep_onmf/`

Contenido esperado:

- configuraciones probadas por numero de capas;
- dimensiones empleadas en las matrices internas;
- tablas con mejores configuraciones;
- resultados historicos complementarios en `resultados/04_optimizacion_deep_onmf_historico/`;
- figuras y documentos de apoyo cuando estan disponibles.

## Escenario sin ruido

Ruta principal: `resultados/05_escenario_sin_ruido/`

Contenido esperado:

- resultados de clasificacion sobre datos limpios;
- comparacion con representaciones clasicas cuando existe evidencia disponible;
- metricas agregadas y matrices de confusion;
- ficheros CSV, imagenes y documentos de interpretacion.

## Escenario ruidoso con AWGN

Ruta principal: `resultados/06_escenario_ruidoso_awgn/`

Contenido esperado:

- resultados por nivel de SNR;
- comparacion de robustez frente a ruido;
- tablas de metricas y figuras resumen;
- salidas intermedias necesarias para auditar la evaluacion.

## Comparacion temporal-espectral y H3-W

Ruta principal: `resultados/07_comparacion_temporal_espectral_h3_w/`

Contenido esperado:

- comparacion de representaciones temporales frente a espectrales;
- evidencias de separabilidad mediante nubes de puntos o proyecciones;
- resultados comparando matrices `H3` y `W`;
- tablas CSV con metricas de apoyo.

## Discusion y cierre

Rutas principales:

- `resultados/08_discusion_resultados/`
- `conclusiones/`
- `informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf`

Contenido esperado:

- sintesis de resultados relevantes;
- ventajas y limitaciones de las caracteristicas temporales;
- relacion entre dimension de entrada, coste computacional y rendimiento;
- lineas futuras razonables a partir de la evidencia incluida.
