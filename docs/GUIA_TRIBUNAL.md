# Guia para evaluacion externa

Esta guia esta pensada para una revision rapida por parte de un tribunal, tutor o lector externo. El repositorio puede revisarse sin ejecutar entrenamientos completos, porque incluye resultados ya generados, tablas, figuras y manifiestos de verificacion.

## Revision recomendada

1. Leer `README.md` para entender el objetivo y la estructura.
2. Abrir `informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf` para una vision completa del metodo y los resultados.
3. Consultar `docs/RESULTADOS_ESPERADOS.md` para saber donde esta cada tipo de evidencia.
4. Revisar `resultados/04_optimizacion_deep_onmf/`, `resultados/05_escenario_sin_ruido/`, `resultados/06_escenario_ruidoso_awgn/` y `resultados/07_comparacion_temporal_espectral_h3_w/`.
5. Ejecutar `run_all.bat todo` si se quiere comprobar que el repositorio descargado conserva los archivos obligatorios.

## Que se puede comprobar directamente

- Codigo y configuraciones del flujo Deep-ONMF.
- Resultados de optimizacion por capas y dimensiones.
- Resultados en escenario sin ruido.
- Resultados en escenario con ruido AWGN.
- Comparacion entre representaciones temporales y espectrales.
- Comparacion entre matrices `H3` y `W`.
- Figuras, tablas, matrices de confusion, CSV de metricas y documentos explicativos.

## Que no se incluye directamente

Los audios fuente y bases de datos completas no se suben al repositorio por tamano y por posibles restricciones externas. Para repetir ejecuciones completas deben colocarse manualmente en `datos_externos/`, siguiendo `docs/DATOS_EXTERNOS.md`.

## Separacion entre evidencia e interpretacion

- Las carpetas de `resultados/` contienen evidencia experimental: metricas, tablas, figuras y salidas generadas.
- `informe_general/` y `docs/` contienen la explicacion y la orientacion de lectura.
- `conclusiones/` contiene material de cierre y lineas futuras.

Esta separacion facilita revisar los datos primero y valorar la interpretacion despues.
