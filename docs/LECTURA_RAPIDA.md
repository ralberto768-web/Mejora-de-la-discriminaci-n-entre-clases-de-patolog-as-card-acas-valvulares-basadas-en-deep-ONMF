# Lectura rapida

Este documento resume como leer el repositorio sin conocer la organizacion interna usada durante el desarrollo del TFG.

## Objetivo del repositorio

El proyecto evalua un flujo de clasificacion de sonidos cardiacos basado en caracteristicas temporales extraidas con Deep-ONMF. La evidencia incluida permite revisar el metodo, los resultados experimentales, la comparacion entre representaciones y la reproducibilidad basica del paquete.

## Orden recomendado de lectura

1. `README.md`: vision general del proyecto y estructura de carpetas.
2. `informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf`: informe integrado con metodologia, resultados y discusion.
3. `docs/GUIA_ESTRUCTURA_REPOSITORIO.md`: relacion entre carpetas y bloques tecnicos.
4. `docs/RESULTADOS_ESPERADOS.md`: que deberia encontrar el lector en cada bloque de resultados.
5. `docs/REPRODUCIBILIDAD.md`: como comprobar el paquete y que hace falta para repetir ejecuciones completas.

## Que revisar primero

- Metodologia Deep-ONMF y clasificacion: `metodologia/`.
- Resultados de optimizacion: `resultados/04_optimizacion_deep_onmf/` y `resultados/04_optimizacion_deep_onmf_historico/`.
- Escenario sin ruido: `resultados/05_escenario_sin_ruido/`.
- Escenario con ruido AWGN: `resultados/06_escenario_ruidoso_awgn/`.
- Comparacion `H3` frente a `W`: `resultados/07_comparacion_temporal_espectral_h3_w/`.
- Integridad de archivos: `verificacion/` y `github/MANIFIESTO_REPOSITORIO.csv`.

## Comprobacion minima

Ejecuta desde la raiz del repositorio:

```powershell
.\run_all.bat todo
```

Si el comando termina con verificacion correcta, la descarga contiene los archivos obligatorios y las evidencias principales estan localizables.
