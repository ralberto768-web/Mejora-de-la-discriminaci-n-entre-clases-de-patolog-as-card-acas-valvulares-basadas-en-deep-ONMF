# Guía para revisión académica

Este repositorio contiene código, resultados y evidencias del Trabajo Fin de Grado:

**Mejora de la discriminación entre clases de patologías cardíacas valvulares basadas en deep-ONMF**

La finalidad del repositorio es que el tribunal o cualquier lector externo pueda revisar qué se ha implementado, qué resultados se han obtenido y qué material permite comprobarlos.

## Nivel 1: entender el trabajo sin ejecutar código

Abrir en este orden:

1. `README.md`
2. `docs/LECTURA_RAPIDA.md`
3. `documento_global/DOCUMENTO_EVALUACION_CAPITULOS_6_7_8.pdf`
4. `docs/MAPA_INDICE_TFG.md`

Con esto se entiende:

- cuál es el problema tratado;
- qué papel tiene Deep-ONMF;
- qué son las matrices `W`, `H` y `H3`;
- cómo se organiza la evidencia;
- qué resultados corresponden a cada apartado de la memoria.

## Nivel 2: revisar resultados ya generados

Los resultados están en:

```text
evidencia/capitulos_6_7_8/07_resultados_y_discusion/
```

Bloques principales:

- `07_04_optimizacion_deep_onmf`: configuraciones por capas y dimensiones.
- `07_05_escenario_real`: resultados sin ruido.
- `07_06_escenario_ruidoso_awgn`: resultados con ruido AWGN/SNR.
- `07_07_espectrales_vs_temporales_h_vs_w`: comparación `H3` frente a `W` y características temporales frente a espectrales.

El documento global resume qué evidencia usar en cada bloque:

```text
documento_global/DOCUMENTO_EVALUACION_CAPITULOS_6_7_8.pdf
```

## Nivel 3: comprobar integridad del repositorio

En Windows:

```powershell
.\run_all.bat todo
```

Este comando comprueba entorno, archivos obligatorios y resultados principales.

También puede ejecutarse de forma separada:

```powershell
python scripts\comprobar_entorno.py
python scripts\verificar_repositorio.py --modo rapido
python scripts\resumen_resultados.py
```

Para una comprobación más estricta de archivos pequeños y de control:

```powershell
python scripts\verificar_repositorio.py --modo completo
```

Los archivos grandes se gestionan con Git LFS y se documentan en:

```text
github/ARCHIVOS_GRANDES_GIT_LFS.csv
```

## Nivel 4: reproducir experimentos

La reproducción completa requiere disponer de los datos externos originales, que deben colocarse en:

```text
datos_externos/
```

Antes de ejecutar experimentos largos, revisar:

```text
docs/DATOS_EXTERNOS.md
docs/REPRODUCIBILIDAD.md
```

El repositorio conserva particiones, configuraciones, scripts, métricas y resultados para comparar una nueva ejecución con los resultados esperados.

