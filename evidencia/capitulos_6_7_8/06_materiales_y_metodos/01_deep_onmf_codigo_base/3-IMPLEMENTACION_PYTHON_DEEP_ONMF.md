# 3-IMPLEMENTACION_PYTHON_DEEP_ONMF

## Archivos de código

El código está en la carpeta `src\tfg_deep_onmf`:

- `configuracion.py`: parámetros generales del experimento.
- `audio.py`: lectura de WAV, tramas y espectrogramas.
- `onmf.py`: implementación de deep ONMF.
- `estadistica.py`: Tabla 2, distancias y características por audio.
- `graficos.py`: Figura 5, Tabla 2, Figura 7 y Figura 11D.
- `pipeline.py`: ejecución completa y guardado de resultados.

El punto de entrada es:

```text
ejecutar_pipeline.py
```

## Deep ONMF implementado

La implementación aplica tres factorizaciones:

```text
X  ≈ W1 * H1
H1 ≈ W2 * H2
H2 ≈ W3 * H3
```

Con rangos:

```text
W1: 126 x 9
W2: 9 x 8
W3: 8 x 7
W_final: 126 x 7
```

`W_final` contiene los siete SBV finales de cada clase.

## Ortogonalidad

El artículo exige una restricción de ortogonalidad sobre `H`. En código se introduce como penalización:

```text
||H * H.T - I||
```

La actualización sigue siendo no negativa, porque todos los factores `W` y `H` se mantienen con valores mayores o iguales a cero.

## Características por audio

Además de `W_final`, el programa usa `H3` para obtener una representación por audio:

1. Cada audio ocupa un rango de columnas en la matriz `X`.
2. Ese mismo rango existe en `H3`.
3. Se calcula la media de las columnas de `H3` de cada audio.
4. Así cada audio queda representado por siete valores: `SBV_1` a `SBV_7`.

Estas características por audio se guardan en:

```text
caracteristicas_por_audio.csv
```

Se usan para generar la Figura 7 y la Figura 11D.

## Reproducibilidad

Cada prueba guarda:

- Semilla aleatoria.
- Parámetros de STFT.
- Rangos de ONMF.
- Iteraciones.
- Penalización de ortogonalidad.
- Auditoría de audios usados.
- Matrices `W_final`.
- Características por audio.

Esto permite repetir y revisar el experimento.
