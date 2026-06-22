# Programacion de la prueba para Juan

Esta carpeta contiene el codigo necesario para repetir la prueba pedida para el 26/05/2026.

## Archivos

- `01_ejecutar_prueba_inicializaciones.py`: ejecuta Deep ONMF con cuatro inicializaciones: aleatoria actual, NNDSVD, NNDSVDa y NNDSVDar.
- `02_EJECUTAR_PRUEBA_JUAN_INICIALIZACIONES.bat`: acceso directo para lanzarlo en Windows.
- `requirements_prueba_juan.txt`: librerias necesarias.
- `src_tfg_deep_onmf_usado/`: copia del codigo base del proyecto que se ha usado como apoyo.

## Que cambia respecto al Deep ONMF actual

En el codigo original, `W` y `H` se inicializan asi:

```python
w = rng.random((filas, rango)) + eps
h = rng.random((rango, columnas)) + eps
```

En esta prueba se conserva esa version como referencia y se comparan tres inicializaciones comunes:

- `NNDSVD`
- `NNDSVDa`
- `NNDSVDar`

Los resultados se guardan en:

`C:\Users\armga\OneDrive\Escritorio\TFG\Programacion objetivo\pruebas para el 26-05-26\resultados comparativos`
