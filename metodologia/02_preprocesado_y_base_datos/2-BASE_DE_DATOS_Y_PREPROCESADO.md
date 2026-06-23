# 2-BASE_DE_DATOS_Y_PREPROCESADO

## Estructura de la base de datos

La carpeta `Bases de Datos` contiene cinco carpetas de audios:

- `N_New_3주기`
- `AS_New_3주기`
- `MR_New_3주기`
- `MS_New_3주기`
- `MVP_New_3주기`

Cada clase contiene 200 archivos `.wav`, por tanto hay 1000 audios en total.

## Lectura de audio

Cada archivo se lee con Python usando el módulo estándar `wave`. La señal se convierte a valores reales normalizados:

```text
int16 -> valor / 32768
```

La base auditada usa:

- 1 canal.
- 8000 Hz.
- 16 bits por muestra.

## División en tramas

Según el artículo:

1. Cada audio se divide en ventanas de 2 segundos.
2. Entre ventanas se usa 1 segundo de solape.
3. Si un audio no tiene duración suficiente para formar una trama completa de 2 segundos, el artículo indica que se descarta.

En esta implementación final para el TFG no se descarta ningún archivo de la base de datos. Si un audio dura menos de 2 segundos, se rellena con ceros hasta llegar a 2 segundos. Esta decisión se toma porque la base disponible es la que se va a usar y se quiere conservar el 100% de las muestras.

## Espectrograma

Cada trama de 2 segundos se transforma en espectrograma:

```text
ventana Hamming = 150 muestras
salto = 75 muestras
FFT = 250 puntos
bins de frecuencia = 126
```

El espectrograma usado por la factorización es la magnitud de la FFT, que siempre es no negativa. Esto es necesario porque ONMF trabaja con matrices no negativas.

## Matriz X por clase

Para cada clase se concatenan los espectrogramas por columnas:

```text
X_N
X_AS
X_MR
X_MS
X_MVP
```

Cada matriz tiene 126 filas, una por bin de frecuencia. El número de columnas depende de cuántas tramas válidas se hayan podido formar con los audios reales.
