# Explicacion basica de la demo

## Objetivo

El objetivo de esta prueba es mostrar el flujo minimo de uso: leer una senal cardiaca almacenada en un fichero y obtener una clasificacion automatica entre una clase sana y cuatro patologias valvulares.

La demo esta pensada para revision rapida por parte del tribunal. Por ese motivo se evita incluir resultados pesados, bases de datos completas o ejecuciones largas.

## Entrada

El script acepta tres formatos:

- `.csv` con columnas `tiempo_s,amplitud`.
- `.csv` o `.txt` con una unica columna de amplitud.
- `.wav` PCM.

Cuando el fichero contiene tiempo, la frecuencia de muestreo se estima a partir de la separacion entre muestras. Cuando solo contiene amplitud, se usa la frecuencia indicada con `--fs`.

## Preprocesado

Antes de clasificar, la senal se convierte a un vector numerico, se eliminan valores no validos, se resta la media y se normaliza por la amplitud maxima. Con esto se reduce el efecto de escalas distintas entre ficheros.

## Caracteristicas utilizadas

La clasificacion se basa en rasgos sencillos:

- Energia RMS de la senal.
- Tasa de cruces por cero.
- Centroide espectral.
- Relacion de energia en bandas asociadas a componentes de soplo.
- Entropia espectral.
- Irregularidad temporal entre tramas.

Estos rasgos no pretenden reemplazar el analisis completo del TFG. Son una version reducida e interpretable para demostrar la lectura y clasificacion de una senal.

## Clasificador

El fichero `modelo_basico.json` guarda cinco prototipos: `sana`, `estenosis_aortica`, `regurgitacion_mitral`, `estenosis_mitral` y `prolapso_mitral`.

El script extrae las caracteristicas de la senal nueva, las normaliza con los parametros del modelo y calcula la distancia a cada prototipo. La clase asignada es la que queda mas cerca.

## Relacion con el TFG

El TFG completo estudia representaciones temporales y espectrales de senales cardiacas, incluyendo variantes basadas en Deep ONMF. Esta demo conserva la idea general de procesar la senal y clasificarla, pero reduce el sistema a una prueba basica y rapida, siguiendo la indicacion de subir solo un ejemplo sencillo al repositorio.

## Uso con una senal propia

Para probar otra senal, guarda el fichero en formato CSV:

```csv
tiempo_s,amplitud
0.000000,0.0012
0.000125,0.0018
```

Despues ejecuta:

```powershell
py clasificar_senal.py ruta\a\mi_senal.csv
```

Si el CSV solo tiene amplitudes:

```powershell
py clasificar_senal.py ruta\a\mi_senal.csv --fs 8000
```

## Aviso

La demo no es una herramienta clinica. Su funcion es explicar el flujo computacional de forma clara y reproducible.
