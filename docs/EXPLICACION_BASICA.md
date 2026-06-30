# Explicación básica de la demo

## Objetivo

El objetivo de esta prueba es mostrar el flujo mínimo de uso: leer una señal cardíaca almacenada en un fichero y obtener una clasificación automática entre una clase sana y cuatro patologías valvulares.

La demo está pensada para revisión rápida por parte del tribunal. Por ese motivo se evita incluir resultados pesados, bases de datos completas o ejecuciones largas, pero los cinco audios incluidos proceden de la base preparada de 2 segundos usada en el trabajo.

## Entrada

El script acepta tres formatos:

- `.csv` con columnas `tiempo_s,amplitud`.
- `.csv` o `.txt` con una única columna de amplitud.
- `.wav` PCM.

Cuando el fichero contiene tiempo, la frecuencia de muestreo se estima a partir de la separación entre muestras. Cuando solo contiene amplitud, se usa la frecuencia indicada con `--fs`.

## Preprocesado

Antes de clasificar, la señal se convierte a un vector numérico, se eliminan valores no válidos, se resta la media y se normaliza por la amplitud máxima. Con esto se reduce el efecto de escalas distintas entre ficheros.

## Características utilizadas

La clasificación se basa en rasgos sencillos:

- Energía RMS de la señal.
- Tasa de cruces por cero.
- Centroide espectral.
- Relación de energía en bandas asociadas a componentes de soplo.
- Entropía espectral.
- Irregularidad temporal entre tramas.

Estos rasgos no pretenden reemplazar el análisis completo del TFG. Son una versión reducida e interpretable para demostrar la lectura y clasificación de una señal.

## Clasificador

El fichero `modelo_basico.json` guarda las referencias de un clasificador KNN ponderado por distancia para cinco clases: `sana`, `estenosis_aortica`, `regurgitacion_mitral`, `estenosis_mitral` y `prolapso_mitral`.

Estas referencias se han calculado con los 1000 audios reales preparados en `segmentos_2_0s`, con 200 audios por clase: `N`, `AS`, `MR`, `MS` y `MVP`. Los cinco WAV de `datos/` son ejemplos reales de 2 segundos, uno por clase. Sus nombres son genéricos (`audio1.wav` a `audio5.wav`) para que la clase no se conozca antes de ver el resumen final.

El script extrae las características de la señal nueva, las normaliza con los parámetros del modelo y busca sus vecinos más cercanos. La clase asignada es la que acumula mayor peso entre esos vecinos.

## Verificación aleatoria

El fichero `verificar_demo.py` baraja los cinco audios en cada lanzamiento. Durante la ejecución muestra la predicción obtenida para cada audio, sin usar la clase en el nombre del fichero. Al final imprime un resumen con la clase esperada, la clase obtenida y si la clasificación ha sido correcta.

## Relación con el TFG

El TFG completo estudia representaciones temporales y espectrales de señales cardíacas, incluyendo variantes basadas en Deep ONMF. Esta demo conserva la idea general de procesar la señal y clasificarla, pero reduce el sistema a una prueba básica y rápida, siguiendo la indicación de subir solo un ejemplo sencillo al repositorio.

## Uso con una señal propia

Para probar otra señal, guarda el fichero en formato CSV:

```csv
tiempo_s,amplitud
0.000000,0.0012
0.000125,0.0018
```

Después ejecuta:

```powershell
py clasificar_senal.py ruta\a\mi_senal.csv
```

Si el CSV solo tiene amplitudes:

```powershell
py clasificar_senal.py ruta\a\mi_senal.csv --fs 8000
```

## Aviso

La demo no es una herramienta clínica. Su función es explicar el flujo computacional de forma clara y reproducible.
