# Ejemplo básico de clasificación de señales cardíacas

Este repositorio contiene una prueba mínima para el tribunal: un código que lee un fichero de señal cardíaca y devuelve una clasificación básica entre cinco clases: señal sana y cuatro patologías valvulares.

No se incluye aquí el pipeline experimental completo del TFG. La finalidad de esta versión es que cualquier miembro del tribunal pueda descargar el repositorio, ejecutar un ejemplo en pocos segundos y entender el flujo principal.

## Contenido

- `clasificar_senal.py`: script principal. Lee una señal `.csv`, `.txt` o `.wav`, extrae características y clasifica.
- `modelo_basico.json`: parámetros de un clasificador KNN ponderado con referencias reales.
- `datos/audio1.wav` a `datos/audio5.wav`: cinco audios PCG reales de la base preparada, con nombres genéricos para no adelantar la clase.
- `verificar_demo.py`: comprobación rápida de los cinco audios incluidos. En cada ejecución baraja el orden de análisis y al final muestra qué clase era la correcta.
- `docs/EXPLICACION_BASICA.md`: explicación breve del método y de sus limitaciones.

## Instalación

Se recomienda usar Python 3.10 o superior.

```powershell
py -m venv .venv
.\.venv\Scripts\activate
py -m pip install -r requirements.txt
```

Si ya tienes `numpy` instalado, también puedes ejecutar directamente los comandos de uso.

## Uso rápido

Clasificar un audio de ejemplo:

```powershell
py clasificar_senal.py datos\audio1.wav
```

Clasificar otro audio de ejemplo:

```powershell
py clasificar_senal.py datos\audio2.wav
```

Comprobar los cinco audios en orden aleatorio y ver el resumen final de aciertos:

```powershell
py verificar_demo.py
```

## Formato de entrada

El caso más simple es un CSV con estas columnas:

```csv
tiempo_s,amplitud
0.000000,0.0123
0.000125,0.0181
```

También se aceptan:

- CSV o TXT con una sola columna de amplitud. En ese caso se usa `--fs` para indicar la frecuencia de muestreo.
- WAV PCM mono o estéreo.

Ejemplo con frecuencia indicada manualmente:

```powershell
py clasificar_senal.py mi_senal.csv --fs 8000
```

## Qué hace el código

1. Lee el fichero de señal.
2. Normaliza la amplitud y elimina el desplazamiento medio.
3. Calcula características temporales y espectrales básicas.
4. Compara esas características con las referencias reales guardadas en `modelo_basico.json`.
5. Muestra la clase predicha, una confianza aproximada y la distancia al audio de referencia más cercano por clase.

El modelo básico incluido usa las características de los 1000 audios reales preparados en `segmentos_2_0s`: 200 por clase (`N`, `AS`, `MR`, `MS` y `MVP`). Los cinco WAV incluidos en `datos/` son ejemplos reales de 2 segundos, uno por clase. Se nombran como `audio1.wav`, `audio2.wav`, `audio3.wav`, `audio4.wav` y `audio5.wav` para que la clase no se deduzca antes de consultar el resumen final.

## Interpretación

La salida indica la clase con mayor voto ponderado entre los vecinos más cercanos. Una distancia menor significa mayor similitud con esa clase. La confianza es orientativa; sirve para interpretar la demo, no para hacer diagnóstico médico.

## Limitación importante

Este repositorio es una demostración docente y reproducible. No sustituye al sistema experimental completo del TFG, no contiene el entrenamiento completo de Deep ONMF/UjaNet y no debe utilizarse como herramienta clínica.
