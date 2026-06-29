# Ejemplo basico de clasificacion de señales cardiacas

Este repositorio contiene una prueba minima para el tribunal: un codigo que lee un fichero de senal cardiaca y devuelve una clasificacion basica entre cinco clases: senal sana y cuatro patologias valvulares.

No se incluye aqui el pipeline experimental completo del TFG. La finalidad de esta version es que cualquier miembro del tribunal pueda descargar el repositorio, ejecutar un ejemplo en pocos segundos y entender el flujo principal.

## Contenido

- `clasificar_senal.py`: script principal. Lee una senal `.csv`, `.txt` o `.wav`, extrae caracteristicas y clasifica.
- `modelo_basico.json`: parametros de un clasificador sencillo por prototipos.
- `datos/pcg_sano.wav`: audio PCG sintetico con patron sano.
- `datos/pcg_estenosis_aortica.wav`: audio PCG sintetico con estenosis aortica.
- `datos/pcg_regurgitacion_mitral.wav`: audio PCG sintetico con regurgitacion mitral.
- `datos/pcg_estenosis_mitral.wav`: audio PCG sintetico con estenosis mitral.
- `datos/pcg_prolapso_mitral.wav`: audio PCG sintetico con prolapso mitral.
- `verificar_demo.py`: comprobacion rapida de los cinco audios incluidos.
- `docs/EXPLICACION_BASICA.md`: explicacion breve del metodo y de sus limitaciones.

## Instalacion

Se recomienda usar Python 3.10 o superior.

```powershell
py -m venv .venv
.\.venv\Scripts\activate
py -m pip install -r requirements.txt
```

Si ya tienes `numpy` instalado, tambien puedes ejecutar directamente los comandos de uso.

## Uso rapido

Clasificar la senal sana:

```powershell
py clasificar_senal.py datos\pcg_sano.wav
```

Clasificar una patologia concreta:

```powershell
py clasificar_senal.py datos\pcg_estenosis_aortica.wav
```

Comprobar los cinco audios de una vez:

```powershell
py verificar_demo.py
```

## Formato de entrada

El caso mas simple es un CSV con estas columnas:

```csv
tiempo_s,amplitud
0.000000,0.0123
0.000125,0.0181
```

Tambien se aceptan:

- CSV o TXT con una sola columna de amplitud. En ese caso se usa `--fs` para indicar la frecuencia de muestreo.
- WAV PCM mono o estereo.

Ejemplo con frecuencia indicada manualmente:

```powershell
py clasificar_senal.py mi_senal.csv --fs 8000
```

## Que hace el codigo

1. Lee el fichero de senal.
2. Normaliza la amplitud y elimina el desplazamiento medio.
3. Calcula caracteristicas temporales y espectrales basicas.
4. Compara esas caracteristicas con cinco prototipos: `sana`, `estenosis_aortica`, `regurgitacion_mitral`, `estenosis_mitral` y `prolapso_mitral`.
5. Muestra la clase predicha, una confianza aproximada y las distancias a cada prototipo.

## Interpretacion

La salida indica la clase mas cercana al prototipo del modelo. Una distancia menor significa mayor similitud con esa clase. La confianza es orientativa; sirve para interpretar la demo, no para hacer diagnostico medico.

## Limitacion importante

Este repositorio es una demostracion docente y reproducible. No sustituye al sistema experimental completo del TFG, no contiene el entrenamiento completo de Deep ONMF/UjaNet y no debe utilizarse como herramienta clinica.
