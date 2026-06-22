# Mejora de la discriminación entre clases de patologías cardíacas valvulares basadas en deep-ONMF

Repositorio preparado para acompanar el Trabajo Fin de Grado y permitir que el tribunal pueda revisar, verificar y, con los datos externos, reproducir la implementacion.

## Nombre del repositorio

Nombre recomendado en GitHub:

`mejora-discriminacion-patologias-cardiacas-valvulares-deep-onmf`

El titulo completo del TFG se conserva en `TITULO_TFG.md`.

## Que contiene

- `evidencia/capitulos_6_7_8`: paquete organizado de materiales y metodos, resultados, discusion, conclusiones y manifiestos.
- `documento_global`: documento PDF/Markdown de evaluacion punto a punto para los capitulos 6, 7 y 8.
- `docs`: guias para el tribunal, datos externos, mapa del indice, reproducibilidad y resultados esperados.
- `scripts`: comprobaciones automaticas de entorno, manifiesto y resultados incluidos.
- `datos_externos`: carpeta preparada para colocar bases de datos que no se suben directamente al repositorio.
- `github`: instrucciones de subida, Git LFS, manifiestos y estado del paquete.

## Comprobacion rapida

En Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\run_all.bat todo
```

Si no se quieren instalar dependencias, al menos se puede ejecutar:

```powershell
python scripts\verificar_repositorio.py --modo rapido
```

## Reproduccion completa

La reproduccion desde cero requiere colocar los audios/datos externos en `datos_externos`, respetando las instrucciones de `docs/DATOS_EXTERNOS.md`.

El repositorio incluye resultados, figuras, tablas, configuraciones, codigo y manifiestos. Los audios fuente completos no se fuerzan dentro del repositorio por tamano/licencia; se documentan como datos externos.

## Estado generado

- Fecha de preparacion: 2026-06-22 17:27:16
- Archivos copiados: 0
- Archivos ya presentes: 66830
- Errores de copia: 0
- Tamano aproximado copiado: 8.13 GB
- Archivos de 50 MB o mas: 16

