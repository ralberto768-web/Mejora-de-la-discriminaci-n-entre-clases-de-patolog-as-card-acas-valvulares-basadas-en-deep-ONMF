# Datos externos

Las bases de datos completas de sonidos cardiacos no se distribuyen dentro del repositorio. Esta decision evita subir audios pesados y respeta posibles restricciones de licencia o redistribucion.

## Uso de la carpeta `datos_externos/`

Coloca aqui las bases de datos necesarias para repetir extracciones o entrenamientos completos. La estructura concreta puede depender de la fuente de datos disponible, pero debe mantenerse separada del codigo y de los resultados ya generados.

Ejemplo de organizacion:

```text
datos_externos/
  yaseen/
    audios/
    etiquetas/
  ruido_awgn/
    configuraciones/
```

## Revision sin datos externos

No hace falta disponer de los audios fuente para revisar el repositorio. Las carpetas `resultados/`, `informe_general/`, `docs/` y `verificacion/` contienen la evidencia ya generada.

## Repeticion de experimentos

Para repetir un experimento completo:

1. Coloca los datos externos en esta carpeta.
2. Comprueba dependencias con `python scripts\comprobar_entorno.py`.
3. Revisa la configuracion del experimento en `metodologia/` o `resultados/`.
4. Ejecuta el script correspondiente.
