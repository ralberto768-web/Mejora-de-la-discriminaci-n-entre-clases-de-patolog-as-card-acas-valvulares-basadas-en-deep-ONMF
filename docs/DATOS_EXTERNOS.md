# Datos externos

El repositorio incluye código, resultados, tablas, figuras, configuraciones y manifiestos. No incluye obligatoriamente todos los audios fuente originales.

## Por qué faltan los audios fuente

Los audios de sonidos cardíacos pueden estar sujetos a restricciones de distribución y además pueden ocupar un volumen elevado. Por ese motivo se tratan como datos externos: el repositorio conserva todo lo necesario para entender y verificar los resultados, pero la reproducción desde cero requiere colocar las bases de datos en una carpeta local.

## Estructura esperada

```text
datos_externos/
  Yaseen/
    audios_originales/
    README_DATOS.txt
  AWGN/
    audios_o_resultados_ruidosos/
    README_DATOS.txt
```

## Qué representa cada carpeta

- `Yaseen`: base de datos de sonidos cardíacos usada como escenario sin ruido o escenario de referencia.
- `AWGN`: material asociado a escenarios con ruido aditivo blanco gaussiano y distintos niveles SNR.

## Qué se puede hacer sin esos datos

Aunque no se tengan los audios fuente, se puede:

- revisar la implementación;
- consultar resultados ya generados;
- comprobar métricas y matrices de confusión;
- revisar las figuras;
- verificar manifiestos;
- leer el documento global del trabajo.

## Qué requiere los datos externos

Repetir todo desde cero requiere:

- cargar audios originales;
- regenerar características;
- volver a ejecutar Deep-ONMF;
- volver a entrenar/evaluar clasificadores;
- comparar las nuevas salidas con los resultados esperados.

## Recomendación de trazabilidad

Si se añaden datos externos, conviene generar un manifiesto propio con:

- ruta del archivo;
- tamaño;
- hash SHA-256;
- origen o versión de la base de datos.

Así una nueva ejecución podrá compararse con la evidencia conservada en el repositorio.

