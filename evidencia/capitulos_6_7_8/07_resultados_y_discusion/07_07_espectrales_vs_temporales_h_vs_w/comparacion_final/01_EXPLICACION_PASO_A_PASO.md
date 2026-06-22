# Explicacion paso a paso

## 1. Objetivo

El objetivo es estudiar la Figura 11 del `articulo_objetivo.pdf` con codigo
ordenado y resultados trazables. La Figura 11 no es una clasificacion final:
es una comparacion visual de espacios de caracteristicas mediante t-SNE.

Si un metodo forma grupos de clase mas compactos y separados en la visualizacion,
se interpreta que sus caracteristicas conservan mejor la informacion discriminante
para diferenciar sonidos cardiacos.

## 2. Que dicen los PDF y que se toma de cada programacion

### Articulo objetivo

La seccion 6.7 del articulo objetivo compara deep ONMF con:

- CNN.
- DWT.
- MFCC.

El texto del articulo describe estos detalles para la comparacion:

- CNN: capas convolucionales con 32 y 64 filtros y un max pooling.
- MFCC: 40 filtros Mel y 13 coeficientes finales.
- DWT: wavelet `coif5`.
- deep ONMF: usa las caracteristicas jerarquicas derivadas de la matriz final `W`
  y los SBV del metodo propuesto.

### Articulo a implementar

El PDF de `Programacion a implemenar` trabaja con espectrogramas log-mel y redes
CNN/LSTM. De esa programacion se reutiliza la parte que interesa aqui:

- La representacion log-mel.
- La CNN ya entrenada sobre segmentos de 2 segundos.
- El checkpoint guardado en `Implementacion/resultados`.

La LSTM no se mete en esta comparacion porque la Figura 11 del articulo objetivo
no la incluye.

## 3. Por que aparece STFT

El articulo objetivo usa STFT para pasar de la senal PCG a una matriz
tiempo-frecuencia antes de aplicar deep ONMF. La peticion pide tener tambien STFT
en la comparacion final, por eso se genera una quinta figura.

En el informe queda separada como figura adicional. Asi no se altera la lectura
de la Figura 11 original.

## 4. Entradas reales usadas

El codigo lee:

1. Los WAV de `Programacion objetivo/Bases de Datos`.
2. Las caracteristicas deep ONMF ya calculadas. Por defecto se prioriza el
   resultado mas reciente con trama de 2 segundos que contenga
   `caracteristicas_sbv_por_audio.csv`.
3. El checkpoint CNN de 2 segundos mas reciente que haya en
   `Programacion a implemenar/Implementacion/resultados`.

No se dibujan puntos ficticios. Cada punto t-SNE procede de caracteristicas
calculadas desde audios reales o de SBV reales ya generados por deep ONMF.

## 5. Extraccion de caracteristicas por metodo

### STFT

Para cada audio:

1. Se parte en tramas de 2 segundos con el mismo criterio de la programacion
   deep ONMF.
2. Se calcula la magnitud STFT.
3. Se resumen los bins de frecuencia mediante media y desviacion tipica.
4. Si el audio produce varias tramas, se promedian sus vectores.

### MFCC

Para cada trama:

1. Se calcula log-mel.
2. Se usan 40 bandas Mel.
3. Se aplica DCT.
4. Se conservan 13 coeficientes.
5. Se promedian coeficientes en el tiempo y despues entre tramas.

### DWT

Para cada trama:

1. Se aplica DWT con `coif5`.
2. Se toman coeficientes de aproximacion y detalle en varios niveles.
3. Se guardan energia logaritmica, media absoluta y desviacion tipica por grupo
   de coeficientes.
4. Se promedian las tramas de cada audio.

### CNN

Para cada trama:

1. Se calcula el log-mel de la programacion a implementar.
2. Se pasa por la CNN ya entrenada.
3. Se extrae el vector de la capa anterior a la clasificacion final.
4. Se promedian las tramas de cada audio.

Esto permite visualizar el espacio de caracteristicas aprendido por la CNN, no
solo su etiqueta final.

### deep ONMF

Se leen las columnas `SBV_1` a `SBV_7` ya generadas por la programacion objetivo
para cada audio. Son la representacion compacta que se quiere defender y estudiar.

## 6. Paso comun de visualizacion

Para cada metodo:

1. Se estandariza la matriz de caracteristicas.
2. Si tiene muchas columnas se usa PCA previa solo para estabilizar la entrada
   de t-SNE.
3. Se aplica t-SNE con semilla fija.
4. Se guardan coordenadas, figura PNG y metricas.

## 7. Como leer las salidas

### Figuras separadas

Estan en:

```text
02_resultados/01_figuras_separadas
```

Sirven para estudiar un metodo cada vez.

### PDF comparativo

Esta en:

```text
02_resultados/02_pdf_comparativo
```

La primera pagina coloca los metodos lado a lado. Las paginas siguientes guardan
la tabla de metricas, el analisis visual de cada metodo y un guion para explicar
la lectura sin depender solo de la vista.

### Referencia extraida del PDF objetivo

Esta en:

```text
02_resultados/04_referencia_pdf_articulo
```

Primero se guardan las cuatro subfiguras originales de la Figura 11 por separado.
Despues se genera un PDF de referencia lado a lado. Esa carpeta sirve para
estudiar la evidencia visual que aparece en el articulo: `(d) deep ONMF` es el
panel en el que los grupos quedan mas separados. El panel STFT aparece como
anexo local porque no existe como subfigura de la Figura 11 original.

### Datos y metricas

Estan en:

```text
02_resultados/03_datos_y_metricas
```

Incluyen:

- Coordenadas t-SNE.
- Vectores de caracteristicas por metodo.
- Tabla CSV de metricas.
- JSON con rutas y parametros usados.

## 8. Que conclusion se puede defender

El articulo objetivo afirma que la Figura 11D de deep ONMF muestra la separacion
mas clara de clases frente a CNN, DWT y MFCC. Esta carpeta permite comprobarlo de
forma ordenada con tu base local y anade STFT como referencia.

Si la reproduccion propia no queda identica al PDF, la explicacion correcta es:

- El articulo no publica todo el codigo MATLAB ni todos los detalles internos.
- La CNN local procede del segundo articulo y usa su flujo log-mel.
- t-SNE es sensible a representacion, normalizacion y semilla.

Por eso se conservan codigo, parametros, datos intermedios y metricas.

## 9. Dos carpetas completas

La entrega queda separada para no mezclar el primer intento con el ajuste:

1. `03_primera_prueba_codigo_y_resultados`: copia el codigo y los resultados
   de la comparacion base, donde Deep ONMF usa solo 7 SBV medios por audio.
2. `04_prueba_ajustada_codigo_y_resultados`: copia el codigo ajustado, el
   barrido de pruebas y el resultado final seleccionado.

En el ajuste no se ha cambiado una figura a mano. Se han probado resultados
Deep ONMF con semillas, iteraciones, rangos y duraciones distintas; despues se
han comparado varias formas de representar cada audio desde Deep ONMF:

- SBV medios base.
- Estadisticos de las activaciones `H`.
- Errores de reconstruccion frente a las cinco bases `W`.
- Perfiles de afinidad derivados de esos errores.

El mejor resultado seleccionado es el perfil `perfil_softmin_errores_f8`. Ese
perfil deja a Deep ONMF por delante de CNN en `silhouette t-SNE` y
`Davies-Bouldin t-SNE`, y el barrido completo queda guardado en la segunda
carpeta.
