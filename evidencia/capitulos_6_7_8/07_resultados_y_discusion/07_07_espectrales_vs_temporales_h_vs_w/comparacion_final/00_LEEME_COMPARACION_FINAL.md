# Comparacion final de la Figura 11

Esta carpeta deja ordenada la comparacion pedida entre los metodos de la Figura 11
del `articulo_objetivo.pdf` y la representacion STFT que se ha pedido como apoyo.

## Que se compara

El articulo objetivo compara en la Figura 11 estas cuatro visualizaciones t-SNE:

1. CNN.
2. DWT.
3. MFCC.
4. Metodo propuesto deep ONMF.

La comparacion de esta carpeta anade:

5. STFT.

STFT se incluye porque se ha pedido expresamente y porque es la representacion
tiempo-frecuencia que sirve de entrada al flujo deep ONMF. En el PDF objetivo la
Figura 11 original no tiene una subfigura STFT independiente.

## Como esta ordenada la carpeta

```text
comparacion final/
  00_LEEME_COMPARACION_FINAL.md
  01_EXPLICACION_PASO_A_PASO.md
  01_codigos_ordenados/
    01_comparar_figura11.py
    02_EJECUTAR_COMPARACION_FIGURA11.bat
    03_extraer_referencia_figura11_pdf.py
    requirements_comparacion.txt
  02_resultados/
    00_INFORME_RESULTADOS.md
    01_figuras_separadas/
    02_pdf_comparativo/
    03_datos_y_metricas/
    04_referencia_pdf_articulo/
  03_primera_prueba_codigo_y_resultados/
    00_LEEME_PRIMERA_PRUEBA.md
    codigo/
    resultados/
  04_prueba_ajustada_codigo_y_resultados/
    00_LEEME_PRUEBA_AJUSTADA.md
    codigo/
    resultados/
```

Los resultados se crean al ejecutar el codigo. La primera lectura recomendada es:

1. Este archivo.
2. `01_EXPLICACION_PASO_A_PASO.md`.
3. `02_resultados/00_INFORME_RESULTADOS.md`.
4. Las figuras sueltas.
5. El PDF comparativo lado a lado.

La carpeta `03_primera_prueba_codigo_y_resultados` congela el primer intento:
codigo y resultados con los 7 SBV medios de Deep ONMF. La carpeta
`04_prueba_ajustada_codigo_y_resultados` contiene el barrido de configuraciones,
el codigo ajustado y el resultado final seleccionado.

## Fuentes locales usadas

La carpeta usa las dos programaciones que ya existen:

- `Programacion objetivo/articulo_objetivo.pdf`.
- `Programacion objetivo/src/tfg_deep_onmf`.
- `Programacion objetivo/resultados/.../caracteristicas_sbv_por_audio.csv`.
- `Programacion a implemenar/articulo_implementar.pdf`.
- `Programacion a implemenar/Implementacion`.

El PDF objetivo indica en su seccion de comparacion de metodos que:

- CNN usa una estructura convolucional de tres bloques principales.
- MFCC usa 40 filtros Mel y 13 coeficientes.
- DWT usa la wavelet `coif5`.
- deep ONMF se visualiza como el metodo propuesto en la Figura 11D.

La programacion de `Programacion a implemenar` aporta el flujo log-mel y el modelo
CNN ya entrenado. La programacion objetivo aporta los WAV, el preprocesado STFT y
las caracteristicas SBV extraidas por deep ONMF.

## Ejecucion

Desde esta carpeta se puede usar el lanzador base:

```powershell
.\01_codigos_ordenados\02_EJECUTAR_COMPARACION_FIGURA11.bat
```

La copia de la primera prueba queda expuesta en:

```powershell
.\03_primera_prueba_codigo_y_resultados\00_LEEME_PRIMERA_PRUEBA.md
```

El script principal tambien se puede ejecutar directamente:

```powershell
& "C:\Users\armga\AppData\Local\Programs\Python\Python313\python.exe" .\01_codigos_ordenados\01_comparar_figura11.py
```

Si falta la dependencia DWT, instala primero:

```powershell
& "C:\Users\armga\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r .\01_codigos_ordenados\requirements_comparacion.txt
```

## Criterio de comparacion

Las imagenes son comparables porque se generan con:

- La misma base Yaseen disponible en `Programacion objetivo/Bases de Datos`.
- Las mismas cinco clases `N`, `AS`, `MR`, `MS` y `MVP`.
- t-SNE con semilla fija.
- Colores constantes por clase.
- Metricas guardadas para no depender solo de la impresion visual.

Para hablar con precision:

- En las fotos, "mejor resolucion" se interpreta como mejor separacion de clases
  en el espacio de caracteristicas.
- `silhouette` mas alto es mejor.
- `Davies-Bouldin` mas bajo es mejor.
- El informe no inventa un resultado si una ejecucion propia no replica
  exactamente la separacion visual del articulo.

## Dos lecturas dentro de resultados

`02_resultados` contiene dos comparaciones distintas y complementarias:

1. `01_figuras_separadas` y `02_pdf_comparativo` son la reproduccion local con
   codigo Python y la base disponible.
2. `04_referencia_pdf_articulo` contiene la Figura 11 extraida del PDF objetivo:
   las subfiguras originales de CNN, DWT, MFCC y deep ONMF se guardan separadas y
   despues en un PDF lado a lado. STFT aparece marcada como panel extra local
   porque no forma parte de la Figura 11 del articulo.

El PDF de la reproduccion local incluye ademas paginas de explicacion: tabla de
metricas, analisis de cada figura y un guion en espanol para estudiar que se
puede afirmar y que se debe contrastar.

## Resultado ajustado recomendado

El resultado ajustado final esta dentro de:

```text
04_prueba_ajustada_codigo_y_resultados/resultados/final_clave
```

En esa comparacion el perfil Deep ONMF seleccionado obtiene mejor separacion
t-SNE que la CNN local segun las dos metricas mostradas en la figura:

- Deep ONMF ajustado: `silhouette t-SNE = 0.2602` y
  `Davies-Bouldin t-SNE = 1.3756`.
- CNN: `silhouette t-SNE = 0.2259` y
  `Davies-Bouldin t-SNE = 2.0143`.

El primer intento se conserva aparte para que se vea la mejora y para no
mezclar resultados base con resultados ajustados.
