# EXPLICACION DE LO IMPLEMENTADO

## 1. Objetivo de la implementacion

El profesor Juan pidio realizar una prueba concreta sobre el metodo **Deep ONMF**:

> En lugar de inicializar las matrices `W` y `H` de forma aleatoria cambiando la semilla, probar tres metodos habituales de inicializacion: `NNDSVD`, `NNDSVDa` y `NNDSVDar`.

Por tanto, lo que se ha implementado no es una nueva arquitectura completa, sino una modificacion controlada del arranque del algoritmo Deep ONMF. La idea es comprobar si una inicializacion mas informativa mejora los resultados frente a la inicializacion aleatoria que ya tenia el proyecto.

## 2. Que hacia antes el codigo

Antes de esta prueba, Deep ONMF inicializaba las matrices `W` y `H` aleatoriamente:

```python
w = rng.random((filas, rango)) + eps
h = rng.random((rango, columnas)) + eps
```

Esto significa que el algoritmo empezaba desde matrices positivas generadas al azar. Aunque se fijaba una semilla para que el resultado fuera reproducible, el punto de partida seguia siendo aleatorio.

El problema de este enfoque es que Deep ONMF usa actualizaciones multiplicativas y puede quedarse en soluciones distintas dependiendo del punto inicial. Por eso Juan propone probar inicializaciones comunes basadas en SVD.

## 3. Que se ha implementado ahora

Se ha creado una prueba nueva dentro de:

```text
Programacion objetivo/pruebas para el 26-05-26/
```

Dentro hay dos partes principales:

```text
programacion/
resultados comparativos/
```

La carpeta `programacion` contiene el codigo necesario para repetir la prueba. La carpeta `resultados comparativos` contiene las tablas, figuras, informes y PDF generados.

## 4. Metodos comparados

Se comparan cuatro formas de inicializar Deep ONMF:

| Metodo | Que significa |
|---|---|
| Aleatoria actual | Es la inicializacion que ya tenia el proyecto. Sirve como referencia. |
| NNDSVD | Usa SVD para crear un punto inicial no negativo y estructurado. |
| NNDSVDa | Variante de NNDSVD que rellena los ceros con la media de la matriz. |
| NNDSVDar | Variante de NNDSVD que rellena los ceros con ruido aleatorio pequeno. |

La comparacion es justa porque el resto de parametros se mantiene constante:

- Misma base de datos.
- Mismas clases: `N`, `AS`, `MR`, `MS`, `MVP`.
- Tramas de 2 segundos.
- Solape de 1 segundo.
- Rangos Deep ONMF: `9, 8, 7`.
- 120 iteraciones por capa.
- Penalizacion ortogonal: `0.05`.
- Misma forma de generar Tabla 2, Figura 5, Figura 7 y Figura 11D.

## 5. Codigo creado

El codigo principal esta en:

```text
pruebas para el 26-05-26/programacion/01_ejecutar_prueba_inicializaciones.py
```

Ese script hace todo el flujo:

1. Localiza la carpeta `Programacion objetivo`.
2. Carga la base de datos de audios.
3. Construye las matrices de espectrogramas por clase.
4. Ejecuta Deep ONMF con inicializacion aleatoria.
5. Ejecuta Deep ONMF con `NNDSVD`.
6. Ejecuta Deep ONMF con `NNDSVDa`.
7. Ejecuta Deep ONMF con `NNDSVDar`.
8. Guarda las matrices `W` finales.
9. Genera las caracteristicas `SBV`.
10. Genera la Tabla 2.
11. Genera la Figura 5.
12. Genera la Figura 7.
13. Genera la Figura 11D.
14. Calcula metricas de separacion.
15. Crea un PDF comparativo.
16. Crea informes explicativos.

Tambien hay un `.bat` para ejecutarlo en Windows:

```text
pruebas para el 26-05-26/programacion/02_EJECUTAR_PRUEBA_JUAN_INICIALIZACIONES.bat
```

## 6. Resultados generados

Los resultados principales estan en:

```text
pruebas para el 26-05-26/resultados comparativos/
```

Archivos importantes:

| Archivo | Para que sirve |
|---|---|
| `00_INFORME_COMPARATIVO.md` | Informe principal de la prueba de inicializaciones. |
| `00_INFORME_COMPARATIVO_INICIALIZACIONES.pdf` | PDF con explicacion, tabla y figuras. |
| `metricas_inicializaciones_deep_onmf.csv` | Tabla numerica con las metricas de las cuatro inicializaciones. |
| `comparativa_tSNE_inicializaciones.png` | Imagen con las cuatro Figuras 11D una al lado de otra. |
| `tabla_metricas_inicializaciones.png` | Tabla visual de las metricas. |
| `01_fotos_por_separado/` | Todas las figuras separadas por metodo. |

Ademas, cada metodo tiene su propia carpeta:

```text
00_aleatoria_actual/
01_NNDSVD/
02_NNDSVDa/
03_NNDSVDar/
```

Cada una contiene sus figuras, datos y una explicacion individual.

## 7. Que metricas se han usado

Se han usado varias metricas porque cada una mide una cosa diferente:

| Metrica | Interpretacion |
|---|---|
| `error_relativo_final_medio` | Mide cuanto error comete Deep ONMF al reconstruir la matriz original. Menor es mejor. |
| `silhouette_features` | Mide separacion entre clases en los rasgos originales. Mayor es mejor. |
| `davies_bouldin_features` | Mide compactacion/separacion en los rasgos originales. Menor es mejor. |
| `silhouette_tsne` | Mide separacion en la visualizacion t-SNE. Mayor es mejor. |
| `davies_bouldin_tsne` | Mide compactacion/separacion en t-SNE. Menor es mejor. |

Es importante entender que **reconstruir mejor no siempre significa separar mejor las clases**. Por eso la inicializacion aleatoria puede tener menor error de reconstruccion, pero `NNDSVD` puede producir una Figura 11D mas clara.

## 8. Resultado de la prueba de inicializaciones

Los resultados obtenidos fueron:

| Metodo | Silhouette t-SNE | Davies-Bouldin t-SNE | Error medio |
|---|---:|---:|---:|
| Aleatoria actual | 0.102157 | 3.593781 | 0.157113 |
| NNDSVD | 0.106104 | 3.220899 | 0.179624 |
| NNDSVDa | 0.088986 | 4.333705 | 0.167966 |
| NNDSVDar | 0.076758 | 3.231910 | 0.168836 |

La lectura es:

- **Aleatoria actual** reconstruye mejor porque tiene el menor error medio.
- **NNDSVD** separa mejor en la Figura 11D porque tiene el mejor `silhouette_tsne` y el mejor `davies_bouldin_tsne`.
- **NNDSVDa** mejora ligeramente la separacion en rasgos, pero no la visualizacion t-SNE.
- **NNDSVDar** no supera a `NNDSVD` ni a la aleatoria en las metricas principales.

Por tanto, si Juan pregunta por las inicializaciones, la conclusion es:

> La inicializacion mas interesante de las tres propuestas es **NNDSVD**, porque mejora la separacion visual t-SNE frente a la inicializacion aleatoria.

## 9. Comparacion con el resultado final del TFG

Despues se creo tambien el documento:

```text
resultados comparativos/COMPARATIVA.md
resultados comparativos/COMPARATIVA.pdf
```

Ese documento junta:

1. La comparacion final entre `CNN`, `DWT`, `MFCC`, `Deep ONMF optimizado` y `STFT`.
2. La prueba de inicializaciones de Deep ONMF.

La razon de hacer esto es que las inicializaciones solo comparan variantes internas de Deep ONMF, mientras que la comparacion final del TFG compara tecnologias completas.

En la comparacion final, los valores principales fueron:

| Metodo | Silhouette rasgos | Davies-Bouldin rasgos | Silhouette t-SNE | Davies-Bouldin t-SNE |
|---|---:|---:|---:|---:|
| CNN | 0.331782 | 1.532654 | 0.225898 | 2.014267 |
| Deep ONMF optimizado | 0.280944 | 1.378302 | 0.260171 | 1.375569 |

Aqui la lectura es:

- `CNN` gana en `silhouette_features`.
- `Deep ONMF optimizado` gana en `davies_bouldin_features`.
- `Deep ONMF optimizado` gana en `silhouette_tsne`.
- `Deep ONMF optimizado` gana en `davies_bouldin_tsne`.

Por eso, para el TFG, el resultado que se debe mantener como principal es:

> **Deep ONMF optimizado**, porque obtiene la mejor separacion t-SNE y el mejor Davies-Bouldin t-SNE frente al resto de tecnologias.

## 10. Conclusion general

Lo implementado permite responder a Juan con una prueba clara:

1. Se ha mantenido Deep ONMF tal como estaba.
2. Se ha anadido la posibilidad de inicializar `W` y `H` con `NNDSVD`, `NNDSVDa` y `NNDSVDar`.
3. Se han ejecutado las cuatro opciones: aleatoria, NNDSVD, NNDSVDa y NNDSVDar.
4. Se han generado figuras, tablas, metricas y PDF.
5. Se ha comparado tambien con la comparacion final del TFG.

La conclusion defendible es:

> Para la prueba de inicializaciones, **NNDSVD** es la mejor opcion visual porque mejora la separacion t-SNE.  
> Para el resultado principal del TFG, se mantiene **Deep ONMF optimizado**, porque frente a CNN, DWT, MFCC y STFT es el que mejor sostiene la Figura 11.

