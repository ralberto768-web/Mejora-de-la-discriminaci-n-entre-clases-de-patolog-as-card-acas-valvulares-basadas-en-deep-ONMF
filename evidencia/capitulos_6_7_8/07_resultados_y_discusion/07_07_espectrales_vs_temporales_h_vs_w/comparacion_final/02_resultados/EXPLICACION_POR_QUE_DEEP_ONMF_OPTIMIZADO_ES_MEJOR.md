# EXPLICACION: POR QUE LA ULTIMA IMPLEMENTACION DEEP ONMF OPTIMIZADA ES LA MEJOR

## 1. Que implementacion se ha elegido

La implementacion que se ha elegido como resultado final es:

```text
Deep ONMF optimizado
```

Concretamente, el mejor candidato guardado en la comparacion final fue:

```text
resultado8-deep_onmf_sin_descartar_menores_2s
variante: perfil_softmin_errores_f8
duracion de trama: 2 segundos
solape: 1 segundo
rangos ONMF: 9-8-7
iteraciones por capa: 120
penalizacion ortogonal: 0.05
semilla: 42
```

Esta implementacion no se eligio a mano. Se eligio despues de hacer un barrido de variantes Deep ONMF y comparar sus metricas.

## 2. Que se probo antes de elegirla

Antes de quedarnos con esta version se probaron varias formas de representar la salida de Deep ONMF:

- SBV medios originales.
- SBV con transformacion logaritmica.
- Estadisticos de las activaciones `H`.
- Afinidades calculadas desde los errores de reconstruccion.
- Diferencias entre errores de reconstruccion.
- Perfiles `softmin` de errores con distintas fuerzas: `f4`, `f8`, `f12`, `f16` y `f24`.
- Combinaciones de afinidades con resumenes de `H`.

La variante que quedo mejor fue:

```text
perfil_softmin_errores_f8
```

Sus metricas dentro del barrido fueron:

| Variante Deep ONMF | Silhouette rasgos | Davies-Bouldin rasgos | Silhouette t-SNE | Davies-Bouldin t-SNE |
|---|---:|---:|---:|---:|
| perfil_softmin_errores_f8 | 0.2809 | 1.3783 | 0.2602 | 1.3756 |
| perfil_softmin_errores_f12 | 0.2763 | 1.3027 | 0.2492 | 1.3605 |
| perfil_softmin_errores_f16 | 0.2652 | 1.3005 | 0.2435 | 1.3802 |
| perfil_softmin_errores_f4 | 0.2501 | 1.6284 | 0.1740 | 1.7307 |
| perfil_softmin_errores_f24 | 0.2388 | 1.3210 | 0.2327 | 1.4151 |
| SBV media base | 0.0530 | 3.5940 | 0.1014 | 3.5501 |

La mejora es clara respecto a la representacion SBV base:

- El `silhouette_features` sube de `0.0530` a `0.2809`.
- El `davies_bouldin_features` baja de `3.5940` a `1.3783`.
- El `silhouette_tsne` sube de `0.1014` a `0.2602`.
- El `davies_bouldin_tsne` baja de `3.5501` a `1.3756`.

Esto significa que la nueva representacion no solo mejora la foto t-SNE, sino tambien la separacion numerica de los rasgos.

## 3. Por que `perfil_softmin_errores_f8` funciona mejor

La primera version usaba rasgos derivados directamente de los SBV. Eso resume la informacion de Deep ONMF, pero puede perder parte de la informacion discriminativa entre clases.

La version optimizada hace algo mas informativo:

1. Aprende una base `W` para cada clase.
2. Para cada audio, calcula cuanto error comete cada base de clase al reconstruirlo.
3. Convierte esos errores en un perfil de afinidad usando `softmin`.
4. El audio queda representado por un vector de 5 valores, uno por clase.

La idea es sencilla:

```text
Si un audio pertenece a una clase, la base W de esa clase deberia reconstruirlo mejor.
```

Por eso, en vez de describir solo la forma media de los SBV, esta representacion describe el comportamiento de cada audio frente a todas las bases aprendidas.

Esto ayuda porque convierte Deep ONMF en una representacion mas discriminativa:

- Si la base de `N` reconstruye muy bien un audio y las demas reconstruyen peor, el perfil queda claro.
- Si dos clases se parecen, el perfil refleja esa duda.
- Si una clase esta bien separada, el perfil genera grupos mas compactos en t-SNE.

La fuerza `f8` fue la mas equilibrada:

- `f4` era demasiado suave: no separaba suficiente.
- `f12`, `f16` y `f24` eran mas agresivas: mejoraban alguna metrica, pero perdian equilibrio global.
- `f8` mantuvo la mejor combinacion general: buena separacion en rasgos y buena separacion visual t-SNE.

## 4. Comparacion final frente a CNN, DWT, MFCC y STFT

La comparacion final dio esta tabla:

| Metodo | Silhouette rasgos | Davies-Bouldin rasgos | Silhouette t-SNE | Davies-Bouldin t-SNE |
|---|---:|---:|---:|---:|
| CNN | 0.3318 | 1.5327 | 0.2259 | 2.0143 |
| DWT | 0.0726 | 3.7698 | 0.1109 | 6.5641 |
| MFCC | 0.1468 | 4.3445 | 0.1446 | 5.5277 |
| Deep ONMF optimizado | 0.2809 | 1.3783 | 0.2602 | 1.3756 |
| STFT | 0.0505 | 3.6796 | 0.1319 | 7.8786 |

La lectura es:

- `Deep ONMF optimizado` gana en `Davies-Bouldin rasgos`.
- `Deep ONMF optimizado` gana en `Silhouette t-SNE`.
- `Deep ONMF optimizado` gana en `Davies-Bouldin t-SNE`.
- `CNN` gana solo en `Silhouette rasgos`.

Por eso se puede decir que Deep ONMF optimizado es el mejor resultado para la Figura 11:

```text
Gana 3 de las 4 metricas principales y, sobre todo, gana las dos metricas calculadas sobre la visualizacion t-SNE.
```

## 5. Por que no se elige CNN aunque tenga mejor silhouette en rasgos

CNN obtiene el mejor `silhouette_features`:

```text
CNN: 0.3318
Deep ONMF: 0.2809
```

Eso significa que, en el espacio interno de rasgos, CNN separa bastante bien las clases.

Pero la comparacion no termina ahi. Cuando se mira la representacion t-SNE, que es la que se esta usando para reproducir y defender la Figura 11, Deep ONMF queda mejor:

```text
Silhouette t-SNE
CNN:       0.2259
Deep ONMF: 0.2602
```

Y tambien queda mejor en Davies-Bouldin:

```text
Davies-Bouldin t-SNE
CNN:       2.0143
Deep ONMF: 1.3756
```

En Davies-Bouldin, menor es mejor. Por tanto, Deep ONMF forma grupos mas compactos y mejor separados en la foto final.

La conclusion correcta no es decir que CNN sea mala. La conclusion correcta es:

```text
CNN es fuerte en rasgos originales, pero Deep ONMF optimizado produce una separacion t-SNE mas clara y mas compacta.
```

## 6. Por que DWT, MFCC y STFT quedan por debajo

### DWT

DWT captura informacion tiempo-frecuencia mediante wavelets, pero en esta prueba no separa suficientemente las clases:

- Silhouette t-SNE: `0.1109`
- Davies-Bouldin t-SNE: `6.5641`

El Davies-Bouldin alto indica grupos poco compactos o cercanos entre si.

### MFCC

MFCC mejora algo respecto a DWT y STFT en algunas zonas, pero sigue por debajo de Deep ONMF:

- Silhouette t-SNE: `0.1446`
- Davies-Bouldin t-SNE: `5.5277`

Los MFCC resumen la envolvente espectral, pero pueden perder detalles especificos de los sonidos cardiacos patologicos.

### STFT

STFT usa muchos rasgos espectrales:

```text
252 rasgos originales
```

Pero tener mas rasgos no significa separar mejor. En esta comparacion, STFT queda con:

- Silhouette t-SNE: `0.1319`
- Davies-Bouldin t-SNE: `7.8786`

Esto indica que la informacion espectral directa no esta tan compactada ni tan orientada a separar clases como la representacion Deep ONMF optimizada.

## 7. Por que Deep ONMF optimizado tiene sentido para el TFG

Deep ONMF es adecuado para este TFG por tres razones:

### 1. Aprende bases no negativas interpretables

Los sonidos cardiacos se transforman en espectrogramas, que son matrices no negativas. Deep ONMF trabaja de forma natural con este tipo de datos porque descompone una matriz no negativa en bases tambien no negativas.

Eso hace que las bases `W` puedan interpretarse como patrones espectrales aprendidos.

### 2. La representacion se basa en reconstruccion por clase

La version optimizada no solo extrae rasgos generales. Tambien pregunta:

```text
Que base de clase reconstruye mejor este audio?
```

Eso conecta directamente con la idea de clasificacion por similitud estructural: un audio normal deberia parecerse mas a la base normal que a las patologicas, y viceversa.

### 3. Mejora la separacion visual y numerica de la Figura 11

El objetivo de esta parte del trabajo era reproducir y comparar la Figura 11. En ese contexto, las metricas t-SNE son importantes porque ponen numeros a la separacion visual.

Deep ONMF optimizado es el que mejor sostiene esa comparacion:

- Mejor `silhouette_tsne`.
- Mejor `davies_bouldin_tsne`.
- Mejor `davies_bouldin_features`.

## 8. Como explicarlo oralmente

Una forma sencilla de explicarlo seria:

```text
Probamos varias representaciones de Deep ONMF. La primera, basada en SBV medios, no separaba bien. Despues probamos una representacion basada en los errores de reconstruccion de cada audio frente a las bases W de cada clase. Esa representacion es mas discriminativa porque mide directamente a que clase se parece mas cada audio segun Deep ONMF.

La mejor variante fue perfil_softmin_errores_f8. En la comparacion final, Deep ONMF optimizado supera a CNN, DWT, MFCC y STFT en las dos metricas t-SNE y en Davies-Bouldin de rasgos. CNN solo gana en silhouette de rasgos. Por eso mantenemos Deep ONMF optimizado como resultado principal: no porque todos los numeros le favorezcan, sino porque es el metodo que mejor reproduce la separacion visual buscada en la Figura 11 y la sostiene con metricas objetivas.
```

## 9. Conclusion final

La ultima implementacion se considera la mejor porque:

1. Fue seleccionada mediante un barrido real de variantes Deep ONMF.
2. Mejora claramente la version SBV base.
3. Usa una representacion mas discriminativa basada en errores de reconstruccion por clase.
4. Gana frente al resto de tecnologias en `silhouette_tsne`.
5. Gana frente al resto de tecnologias en `davies_bouldin_tsne`.
6. Gana frente al resto de tecnologias en `davies_bouldin_features`.
7. Mantiene una explicacion tecnica coherente con Deep ONMF: las bases `W` aprendidas por clase se usan para medir que clase reconstruye mejor cada audio.

Por tanto, la frase final defendible es:

```text
La implementacion Deep ONMF optimizada con perfil_softmin_errores_f8 es la que se mantiene como mejor resultado porque ofrece la separacion t-SNE mas clara y compacta de la comparacion final, y porque esa mejora se explica tecnicamente por el uso de errores de reconstruccion frente a bases W aprendidas por clase.
```

