# Resultados de las pruebas Deep ONMF 4 s

## Que se ha probado

Se han generado tres resultados Deep ONMF con la misma configuracion de trama:

| resultado | duracion trama | solape | iteraciones | semilla |
| --- | --- | --- | --- | --- |
| `resultado9-deep_onmf_4s_sin_eliminar_audios` | 4 s | 2 s | 120 | 42 |
| `resultado10-deep_onmf_4s_sin_eliminar_audios` | 4 s | 2 s | 120 | 99 |
| `resultado11-deep_onmf_4s_sin_eliminar_audios` | 4 s | 2 s | 120 | 7 |

En las tres ejecuciones se han usado los `1000` WAV y no se ha descartado
ninguno. La auditoria muestra `85000` columnas STFT por clase porque todos los
audios se completan hasta la trama de 4 segundos.

## Metricas Deep ONMF en la comparacion Figura 11

Estas filas corresponden a Deep ONMF al pasar cada CSV de SBV 4 s por el mismo
comparador frente a CNN, DWT, MFCC y STFT.

| variante | silhouette_features | davies_bouldin_features | silhouette_tsne | davies_bouldin_tsne |
| --- | --- | --- | --- | --- |
| 2 s base, semilla 42 | 0.0530 | 3.5940 | 0.1014 | 3.5501 |
| 4 s, semilla 42 | 0.0802 | 3.2828 | 0.0983 | 2.4282 |
| 4 s, semilla 99 | 0.0507 | 3.8372 | 0.1427 | 2.4370 |
| 4 s, semilla 7 | 0.0662 | 3.5293 | 0.1437 | 2.9577 |

## Lectura de las pruebas

1. Si se priorizan los rasgos antes de t-SNE, la mejor variante 4 s probada es
   la de semilla `42`: sube `silhouette_features` y baja
   `davies_bouldin_features` frente a la base 2 s.
2. Si se mira solo la proyeccion t-SNE, las semillas `99` y `7` suben mucho el
   silhouette dibujado. Eso no basta para declararlas mejores, porque la
   separacion previa de rasgos empeora o queda por debajo de la de semilla 42.
3. Ninguna de estas variantes 4 s supera la CNN local en la tabla completa. En
   la comparacion usada para estas pruebas CNN mantuvo aproximadamente
   `silhouette_features=0.3318`, `davies_bouldin_features=1.5327`,
   `silhouette_tsne=0.2259` y `davies_bouldin_tsne=2.0143`.
4. La prueba 4 s es importante para el TFG porque demuestra el efecto de no
   eliminar audios con una trama mas larga, pero el relleno con ceros cambia la
   geometria de las matrices y no garantiza una separacion mejor.

## Decision despues del barrido completo

Estas pruebas 4 s quedan como parte del historial de ajuste. El resultado final
de esta carpeta no se ha elegido con SBV de 4 s: el barrido completo encontro
un perfil Deep ONMF de reconstruccion mejor con las bases de 2 s de
`resultado8`.

Ademas, los testeos existentes de Deep ONMF de 2 s ya muestran que la
inicializacion influye: en
`resultados_deep_onmf/resultado1-testeos_deep_onmf_figura11D_sin_descartar`
la variante `H_media_por_audio` con semilla `99` alcanzo
`silhouette_features=0.0908`. Por tanto la siguiente mejora razonable no es
maquillar t-SNE, sino comparar de forma controlada semillas, forma de agregar H,
normalizacion y criterio de trama.
