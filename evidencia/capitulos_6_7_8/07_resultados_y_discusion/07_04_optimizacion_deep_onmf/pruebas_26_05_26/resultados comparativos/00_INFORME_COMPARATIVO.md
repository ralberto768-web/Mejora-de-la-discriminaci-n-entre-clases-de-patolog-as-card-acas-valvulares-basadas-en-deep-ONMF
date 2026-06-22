# Prueba para Juan - Inicializaciones Deep ONMF

## Objetivo

Comparar el Deep ONMF actual con tres inicializaciones comunes de matrices W y H:

- Implementacion actual con inicializacion aleatoria.
- NNDSVD.
- NNDSVDa.
- NNDSVDar.

La prueba mantiene constantes el resto de parametros para que la comparacion sea justa.

## Parametros usados

- Trama PCG: 2 segundos.
- Solape: 1 segundo.
- Rangos Deep ONMF: 9, 8 y 7.
- Iteraciones por capa: 120.
- Penalizacion ortogonal: 0.05.
- Semilla base: 42.
- Audios cortos: se rellenan con ceros para no eliminar muestras.

## Auditoria de datos

| clase | audios_totales | audios_usados | audios_descartados_por_duracion | audios_cortos_menores_2s | duracion_min_s | duracion_media_s | duracion_max_s | columnas_matriz_x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | 200 | 200 | 0 | 0 | 2.033750 | 2.381381 | 3.007250 | 42612 |
| AS | 200 | 200 | 0 | 0 | 2.349875 | 2.741807 | 3.550250 | 49184 |
| MR | 200 | 200 | 0 | 16 | 1.575500 | 2.256554 | 3.034625 | 43248 |
| MS | 200 | 200 | 0 | 14 | 1.155625 | 2.356657 | 3.201375 | 45156 |
| MVP | 200 | 200 | 0 | 19 | 1.726375 | 2.481116 | 3.992875 | 43248 |

## Tabla comparativa

| metodo | error_relativo_final_medio | ortogonalidad_media_capas | silhouette_features | davies_bouldin_features | silhouette_tsne | davies_bouldin_tsne |
| --- | --- | --- | --- | --- | --- | --- |
| Aleatoria actual | 0.157113 | 0.316526 | 0.053016 | 3.593991 | 0.102157 | 3.593781 |
| NNDSVD | 0.179624 | 0.326715 | -0.010275 | 5.125172 | 0.106104 | 3.220899 |
| NNDSVDa | 0.167966 | 0.318623 | 0.055070 | 3.506219 | 0.088986 | 4.333705 |
| NNDSVDar | 0.168836 | 0.328599 | -0.003665 | 3.620077 | 0.076758 | 3.231910 |

## Lectura de la tabla

- `error_relativo_final_medio`: cuanto menor, mejor reconstruye Deep ONMF las matrices de espectrograma.
- `silhouette_features`: cuanto mayor, mejor separacion de clases en los rasgos SBV antes de t-SNE.
- `davies_bouldin_features`: cuanto menor, mejor separacion/compactacion en rasgos SBV.
- `silhouette_tsne`: cuanto mayor, mejor separacion visible en la Figura 11D.
- `davies_bouldin_tsne`: cuanto menor, mejor separacion visible en la Figura 11D.

## Resultado principal

- Mejor `silhouette_tsne`: **NNDSVD** con 0.106104.
- Mejor `davies_bouldin_tsne`: **NNDSVD** con 3.220899.
- Menor error relativo medio: **Aleatoria actual** con 0.157113.

## Donde mirar las fotos

- `01_fotos_por_separado/`: todas las figuras de cada inicializacion por separado.
- `comparativa_tSNE_inicializaciones.png`: las cuatro Figuras 11D una al lado de otra.
- `tabla_metricas_inicializaciones.png`: tabla visual con las metricas clave.
- `00_INFORME_COMPARATIVO_INICIALIZACIONES.pdf`: PDF final con explicacion y comparativa.

## Nota para defenderlo

Esta prueba no cambia la arquitectura Deep ONMF ni la base de datos. Solo cambia la forma de arrancar W y H.
Si una inicializacion mejora la foto o las metricas, la explicacion es que el algoritmo empieza desde una base
mas informativa que el azar y evita algunos minimos locales de las actualizaciones multiplicativas.
