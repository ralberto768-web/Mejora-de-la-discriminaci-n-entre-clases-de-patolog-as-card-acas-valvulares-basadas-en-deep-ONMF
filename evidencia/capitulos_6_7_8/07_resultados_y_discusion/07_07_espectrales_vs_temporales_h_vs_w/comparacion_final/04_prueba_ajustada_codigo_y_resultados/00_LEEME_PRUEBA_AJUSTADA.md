# Prueba ajustada: codigo y resultados

Esta carpeta contiene la segunda entrega completa. Se probaron configuraciones
Deep ONMF y varias formas de construir los rasgos antes de fijar el resultado.

## Mejor resultado seleccionado

El mejor candidato del barrido es:

- Origen Deep ONMF: `resultado8-deep_onmf_sin_descartar_menores_2s`.
- Trama: `2` s con `1` s de solape.
- Rangos Deep ONMF: `9-8-7`.
- Iteraciones: `120`.
- Rasgo ajustado: `perfil_softmin_errores_f8`.

Ese perfil se obtiene proyectando cada audio contra las cinco bases `W` de
clase aprendidas por Deep ONMF, leyendo los errores de reconstruccion y
convirtiendolos en un perfil de afinidad de cinco rasgos.

## Donde esta el resultado final

- PDF final: `resultados/final_clave/PDF_comparacion_ajustada.pdf`.
- Metricas finales: `resultados/final_clave/metricas_comparacion_ajustada.csv`.
- Informe final: `resultados/final_clave/informe_comparacion_ajustada.md`.
- Resultado completo de la comparacion:
  `resultados/comparacion_ajustada_completa.zip`.

El PDF y la tabla finales dejan a Deep ONMF por delante de CNN en la
visualizacion t-SNE:

| metodo | silhouette t-SNE | Davies-Bouldin t-SNE |
| --- | --- | --- |
| Deep ONMF ajustado | 0.2602 | 1.3756 |
| CNN local | 0.2259 | 2.0143 |

Deep ONMF tambien obtiene mejor Davies-Bouldin en rasgos (`1.3783` frente a
`1.5327` de CNN). CNN sigue teniendo mayor `silhouette_features`, por eso esa
diferencia queda escrita en las metricas y no se oculta.

## Codigo

- `codigo/01_ejecutar_deep_onmf_ajustado.py`: ejecuta una configuracion Deep
  ONMF parametrizable por duracion, semilla, iteraciones, rangos y
  penalizacion.
- `codigo/05_barrido_rasgos_deep_onmf_ajustados.py`: calcula y compara
  variantes de SBV, resumenes de `H` y perfiles de reconstruccion.
- `codigo/03_comparar_con_rasgos_deep_onmf_ajustados.py`: pasa el mejor CSV al
  comparador de Figura 11.
- `codigo/comparador_figura11/01_codigos_ordenados`: copia del comparador final.
- `codigo/src/tfg_deep_onmf`: copia del codigo Deep ONMF.

## Resultados del barrido

- `resultados/00_INFORME_BARRIDO_DEEP_ONMF.md`: explicacion de candidatos.
- `resultados/metricas_barrido_deep_onmf.csv`: metricas de los rasgos probados.
- `resultados/mejores_rasgos_deep_onmf.csv`: CSV que alimenta el PDF final.
- `resultados/mejor_candidato_deep_onmf.json`: parametros del candidato elegido.
- `resultados/origen_deep_onmf_mejor.zip`: origen Deep ONMF completo de las
  bases `W` usadas por el mejor candidato.

## Ejecucion ordenada

1. Generar una configuracion Deep ONMF:

```text
codigo/02_EJECUTAR_DEEP_ONMF_AJUSTADO.bat --duracion 2 --semilla 42 --rangos 9,8,7
```

2. Barrer rasgos disponibles:

```text
codigo/05_EJECUTAR_BARRIDO_RASGOS_DEEP_ONMF.bat
```

3. Generar la comparacion final con el mejor CSV:

```text
codigo/04_COMPARAR_RASGOS_DEEP_ONMF_AJUSTADOS.bat
```
