# Barrido de rasgos Deep ONMF ajustados

## Criterio

Se comparan rasgos que salen de Deep ONMF sin escribir resultados a mano:

- SBV medios de la primera prueba.
- Estadisticos de las activaciones H de la base real de cada audio.
- Errores de reconstruccion de cada audio frente a las cinco bases W de clase.
- Perfiles de afinidad construidos desde esos errores.

El candidato final se elige por `silhouette_features`: mide la separacion antes de t-SNE.
Se conservan tambien Davies-Bouldin y las metricas t-SNE para revisar la foto.

## Mejor candidato

- Resultado Deep ONMF: `resultado8-deep_onmf_sin_descartar_menores_2s`.
- Variante de rasgos: `perfil_softmin_errores_f8`.
- Silhouette en rasgos: `0.2809`.
- Davies-Bouldin en rasgos: `1.3783`.
- Silhouette t-SNE: `0.2602`.
- Davies-Bouldin t-SNE: `1.3756`.

## Tabla completa

| resultado_deep_onmf | variante | duracion_trama_s | rangos_onmf | iteraciones_onmf | semilla_onmf | silhouette_features | davies_bouldin_features | silhouette_tsne | davies_bouldin_tsne |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resultado8-deep_onmf_sin_descartar_menores_2s | perfil_softmin_errores_f8 | 2.0 | 9-8-7 | 120 | 42 | 0.2809 | 1.3783 | 0.2602 | 1.3756 |
| resultado8-deep_onmf_sin_descartar_menores_2s | perfil_softmin_errores_f12 | 2.0 | 9-8-7 | 120 | 42 | 0.2763 | 1.3027 | 0.2492 | 1.3605 |
| resultado8-deep_onmf_sin_descartar_menores_2s | perfil_softmin_errores_f16 | 2.0 | 9-8-7 | 120 | 42 | 0.2652 | 1.3005 | 0.2435 | 1.3802 |
| resultado8-deep_onmf_sin_descartar_menores_2s | perfil_softmin_errores_f4 | 2.0 | 9-8-7 | 120 | 42 | 0.2501 | 1.6284 | 0.1740 | 1.7307 |
| resultado8-deep_onmf_sin_descartar_menores_2s | perfil_softmin_errores_f24 | 2.0 | 9-8-7 | 120 | 42 | 0.2388 | 1.3210 | 0.2327 | 1.4151 |
| resultado8-deep_onmf_sin_descartar_menores_2s | afinidad_y_diferencias_errores | 2.0 | 9-8-7 | 120 | 42 | 0.1902 | 2.3121 | 0.1434 | 3.1046 |
| resultado8-deep_onmf_sin_descartar_menores_2s | softmin_f8_y_resumen_h | 2.0 | 9-8-7 | 120 | 42 | 0.1716 | 2.7970 | 0.1207 | 10.9448 |
| resultado8-deep_onmf_sin_descartar_menores_2s | afinidad_y_resumen_h | 2.0 | 9-8-7 | 120 | 42 | 0.1437 | 3.1785 | 0.1726 | 2.4481 |
| resultado8-deep_onmf_sin_descartar_menores_2s | afinidad_errores_por_base | 2.0 | 9-8-7 | 120 | 42 | 0.1342 | 2.7094 | 0.1597 | 22.7030 |
| resultado8-deep_onmf_sin_descartar_menores_2s | resumen_h_media_std_q90_max | 2.0 | 9-8-7 | 120 | 42 | 0.1084 | 3.7880 | 0.1081 | 14.0467 |
| resultado8-deep_onmf_sin_descartar_menores_2s | sbv_media_log | 2.0 | 9-8-7 | 120 | 42 | 0.1074 | 3.3566 | 0.0868 | 3.8182 |
| resultado8-deep_onmf_sin_descartar_menores_2s | sbv_media_base | 2.0 | 9-8-7 | 120 | 42 | 0.0530 | 3.5940 | 0.1014 | 3.5501 |

## Nota metodologica

Los perfiles por error usan las bases W aprendidas por clase. Esto sigue siendo Deep ONMF,
pero no es el mismo vector reducido de 7 SBV de la primera prueba. En la carpeta ajustada
se deja el codigo y el CSV de rasgos para que esa diferencia quede visible.