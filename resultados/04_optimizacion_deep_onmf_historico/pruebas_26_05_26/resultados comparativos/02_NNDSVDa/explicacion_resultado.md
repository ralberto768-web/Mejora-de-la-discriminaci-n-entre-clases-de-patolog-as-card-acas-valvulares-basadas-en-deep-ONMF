# Resultado NNDSVDa

## Que se ha cambiado

Variante de NNDSVD que sustituye los ceros por la media de la matriz X. Suele evitar bloqueos por ceros en las actualizaciones multiplicativas.

El resto del procedimiento se mantiene igual: misma base de datos, mismas clases, tramas de 2 segundos, solape de 1 segundo, rangos ONMF 9-8-7, 120 iteraciones por capa y penalizacion ortogonal 0.05.

## Como leer sus resultados

- `Figura_5_SBV_por_clase.png`: muestra las bases espectrales aprendidas por clase. Si las curvas son mas limpias y diferenciadas, la base W es mas interpretable.
- `Tabla_2_estadistica_SBV.png`: resume medias, desviaciones y p-valores de los SBV. P-valores mas pequenos indican mayor diferencia estadistica entre clases.
- `Figura_7_distancias_euclideas.png`: compara separacion entre clases y dispersion dentro de clase.
- `Figura_11D_tSNE_deep_ONMF.png`: visualiza los audios en 2D. Es la foto clave para ver si las clases quedan mas separadas.

## Metricas obtenidas

| Metrica | Valor |
|---|---:|
| Error relativo final medio | 0.167966 |
| Error relativo final maximo | 0.202892 |
| Ortogonalidad media de capas | 0.318623 |
| Silhouette en rasgos SBV | 0.055070 |
| Davies-Bouldin en rasgos SBV | 3.506219 |
| Silhouette en t-SNE | 0.088986 |
| Davies-Bouldin en t-SNE | 4.333705 |

## Archivos de este metodo

- Figura 5: `02_NNDSVDa_Figura_5_SBV_por_clase.png`
- Tabla 2 imagen: `02_NNDSVDa_Tabla_2_estadistica_SBV.png`
- Figura 7: `02_NNDSVDa_Figura_7_distancias_euclideas.png`
- Figura 11D: `02_NNDSVDa_Figura_11D_tSNE_deep_ONMF.png`
- Coordenadas t-SNE: `coordenadas_figura_11D_tSNE.csv`

