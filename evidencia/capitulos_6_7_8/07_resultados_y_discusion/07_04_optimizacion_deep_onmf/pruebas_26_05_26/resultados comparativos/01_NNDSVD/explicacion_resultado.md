# Resultado NNDSVD

## Que se ha cambiado

Inicializacion por SVD no negativa. Deja ceros estructurales donde la descomposicion no aporta valores positivos.

El resto del procedimiento se mantiene igual: misma base de datos, mismas clases, tramas de 2 segundos, solape de 1 segundo, rangos ONMF 9-8-7, 120 iteraciones por capa y penalizacion ortogonal 0.05.

## Como leer sus resultados

- `Figura_5_SBV_por_clase.png`: muestra las bases espectrales aprendidas por clase. Si las curvas son mas limpias y diferenciadas, la base W es mas interpretable.
- `Tabla_2_estadistica_SBV.png`: resume medias, desviaciones y p-valores de los SBV. P-valores mas pequenos indican mayor diferencia estadistica entre clases.
- `Figura_7_distancias_euclideas.png`: compara separacion entre clases y dispersion dentro de clase.
- `Figura_11D_tSNE_deep_ONMF.png`: visualiza los audios en 2D. Es la foto clave para ver si las clases quedan mas separadas.

## Metricas obtenidas

| Metrica | Valor |
|---|---:|
| Error relativo final medio | 0.179624 |
| Error relativo final maximo | 0.213147 |
| Ortogonalidad media de capas | 0.326715 |
| Silhouette en rasgos SBV | -0.010275 |
| Davies-Bouldin en rasgos SBV | 5.125172 |
| Silhouette en t-SNE | 0.106104 |
| Davies-Bouldin en t-SNE | 3.220899 |

## Archivos de este metodo

- Figura 5: `01_NNDSVD_Figura_5_SBV_por_clase.png`
- Tabla 2 imagen: `01_NNDSVD_Tabla_2_estadistica_SBV.png`
- Figura 7: `01_NNDSVD_Figura_7_distancias_euclideas.png`
- Figura 11D: `01_NNDSVD_Figura_11D_tSNE_deep_ONMF.png`
- Coordenadas t-SNE: `coordenadas_figura_11D_tSNE.csv`

