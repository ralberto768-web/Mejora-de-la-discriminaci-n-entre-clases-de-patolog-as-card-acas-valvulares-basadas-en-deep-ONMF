# Resultado Aleatoria actual

## Que se ha cambiado

Es la implementacion que ya estaba en el proyecto: W y H se inicializan con numeros aleatorios positivos usando la semilla base.

El resto del procedimiento se mantiene igual: misma base de datos, mismas clases, tramas de 2 segundos, solape de 1 segundo, rangos ONMF 9-8-7, 120 iteraciones por capa y penalizacion ortogonal 0.05.

## Como leer sus resultados

- `Figura_5_SBV_por_clase.png`: muestra las bases espectrales aprendidas por clase. Si las curvas son mas limpias y diferenciadas, la base W es mas interpretable.
- `Tabla_2_estadistica_SBV.png`: resume medias, desviaciones y p-valores de los SBV. P-valores mas pequenos indican mayor diferencia estadistica entre clases.
- `Figura_7_distancias_euclideas.png`: compara separacion entre clases y dispersion dentro de clase.
- `Figura_11D_tSNE_deep_ONMF.png`: visualiza los audios en 2D. Es la foto clave para ver si las clases quedan mas separadas.

## Metricas obtenidas

| Metrica | Valor |
|---|---:|
| Error relativo final medio | 0.157113 |
| Error relativo final maximo | 0.196418 |
| Ortogonalidad media de capas | 0.316526 |
| Silhouette en rasgos SBV | 0.053016 |
| Davies-Bouldin en rasgos SBV | 3.593991 |
| Silhouette en t-SNE | 0.102157 |
| Davies-Bouldin en t-SNE | 3.593781 |

## Archivos de este metodo

- Figura 5: `00_aleatoria_actual_Figura_5_SBV_por_clase.png`
- Tabla 2 imagen: `00_aleatoria_actual_Tabla_2_estadistica_SBV.png`
- Figura 7: `00_aleatoria_actual_Figura_7_distancias_euclideas.png`
- Figura 11D: `00_aleatoria_actual_Figura_11D_tSNE_deep_ONMF.png`
- Coordenadas t-SNE: `coordenadas_figura_11D_tSNE.csv`

