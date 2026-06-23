# Informe de resultados de la comparacion final

## Que se ha generado

- Figuras separadas en `01_figuras_separadas`.
- PDF comparativo lado a lado en `02_pdf_comparativo`.
- Rasgos, coordenadas t-SNE y metricas CSV en `03_datos_y_metricas`.

## Origen real de los datos

- Audios comparados: `1000` WAV referenciados por deep ONMF.
- Caracteristicas deep ONMF: `C:\Users\armga\OneDrive\Escritorio\TFG\Programacion objetivo\resultados\resultado8-deep_onmf_sin_descartar_menores_2s\documentacion_tecnica\caracteristicas_sbv_por_audio.csv`.
- Checkpoint CNN log-mel: `C:\Users\armga\OneDrive\Escritorio\TFG\Programacion a implemenar\Implementacion\resultados\modelo_cnn_2_0s_entrenamiento_70_prueba_30.pt`.
- Semilla t-SNE: `42`.

## Tabla de metricas

| metodo | muestras | rasgos_originales | rasgos_entrada_tsne | silhouette_features | davies_bouldin_features | silhouette_tsne | davies_bouldin_tsne |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CNN | 1000 | 100 | 50 | 0.3318 | 1.5327 | 0.2259 | 2.0143 |
| DWT | 1000 | 18 | 18 | 0.0726 | 3.7698 | 0.1109 | 6.5641 |
| MFCC | 1000 | 13 | 13 | 0.1468 | 4.3445 | 0.1446 | 5.5277 |
| Deep ONMF | 1000 | 7 | 7 | 0.0530 | 3.5940 | 0.1014 | 3.5501 |
| STFT | 1000 | 252 | 50 | 0.0505 | 3.6796 | 0.1319 | 7.8786 |

## Como leer la tabla

La tabla separa dos lecturas. Las columnas con sufijo `features` se calculan antes de reducir a dos dimensiones y describen los rasgos reales que alimentan t-SNE. Las columnas con sufijo `tsne` se calculan sobre las coordenadas dibujadas y ayudan a leer la figura, pero no son una accuracy de clasificacion.

Silhouette debe subir: valores mayores indican que cada audio queda mas cerca de su clase que de las otras. Davies-Bouldin debe bajar: valores menores indican grupos mas compactos y mas separados entre si.

Mejor silhouette en rasgos: CNN (0.3318). Mejor Davies-Bouldin en rasgos: CNN (1.5327). Mejor silhouette t-SNE: CNN (0.2259). Mejor Davies-Bouldin t-SNE: CNN (2.0143).

## Analisis de cada figura

- CNN. La imagen muestra varias islas compactas por clase. En esta ejecucion las agrupaciones de `N`, `MR` y `MS` quedan bastante reconocibles, aunque hay fragmentos y puntos aislados de otras clases. La tabla confirma que la representacion CNN es la referencia local mas fuerte de esta comparacion. silhouette rasgos=0.3318; DB rasgos=1.5327; silhouette t-SNE=0.2259; DB t-SNE=2.0143.
- DWT. La transformada wavelet captura cambios temporales y de frecuencia, pero el mapa presenta islas pequenas mezcladas con outliers. La separacion visual parcial no se convierte en una buena compacidad global: su Davies-Bouldin t-SNE queda alto. silhouette rasgos=0.0726; DB rasgos=3.7698; silhouette t-SNE=0.1109; DB t-SNE=6.5641.
- MFCC. Los MFCC condensan la envolvente espectral. Se ven grupos mejor definidos que en DWT para algunas clases, pero todavia aparecen regiones compartidas y puntos de clases distintas cerca entre si. Por eso mejora algunos indicadores sin reproducir la claridad del panel Deep ONMF del articulo. silhouette rasgos=0.1468; DB rasgos=4.3445; silhouette t-SNE=0.1446; DB t-SNE=5.5277.
- Deep ONMF. Deep ONMF resume cada audio mediante siete SBV. En la foto local hay estructura visible: zonas compactas de `N`, `MR`, `MS` y `AS`, junto con cruces y muestras dispersas, sobre todo cuando `MVP` se aproxima a otras clases. Su Davies-Bouldin t-SNE es competitivo frente a DWT, MFCC y STFT, pero el silhouette local no supera a CNN. silhouette rasgos=0.0530; DB rasgos=3.5940; silhouette t-SNE=0.1014; DB t-SNE=3.5501.
- STFT. STFT usa un resumen espectral de mayor dimension. El t-SNE encuentra trayectorias e islas estrechas, pero varias quedan fragmentadas y proximas a otras clases. Esa geometria explica que el silhouette t-SNE pueda parecer aceptable mientras Davies-Bouldin penaliza mucho la dispersion y la vecindad entre grupos. silhouette rasgos=0.0505; DB rasgos=3.6796; silhouette t-SNE=0.1319; DB t-SNE=7.8786.

## Lectura final

La conclusion visual debe leerse junto a la tabla: esta ejecucion local no coloca a deep ONMF primero en las dos metricas t-SNE a la vez. El articulo objetivo si describe la Figura 11D como la separacion mas clara frente a CNN, DWT y MFCC.

Para la comparacion visual abre primero las fotos separadas y despues el PDF.
La primera pagina del PDF coloca los metodos lado a lado. Las paginas siguientes
explican la tabla, cada figura y un guion de defensa para estudiar el resultado.

## Guion para explicarlo

1. Primero se mira la foto: un buen mapa t-SNE deja nubes compactas de un color y evita regiones donde muchos colores se pisan. Los ejes t-SNE no tienen unidades fisicas; importa la vecindad relativa de los puntos.

2. Despues se contrasta con la tabla. Una foto atractiva no basta: silhouette y Davies-Bouldin ponen numeros a la separacion. En este informe se muestran tanto los rasgos originales como las coordenadas t-SNE para no confundir calidad de caracteristicas con una proyeccion de dos dimensiones.

3. La Figura 11 del articulo es la referencia conceptual: su panel Deep ONMF muestra las clases `N` y `MR` muy diferenciadas y el bloque superior de patologias queda mas ordenado que en CNN, DWT y MFCC. La reproduccion local puede variar por semillas, normalizacion, duracion de trama, forma exacta de construir los SBV y detalles del t-SNE no publicados.

4. Conclusion de esta ejecucion: La conclusion visual debe leerse junto a la tabla: esta ejecucion local no coloca a deep ONMF primero en las dos metricas t-SNE a la vez. El articulo objetivo si describe la Figura 11D como la separacion mas clara frente a CNN, DWT y MFCC. Por eso los ajustes se guardan como pruebas reproducibles y no se fuerza una conclusion que la tabla no sostenga.

## Nota metodologica

La Figura 11 original del articulo objetivo contiene CNN, DWT, MFCC y deep ONMF.
STFT aparece aqui como comparacion adicional pedida. La CNN local procede de la
programacion del articulo de log-mel, mientras deep ONMF procede de la
programacion objetivo.
