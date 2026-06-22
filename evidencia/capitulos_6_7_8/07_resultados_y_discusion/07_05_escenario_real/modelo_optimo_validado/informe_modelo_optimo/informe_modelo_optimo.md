# Correccion completa: W, H3, Deep-ONMF mejorado y modelo optimo F8

## 1. Que se corrige

La validacion anterior usaba 60 iteraciones y solo incluia las matrices `DeepONMF_W` y `DeepONMF_H3`. Esta ejecucion se ha calculado de nuevo con el perfil optimo historico: 120 iteraciones por capa, rangos 9-8-7, penalizacion ortogonal 0.05, tramas de 2 s y semilla 42.

Se distinguen cuatro objetos que no deben confundirse:

- `DeepONMF_W`: bases espectrales finales `W1·W2·W3`.
- `DeepONMF_H3`: activaciones temporales de la tercera capa.
- `DeepONMF_Mejorado`: cinco valores `-log(error)` frente a los diccionarios de clase.
- `DeepONMF_F8`: softmin de los cinco errores relativos con fuerza 8.

## 2. Obtencion exacta de W y H3

Cada audio se convierte en una STFT de magnitud no negativa. Deep-ONMF aplica tres factorizaciones ONMF consecutivas. La salida espectral final se obtiene multiplicando `W1·W2·W3`; la salida temporal es `H3`. Como NMF permite permutar componentes sin cambiar la reconstruccion, las columnas de W se ordenan por centroide espectral y la misma permutacion se aplica a las filas de H3. Asi, W y H3 mantienen una correspondencia pieza a pieza.

## 3. Modelo mejorado y F8 sin fuga de informacion

En cada fold se aprenden cinco diccionarios, uno por clase, usando exclusivamente los 800 audios de entrenamiento: 160 N, 160 AS, 160 MR, 160 MS y 160 MVP. Cada audio se proyecta sobre los cinco W con W fija. Los cinco errores de reconstruccion producen la representacion mejorada y F8. Ningun audio de test participa en los diccionarios del fold.

La auditoria registra un maximo de 0 audios de test usados en diccionarios; el valor exigido es cero.

## 4. Base de datos y protocolo

| clase | audios | duracion_min_s | duracion_media_s | duracion_max_s | audios_rellenados_menores_2s | frecuencia_ok |
| ----- | ------ | -------------- | ---------------- | -------------- | ---------------------------- | ------------- |
| N     | 200    | 2.0337         | 2.3814           | 3.0072         | 0                            | True          |
| AS    | 200    | 2.3499         | 2.7418           | 3.5503         | 0                            | True          |
| MR    | 200    | 1.5755         | 2.2566           | 3.0346         | 16                           | True          |
| MS    | 200    | 1.1556         | 2.3567           | 3.2014         | 14                           | True          |
| MVP   | 200    | 1.7264         | 2.4811           | 3.9929         | 19                           | True          |

Se aplica 5-Fold estratificado. Cada audio aparece exactamente una vez en test, por lo que las predicciones agregadas son completamente fuera de muestra.

## 5. Representaciones y clasificadores

Se comparan STFT, MFCC, MelSpectrogram, LogMelSpectrogram, DeepONMF_W, DeepONMF_H3, DeepONMF_Mejorado y DeepONMF_F8. Todas aparecen en SVM, KNN y UjaNet. Para UjaNet, los cinco valores de Mejorado/F8 se colocan en la diagonal de una matriz 5x5. No se duplican ni se inventan rasgos; es una adaptacion formal de entrada y sus resultados se interpretan como complementarios.

DWT no forma parte de las seis representaciones solicitadas en el correo para el punto 3. Se mantiene como baseline historico de los puntos 1-2, pero no se presenta falsamente como una validacion nueva.

## 6. Metricas

`TP` es anomalo detectado como anomalo; `TN`, normal reconocido como normal; `FP`, normal marcado como anomalo; `FN`, patologia confundida como normal. Se calculan Accuracy, Sensitivity, Specificity, Precision y `Score=(Sensitivity+Specificity)/2`.

## 7. Separabilidad visual y objetiva

| representacion    | muestras | rasgos | silhouette_features | davies_bouldin_features | silhouette_tsne | davies_bouldin_tsne |
| ----------------- | -------- | ------ | ------------------- | ----------------------- | --------------- | ------------------- |
| DeepONMF_F8       | 1000     | 5      | 0.1880              | 1.6339                  | 0.1448          | 1.8525              |
| DeepONMF_Mejorado | 1000     | 5      | 0.0770              | 3.1340                  | 0.1381          | 3.1855              |
| DeepONMF_W        | 1000     | 882    | -0.0500             | 6.1816                  | 0.0483          | 3.0897              |
| DeepONMF_H3       | 1000     | 1484   | -0.0508             | 7.5951                  | 0.0499          | 4.3937              |

Silhouette alto y Davies-Bouldin bajo indican mejor separacion, pero estas metricas no son equivalentes a la clasificacion. t-SNE solo proyecta vecindades a dos dimensiones y puede deformar distancias globales. Por eso MFCC puede clasificar muy bien aunque su dibujo t-SNE no sea el mas limpio: un clasificador supervisado aprende fronteras que una proyeccion 2D no tiene por que mostrar.

## 8. Resultados binarios globales

| clasificador | representacion    | Accuracy_mean | Accuracy_std | Sensitivity_mean | Sensitivity_std | Specificity_mean | Specificity_std | Precision_mean | Precision_std | Score_mean | Score_std |
| ------------ | ----------------- | ------------- | ------------ | ---------------- | --------------- | ---------------- | --------------- | -------------- | ------------- | ---------- | --------- |
| KNN          | LogMelSpectrogram | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| UjaNet       | MelSpectrogram    | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| UjaNet       | LogMelSpectrogram | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| SVM          | MFCC              | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| UjaNet       | MFCC              | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| SVM          | LogMelSpectrogram | 0.9990        | 0.0022       | 0.9988           | 0.0028          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 0.9994     | 0.0014    |
| KNN          | MFCC              | 0.9980        | 0.0027       | 0.9975           | 0.0034          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 0.9988     | 0.0017    |
| UjaNet       | STFT              | 0.9990        | 0.0022       | 1.0000           | 0.0000          | 0.9950           | 0.0112          | 0.9988         | 0.0028        | 0.9975     | 0.0056    |
| SVM          | MelSpectrogram    | 0.9880        | 0.0076       | 0.9862           | 0.0081          | 0.9950           | 0.0112          | 0.9987         | 0.0028        | 0.9906     | 0.0080    |
| SVM          | STFT              | 0.9860        | 0.0055       | 0.9850           | 0.0056          | 0.9900           | 0.0137          | 0.9975         | 0.0035        | 0.9875     | 0.0077    |
| KNN          | MelSpectrogram    | 0.9910        | 0.0065       | 0.9950           | 0.0052          | 0.9750           | 0.0250          | 0.9938         | 0.0062        | 0.9850     | 0.0128    |
| KNN          | STFT              | 0.9910        | 0.0065       | 0.9950           | 0.0052          | 0.9750           | 0.0250          | 0.9938         | 0.0062        | 0.9850     | 0.0128    |
| UjaNet       | DeepONMF_H3       | 0.9880        | 0.0097       | 0.9912           | 0.0071          | 0.9750           | 0.0250          | 0.9937         | 0.0063        | 0.9831     | 0.0151    |
| KNN          | DeepONMF_Mejorado | 0.9860        | 0.0065       | 0.9900           | 0.0034          | 0.9700           | 0.0326          | 0.9925         | 0.0081        | 0.9800     | 0.0160    |
| KNN          | DeepONMF_F8       | 0.9890        | 0.0042       | 0.9950           | 0.0052          | 0.9650           | 0.0379          | 0.9914         | 0.0093        | 0.9800     | 0.0166    |
| SVM          | DeepONMF_F8       | 0.9830        | 0.0045       | 0.9925           | 0.0068          | 0.9450           | 0.0209          | 0.9864         | 0.0051        | 0.9688     | 0.0088    |
| UjaNet       | DeepONMF_Mejorado | 0.9850        | 0.0061       | 0.9962           | 0.0034          | 0.9400           | 0.0224          | 0.9852         | 0.0055        | 0.9681     | 0.0120    |
| SVM          | DeepONMF_Mejorado | 0.9760        | 0.0119       | 0.9825           | 0.0120          | 0.9500           | 0.0177          | 0.9874         | 0.0045        | 0.9663     | 0.0131    |
| UjaNet       | DeepONMF_F8       | 0.9810        | 0.0042       | 0.9925           | 0.0068          | 0.9350           | 0.0335          | 0.9840         | 0.0081        | 0.9637     | 0.0143    |
| SVM          | DeepONMF_H3       | 0.9660        | 0.0119       | 0.9762           | 0.0093          | 0.9250           | 0.0250          | 0.9811         | 0.0063        | 0.9506     | 0.0166    |
| UjaNet       | DeepONMF_W        | 0.9650        | 0.0100       | 0.9788           | 0.0071          | 0.9100           | 0.0379          | 0.9776         | 0.0093        | 0.9444     | 0.0197    |
| SVM          | DeepONMF_W        | 0.9360        | 0.0102       | 0.9325           | 0.0081          | 0.9500           | 0.0250          | 0.9868         | 0.0066        | 0.9413     | 0.0152    |
| KNN          | DeepONMF_W        | 0.9270        | 0.0189       | 0.9363           | 0.0190          | 0.8900           | 0.0379          | 0.9715         | 0.0098        | 0.9131     | 0.0236    |
| KNN          | DeepONMF_H3       | 0.9260        | 0.0156       | 0.9812           | 0.0117          | 0.7050           | 0.0512          | 0.9302         | 0.0115        | 0.8431     | 0.0277    |

### SVM

| clasificador | representacion    | Accuracy_mean | Accuracy_std | Sensitivity_mean | Sensitivity_std | Specificity_mean | Specificity_std | Precision_mean | Precision_std | Score_mean | Score_std |
| ------------ | ----------------- | ------------- | ------------ | ---------------- | --------------- | ---------------- | --------------- | -------------- | ------------- | ---------- | --------- |
| SVM          | MFCC              | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| SVM          | LogMelSpectrogram | 0.9990        | 0.0022       | 0.9988           | 0.0028          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 0.9994     | 0.0014    |
| SVM          | MelSpectrogram    | 0.9880        | 0.0076       | 0.9862           | 0.0081          | 0.9950           | 0.0112          | 0.9987         | 0.0028        | 0.9906     | 0.0080    |
| SVM          | STFT              | 0.9860        | 0.0055       | 0.9850           | 0.0056          | 0.9900           | 0.0137          | 0.9975         | 0.0035        | 0.9875     | 0.0077    |
| SVM          | DeepONMF_F8       | 0.9830        | 0.0045       | 0.9925           | 0.0068          | 0.9450           | 0.0209          | 0.9864         | 0.0051        | 0.9688     | 0.0088    |
| SVM          | DeepONMF_Mejorado | 0.9760        | 0.0119       | 0.9825           | 0.0120          | 0.9500           | 0.0177          | 0.9874         | 0.0045        | 0.9663     | 0.0131    |
| SVM          | DeepONMF_H3       | 0.9660        | 0.0119       | 0.9762           | 0.0093          | 0.9250           | 0.0250          | 0.9811         | 0.0063        | 0.9506     | 0.0166    |
| SVM          | DeepONMF_W        | 0.9360        | 0.0102       | 0.9325           | 0.0081          | 0.9500           | 0.0250          | 0.9868         | 0.0066        | 0.9413     | 0.0152    |

### KNN

| clasificador | representacion    | Accuracy_mean | Accuracy_std | Sensitivity_mean | Sensitivity_std | Specificity_mean | Specificity_std | Precision_mean | Precision_std | Score_mean | Score_std |
| ------------ | ----------------- | ------------- | ------------ | ---------------- | --------------- | ---------------- | --------------- | -------------- | ------------- | ---------- | --------- |
| KNN          | LogMelSpectrogram | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| KNN          | MFCC              | 0.9980        | 0.0027       | 0.9975           | 0.0034          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 0.9988     | 0.0017    |
| KNN          | STFT              | 0.9910        | 0.0065       | 0.9950           | 0.0052          | 0.9750           | 0.0250          | 0.9938         | 0.0062        | 0.9850     | 0.0128    |
| KNN          | MelSpectrogram    | 0.9910        | 0.0065       | 0.9950           | 0.0052          | 0.9750           | 0.0250          | 0.9938         | 0.0062        | 0.9850     | 0.0128    |
| KNN          | DeepONMF_Mejorado | 0.9860        | 0.0065       | 0.9900           | 0.0034          | 0.9700           | 0.0326          | 0.9925         | 0.0081        | 0.9800     | 0.0160    |
| KNN          | DeepONMF_F8       | 0.9890        | 0.0042       | 0.9950           | 0.0052          | 0.9650           | 0.0379          | 0.9914         | 0.0093        | 0.9800     | 0.0166    |
| KNN          | DeepONMF_W        | 0.9270        | 0.0189       | 0.9363           | 0.0190          | 0.8900           | 0.0379          | 0.9715         | 0.0098        | 0.9131     | 0.0236    |
| KNN          | DeepONMF_H3       | 0.9260        | 0.0156       | 0.9812           | 0.0117          | 0.7050           | 0.0512          | 0.9302         | 0.0115        | 0.8431     | 0.0277    |

### UjaNet

| clasificador | representacion    | Accuracy_mean | Accuracy_std | Sensitivity_mean | Sensitivity_std | Specificity_mean | Specificity_std | Precision_mean | Precision_std | Score_mean | Score_std |
| ------------ | ----------------- | ------------- | ------------ | ---------------- | --------------- | ---------------- | --------------- | -------------- | ------------- | ---------- | --------- |
| UjaNet       | MFCC              | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| UjaNet       | MelSpectrogram    | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| UjaNet       | LogMelSpectrogram | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| UjaNet       | STFT              | 0.9990        | 0.0022       | 1.0000           | 0.0000          | 0.9950           | 0.0112          | 0.9988         | 0.0028        | 0.9975     | 0.0056    |
| UjaNet       | DeepONMF_H3       | 0.9880        | 0.0097       | 0.9912           | 0.0071          | 0.9750           | 0.0250          | 0.9937         | 0.0063        | 0.9831     | 0.0151    |
| UjaNet       | DeepONMF_Mejorado | 0.9850        | 0.0061       | 0.9962           | 0.0034          | 0.9400           | 0.0224          | 0.9852         | 0.0055        | 0.9681     | 0.0120    |
| UjaNet       | DeepONMF_F8       | 0.9810        | 0.0042       | 0.9925           | 0.0068          | 0.9350           | 0.0335          | 0.9840         | 0.0081        | 0.9637     | 0.0143    |
| UjaNet       | DeepONMF_W        | 0.9650        | 0.0100       | 0.9788           | 0.0071          | 0.9100           | 0.0379          | 0.9776         | 0.0093        | 0.9444     | 0.0197    |

## 9. Por que algunos valores pueden ser 1

En esta ejecucion hay 5 combinaciones con Accuracy, Sensitivity y Specificity medias iguales a 1. Un valor 1 no aparece por usar matrices ni porque el programa lo asigne manualmente. Significa que, en los cinco tests, no hubo ningun FP ni FN para esa combinacion. Debe interpretarse junto con la auditoria de duplicados, el ajuste de escalado/PCA solo con train y las predicciones fuera de muestra.

MFCC puede superar a una representacion con mejor t-SNE porque ambas pruebas responden a preguntas distintas. t-SNE intenta dibujar vecindades en 2D; SVM puede construir una frontera no lineal en el espacio completo de MFCC. Una nube visualmente mezclada no implica que no exista una frontera discriminativa en trece dimensiones.

## 10. Comparacion estadistica W frente a H3

| clasificador | Score_H3_OOF | Score_W_OOF | diferencia_Score_H3_menos_W | Accuracy_H3_OOF | Accuracy_W_OOF | diferencia_Accuracy_H3_menos_W | McNemar_solo_H3_correcto | McNemar_solo_W_correcto | McNemar_discordantes | McNemar_p_exacto | IC95_diferencia_Score_inferior | IC95_diferencia_Score_superior | IC95_diferencia_Accuracy_inferior | IC95_diferencia_Accuracy_superior |
| ------------ | ------------ | ----------- | --------------------------- | --------------- | -------------- | ------------------------------ | ------------------------ | ----------------------- | -------------------- | ---------------- | ------------------------------ | ------------------------------ | --------------------------------- | --------------------------------- |
| KNN          | 0.8431       | 0.9131      | -0.0700                     | 0.9260          | 0.9270         | -0.0010                        | 49                       | 50                      | 99                   | 1.0000           | -0.1045                        | -0.0345                        | -0.0200                           | 0.0170                            |
| SVM          | 0.9506       | 0.9413      | 0.0094                      | 0.9660          | 0.9360         | 0.0300                         | 50                       | 20                      | 70                   | 0.0004           | -0.0126                        | 0.0318                         | 0.0150                            | 0.0470                            |
| UjaNet       | 0.9831       | 0.9444      | 0.0387                      | 0.9880          | 0.9650         | 0.0230                         | 29                       | 6                       | 35                   | 0.0001           | 0.0157                         | 0.0608                         | 0.0110                            | 0.0340                            |

McNemar compara, audio a audio, los desacuerdos de W y H3. El intervalo bootstrap del 95% cuantifica la incertidumbre de la diferencia de Score y Accuracy. Un promedio mayor sin intervalo ni prueba emparejada es una evidencia mas debil.

### Diferencias por fold

| clasificador | fold | Score_H3 | Score_W | diferencia_Score_H3_menos_W | Accuracy_H3 | Accuracy_W | diferencia_Accuracy_H3_menos_W |
| ------------ | ---- | -------- | ------- | --------------------------- | ----------- | ---------- | ------------------------------ |
| KNN          | 1    | 0.8656   | 0.9125  | -0.0469                     | 0.9350      | 0.9350     | 0.0000                         |
| KNN          | 2    | 0.8688   | 0.9062  | -0.0375                     | 0.9400      | 0.9100     | 0.0300                         |
| KNN          | 3    | 0.8063   | 0.8844  | -0.0781                     | 0.9000      | 0.9050     | -0.0050                        |
| KNN          | 4    | 0.8531   | 0.9125  | -0.0594                     | 0.9300      | 0.9350     | -0.0050                        |
| KNN          | 5    | 0.8219   | 0.9500  | -0.1281                     | 0.9250      | 0.9500     | -0.0250                        |
| SVM          | 1    | 0.9500   | 0.9563  | -0.0063                     | 0.9650      | 0.9450     | 0.0200                         |
| SVM          | 2    | 0.9656   | 0.9219  | 0.0437                      | 0.9750      | 0.9200     | 0.0550                         |
| SVM          | 3    | 0.9313   | 0.9313  | 0.0000                      | 0.9500      | 0.9350     | 0.0150                         |
| SVM          | 4    | 0.9375   | 0.9563  | -0.0187                     | 0.9600      | 0.9450     | 0.0150                         |
| SVM          | 5    | 0.9688   | 0.9406  | 0.0281                      | 0.9800      | 0.9350     | 0.0450                         |
| UjaNet       | 1    | 0.9719   | 0.9688  | 0.0031                      | 0.9850      | 0.9800     | 0.0050                         |
| UjaNet       | 2    | 0.9969   | 0.9281  | 0.0687                      | 0.9950      | 0.9600     | 0.0350                         |
| UjaNet       | 3    | 0.9656   | 0.9344  | 0.0312                      | 0.9750      | 0.9550     | 0.0200                         |
| UjaNet       | 4    | 0.9812   | 0.9625  | 0.0187                      | 0.9850      | 0.9700     | 0.0150                         |
| UjaNet       | 5    | 1.0000   | 0.9281  | 0.0719                      | 1.0000      | 0.9600     | 0.0400                         |

## 11. Resultados multiclase

| clasificador | representacion    | Accuracy_mean | Accuracy_std | Sensitivity_mean | Sensitivity_std | Specificity_mean | Specificity_std | Precision_mean | Precision_std | Score_mean | Score_std |
| ------------ | ----------------- | ------------- | ------------ | ---------------- | --------------- | ---------------- | --------------- | -------------- | ------------- | ---------- | --------- |
| SVM          | MFCC              | 0.9960        | 0.0024       | 0.9900           | 0.0061          | 0.9975           | 0.0015          | 0.9904         | 0.0058        | 0.9938     | 0.0038    |
| UjaNet       | MelSpectrogram    | 0.9904        | 0.0059       | 0.9760           | 0.0147          | 0.9940           | 0.0037          | 0.9768         | 0.0143        | 0.9850     | 0.0092    |
| KNN          | MFCC              | 0.9900        | 0.0032       | 0.9750           | 0.0079          | 0.9938           | 0.0020          | 0.9762         | 0.0069        | 0.9844     | 0.0049    |
| SVM          | LogMelSpectrogram | 0.9900        | 0.0028       | 0.9750           | 0.0071          | 0.9938           | 0.0018          | 0.9755         | 0.0070        | 0.9844     | 0.0044    |
| SVM          | MelSpectrogram    | 0.9892        | 0.0058       | 0.9730           | 0.0144          | 0.9932           | 0.0036          | 0.9738         | 0.0145        | 0.9831     | 0.0090    |
| SVM          | STFT              | 0.9884        | 0.0077       | 0.9710           | 0.0192          | 0.9928           | 0.0048          | 0.9719         | 0.0187        | 0.9819     | 0.0120    |
| UjaNet       | STFT              | 0.9884        | 0.0052       | 0.9710           | 0.0129          | 0.9927           | 0.0032          | 0.9722         | 0.0122        | 0.9819     | 0.0081    |
| UjaNet       | LogMelSpectrogram | 0.9884        | 0.0050       | 0.9710           | 0.0124          | 0.9928           | 0.0031          | 0.9719         | 0.0123        | 0.9819     | 0.0078    |
| UjaNet       | MFCC              | 0.9876        | 0.0061       | 0.9690           | 0.0152          | 0.9922           | 0.0038          | 0.9697         | 0.0146        | 0.9806     | 0.0095    |
| KNN          | LogMelSpectrogram | 0.9872        | 0.0030       | 0.9680           | 0.0076          | 0.9920           | 0.0019          | 0.9702         | 0.0073        | 0.9800     | 0.0047    |
| KNN          | MelSpectrogram    | 0.9828        | 0.0070       | 0.9570           | 0.0175          | 0.9892           | 0.0044          | 0.9581         | 0.0170        | 0.9731     | 0.0110    |
| KNN          | STFT              | 0.9820        | 0.0076       | 0.9550           | 0.0190          | 0.9888           | 0.0048          | 0.9557         | 0.0189        | 0.9719     | 0.0119    |
| UjaNet       | DeepONMF_H3       | 0.9688        | 0.0041       | 0.9220           | 0.0104          | 0.9805           | 0.0026          | 0.9245         | 0.0088        | 0.9513     | 0.0065    |
| KNN          | DeepONMF_Mejorado | 0.9652        | 0.0039       | 0.9130           | 0.0097          | 0.9782           | 0.0024          | 0.9153         | 0.0101        | 0.9456     | 0.0061    |
| SVM          | DeepONMF_H3       | 0.9512        | 0.0023       | 0.8780           | 0.0057          | 0.9695           | 0.0014          | 0.8818         | 0.0100        | 0.9237     | 0.0036    |
| KNN          | DeepONMF_F8       | 0.9460        | 0.0024       | 0.8650           | 0.0061          | 0.9663           | 0.0015          | 0.8679         | 0.0063        | 0.9156     | 0.0038    |
| SVM          | DeepONMF_Mejorado | 0.9444        | 0.0101       | 0.8610           | 0.0253          | 0.9653           | 0.0063          | 0.8705         | 0.0234        | 0.9131     | 0.0158    |
| UjaNet       | DeepONMF_W        | 0.9432        | 0.0125       | 0.8580           | 0.0311          | 0.9645           | 0.0078          | 0.8616         | 0.0317        | 0.9113     | 0.0195    |
| UjaNet       | DeepONMF_F8       | 0.9256        | 0.0071       | 0.8140           | 0.0178          | 0.9535           | 0.0045          | 0.8305         | 0.0137        | 0.8838     | 0.0111    |
| SVM          | DeepONMF_F8       | 0.9236        | 0.0095       | 0.8090           | 0.0238          | 0.9523           | 0.0060          | 0.8249         | 0.0199        | 0.8806     | 0.0149    |
| SVM          | DeepONMF_W        | 0.9184        | 0.0055       | 0.7960           | 0.0139          | 0.9490           | 0.0035          | 0.8050         | 0.0164        | 0.8725     | 0.0087    |
| KNN          | DeepONMF_H3       | 0.9104        | 0.0107       | 0.7760           | 0.0268          | 0.9440           | 0.0067          | 0.8056         | 0.0221        | 0.8600     | 0.0167    |
| UjaNet       | DeepONMF_Mejorado | 0.8808        | 0.0076       | 0.7020           | 0.0189          | 0.9255           | 0.0047          | 0.7182         | 0.0481        | 0.8137     | 0.0118    |
| KNN          | DeepONMF_W        | 0.8796        | 0.0135       | 0.6990           | 0.0338          | 0.9247           | 0.0085          | 0.7270         | 0.0273        | 0.8119     | 0.0211    |

## 12. Auditorias

Se encontraron 0 archivos que pertenecen a algun grupo de WAV exactamente duplicados. La tabla completa conserva su hash SHA-256 y permite comprobar si comparten o no etiqueta. Los diccionarios se auditan por fold y los vectores se comparan entre train y test mediante huellas exactas.

| representacion    | fold | vectores_totales | grupos_duplicados | vectores_en_grupos_duplicados | huellas_compartidas_train_test |
| ----------------- | ---- | ---------------- | ----------------- | ----------------------------- | ------------------------------ |
| STFT              | 1    | 1000             | 0                 | 0                             | 0                              |
| STFT              | 2    | 1000             | 0                 | 0                             | 0                              |
| STFT              | 3    | 1000             | 0                 | 0                             | 0                              |
| STFT              | 4    | 1000             | 0                 | 0                             | 0                              |
| STFT              | 5    | 1000             | 0                 | 0                             | 0                              |
| MFCC              | 1    | 1000             | 0                 | 0                             | 0                              |
| MFCC              | 2    | 1000             | 0                 | 0                             | 0                              |
| MFCC              | 3    | 1000             | 0                 | 0                             | 0                              |
| MFCC              | 4    | 1000             | 0                 | 0                             | 0                              |
| MFCC              | 5    | 1000             | 0                 | 0                             | 0                              |
| MelSpectrogram    | 1    | 1000             | 0                 | 0                             | 0                              |
| MelSpectrogram    | 2    | 1000             | 0                 | 0                             | 0                              |
| MelSpectrogram    | 3    | 1000             | 0                 | 0                             | 0                              |
| MelSpectrogram    | 4    | 1000             | 0                 | 0                             | 0                              |
| MelSpectrogram    | 5    | 1000             | 0                 | 0                             | 0                              |
| LogMelSpectrogram | 1    | 1000             | 0                 | 0                             | 0                              |
| LogMelSpectrogram | 2    | 1000             | 0                 | 0                             | 0                              |
| LogMelSpectrogram | 3    | 1000             | 0                 | 0                             | 0                              |
| LogMelSpectrogram | 4    | 1000             | 0                 | 0                             | 0                              |
| LogMelSpectrogram | 5    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_W        | 1    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_W        | 2    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_W        | 3    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_W        | 4    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_W        | 5    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_H3       | 1    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_H3       | 2    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_H3       | 3    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_H3       | 4    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_H3       | 5    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_Mejorado | 1    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_Mejorado | 2    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_Mejorado | 3    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_Mejorado | 4    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_Mejorado | 5    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_F8       | 1    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_F8       | 2    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_F8       | 3    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_F8       | 4    | 1000             | 0                 | 0                             | 0                              |
| DeepONMF_F8       | 5    | 1000             | 0                 | 0                             | 0                              |

## 13. Conclusion calculada desde los resultados

H3 supera a W en parte de los clasificadores, pero no en todos. La hipotesis recibe apoyo parcial y debe explicarse clasificador por clasificador.

La afirmacion final sobre F8 se formula tambien desde la tabla nueva: F8 es el modelo optimo historico por separabilidad, pero su rendimiento de clasificacion se juzga con esta validacion 5-Fold y no con las metricas antiguas de t-SNE.