# Informe punto 3: validacion automatica de representaciones

## 1. Resumen ejecutivo del punto 3

Este documento cubre el punto 3 pedido por Juan: validar automaticamente las representaciones mediante clasificadores. Los puntos 1 y 2, donde se demuestra con graficas y metricas que `DeepONMF_H3` discrimina mejor que `DeepONMF_W`, se consideran contexto previo. Aqui se comprueba si esa ventaja temporal se mantiene al usar SVM, KNN y UjaNet.

## 2. Base de datos y protocolo

Se utiliza la base Yaseen con clases `N, AS, MR, MS, MVP`. La evaluacion principal es binaria: `N` se considera normal y `AS/MR/MS/MVP` se consideran anomalas. Tambien se genera una evaluacion multiclase secundaria para estudiar los cinco tipos de sonido.

Se aplica validacion cruzada estratificada con hasta 5 folds. En la ejecucion completa, cada fold contiene 800 senales de entrenamiento y 200 de prueba.

## 3. Auditoria de datos

| clase | audios | duracion_min_s | duracion_media_s | duracion_max_s | audios_rellenados_menores_2s | frecuencia_ok |
| ----- | ------ | -------------- | ---------------- | -------------- | ---------------------------- | ------------- |
| N     | 200    | 2.0337         | 2.3814           | 3.0072         | 0                            | True          |
| AS    | 200    | 2.3499         | 2.7418           | 3.5503         | 0                            | True          |
| MR    | 200    | 1.5755         | 2.2566           | 3.0346         | 16                           | True          |
| MS    | 200    | 1.1556         | 2.3567           | 3.2014         | 14                           | True          |
| MVP   | 200    | 1.7264         | 2.4811           | 3.9929         | 19                           | True          |

## 4. Representaciones comparadas

- `STFT`: espectrograma de magnitud normalizado.
- `MFCC`: 13 coeficientes cepstrales calculados sobre banco Mel.
- `MelSpectrogram`: energia proyectada sobre filtros Mel.
- `LogMelSpectrogram`: version logaritmica de Mel-Spectrogram.
- `DeepONMF_W`: matriz espectral final aprendida por Deep-ONMF.
- `DeepONMF_H3`: matriz temporal final de activaciones de la tercera capa.

Para SVM y KNN las matrices se vectorizan dentro de cada fold y se escalan solo con el entrenamiento. Si la dimensionalidad es alta, se aplica PCA dentro del pipeline del fold, por lo que no hay fuga de informacion hacia el test.

## 5. Clasificadores

- `SVM`: RBF, `C=1.0`, `gamma=scale`, pesos balanceados.
- `KNN`: 5 vecinos con ponderacion por distancia.
- `UjaNet`: arquitectura del articulo con salida sigmoide para la evaluacion binaria y una version equivalente de cinco salidas para el anexo multiclase.

## 6. Metricas binarias del articulo

Se calculan desde la matriz de confusion: `TP`, `TN`, `FP`, `FN`, `Accuracy`, `Sensitivity`, `Specificity`, `Precision` y `Score`. El positivo es la clase anomala.

## 7. Tabla global binaria ordenada por Score

| clasificador | representacion    | Accuracy_mean | Accuracy_std | Sensitivity_mean | Sensitivity_std | Specificity_mean | Specificity_std | Precision_mean | Precision_std | Score_mean | Score_std |
| ------------ | ----------------- | ------------- | ------------ | ---------------- | --------------- | ---------------- | --------------- | -------------- | ------------- | ---------- | --------- |
| SVM          | MFCC              | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| KNN          | LogMelSpectrogram | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| UjaNet       | LogMelSpectrogram | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| UjaNet       | MFCC              | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| UjaNet       | MelSpectrogram    | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| SVM          | LogMelSpectrogram | 0.9990        | 0.0022       | 0.9988           | 0.0028          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 0.9994     | 0.0014    |
| KNN          | MFCC              | 0.9980        | 0.0027       | 0.9975           | 0.0034          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 0.9988     | 0.0017    |
| UjaNet       | STFT              | 0.9990        | 0.0022       | 1.0000           | 0.0000          | 0.9950           | 0.0112          | 0.9988         | 0.0028        | 0.9975     | 0.0056    |
| SVM          | MelSpectrogram    | 0.9880        | 0.0076       | 0.9863           | 0.0081          | 0.9950           | 0.0112          | 0.9987         | 0.0028        | 0.9906     | 0.0080    |
| UjaNet       | DeepONMF_H3       | 0.9920        | 0.0084       | 0.9938           | 0.0062          | 0.9850           | 0.0224          | 0.9963         | 0.0056        | 0.9894     | 0.0132    |
| SVM          | STFT              | 0.9860        | 0.0055       | 0.9850           | 0.0056          | 0.9900           | 0.0137          | 0.9975         | 0.0035        | 0.9875     | 0.0077    |
| KNN          | STFT              | 0.9910        | 0.0065       | 0.9950           | 0.0052          | 0.9750           | 0.0250          | 0.9938         | 0.0062        | 0.9850     | 0.0128    |
| KNN          | MelSpectrogram    | 0.9910        | 0.0065       | 0.9950           | 0.0052          | 0.9750           | 0.0250          | 0.9938         | 0.0062        | 0.9850     | 0.0128    |
| UjaNet       | DeepONMF_W        | 0.9680        | 0.0076       | 0.9738           | 0.0168          | 0.9450           | 0.0481          | 0.9863         | 0.0118        | 0.9594     | 0.0177    |
| SVM          | DeepONMF_H3       | 0.9690        | 0.0139       | 0.9762           | 0.0120          | 0.9400           | 0.0285          | 0.9849         | 0.0072        | 0.9581     | 0.0186    |
| SVM          | DeepONMF_W        | 0.9250        | 0.0150       | 0.9238           | 0.0120          | 0.9300           | 0.0326          | 0.9814         | 0.0087        | 0.9269     | 0.0210    |
| KNN          | DeepONMF_H3       | 0.9370        | 0.0251       | 0.9762           | 0.0103          | 0.7800           | 0.1165          | 0.9473         | 0.0270        | 0.8781     | 0.0588    |
| KNN          | DeepONMF_W        | 0.8800        | 0.0209       | 0.8825           | 0.0195          | 0.8700           | 0.0371          | 0.9645         | 0.0102        | 0.8763     | 0.0256    |

## 8. Tablas binarias por clasificador

### SVM

| clasificador | representacion    | Accuracy_mean | Accuracy_std | Sensitivity_mean | Sensitivity_std | Specificity_mean | Specificity_std | Precision_mean | Precision_std | Score_mean | Score_std |
| ------------ | ----------------- | ------------- | ------------ | ---------------- | --------------- | ---------------- | --------------- | -------------- | ------------- | ---------- | --------- |
| SVM          | MFCC              | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| SVM          | LogMelSpectrogram | 0.9990        | 0.0022       | 0.9988           | 0.0028          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 0.9994     | 0.0014    |
| SVM          | MelSpectrogram    | 0.9880        | 0.0076       | 0.9863           | 0.0081          | 0.9950           | 0.0112          | 0.9987         | 0.0028        | 0.9906     | 0.0080    |
| SVM          | STFT              | 0.9860        | 0.0055       | 0.9850           | 0.0056          | 0.9900           | 0.0137          | 0.9975         | 0.0035        | 0.9875     | 0.0077    |
| SVM          | DeepONMF_H3       | 0.9690        | 0.0139       | 0.9762           | 0.0120          | 0.9400           | 0.0285          | 0.9849         | 0.0072        | 0.9581     | 0.0186    |
| SVM          | DeepONMF_W        | 0.9250        | 0.0150       | 0.9238           | 0.0120          | 0.9300           | 0.0326          | 0.9814         | 0.0087        | 0.9269     | 0.0210    |

### KNN

| clasificador | representacion    | Accuracy_mean | Accuracy_std | Sensitivity_mean | Sensitivity_std | Specificity_mean | Specificity_std | Precision_mean | Precision_std | Score_mean | Score_std |
| ------------ | ----------------- | ------------- | ------------ | ---------------- | --------------- | ---------------- | --------------- | -------------- | ------------- | ---------- | --------- |
| KNN          | LogMelSpectrogram | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| KNN          | MFCC              | 0.9980        | 0.0027       | 0.9975           | 0.0034          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 0.9988     | 0.0017    |
| KNN          | MelSpectrogram    | 0.9910        | 0.0065       | 0.9950           | 0.0052          | 0.9750           | 0.0250          | 0.9938         | 0.0062        | 0.9850     | 0.0128    |
| KNN          | STFT              | 0.9910        | 0.0065       | 0.9950           | 0.0052          | 0.9750           | 0.0250          | 0.9938         | 0.0062        | 0.9850     | 0.0128    |
| KNN          | DeepONMF_H3       | 0.9370        | 0.0251       | 0.9762           | 0.0103          | 0.7800           | 0.1165          | 0.9473         | 0.0270        | 0.8781     | 0.0588    |
| KNN          | DeepONMF_W        | 0.8800        | 0.0209       | 0.8825           | 0.0195          | 0.8700           | 0.0371          | 0.9645         | 0.0102        | 0.8763     | 0.0256    |

### UjaNet

| clasificador | representacion    | Accuracy_mean | Accuracy_std | Sensitivity_mean | Sensitivity_std | Specificity_mean | Specificity_std | Precision_mean | Precision_std | Score_mean | Score_std |
| ------------ | ----------------- | ------------- | ------------ | ---------------- | --------------- | ---------------- | --------------- | -------------- | ------------- | ---------- | --------- |
| UjaNet       | MFCC              | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| UjaNet       | LogMelSpectrogram | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| UjaNet       | MelSpectrogram    | 1.0000        | 0.0000       | 1.0000           | 0.0000          | 1.0000           | 0.0000          | 1.0000         | 0.0000        | 1.0000     | 0.0000    |
| UjaNet       | STFT              | 0.9990        | 0.0022       | 1.0000           | 0.0000          | 0.9950           | 0.0112          | 0.9988         | 0.0028        | 0.9975     | 0.0056    |
| UjaNet       | DeepONMF_H3       | 0.9920        | 0.0084       | 0.9938           | 0.0062          | 0.9850           | 0.0224          | 0.9963         | 0.0056        | 0.9894     | 0.0132    |
| UjaNet       | DeepONMF_W        | 0.9680        | 0.0076       | 0.9738           | 0.0168          | 0.9450           | 0.0481          | 0.9863         | 0.0118        | 0.9594     | 0.0177    |

## 9. Mejor representacion por clasificador

| clasificador | representacion    | Score_mean | Sensitivity_mean | Specificity_mean |
| ------------ | ----------------- | ---------- | ---------------- | ---------------- |
| KNN          | LogMelSpectrogram | 1.0000     | 1.0000           | 1.0000           |
| SVM          | MFCC              | 1.0000     | 1.0000           | 1.0000           |
| UjaNet       | LogMelSpectrogram | 1.0000     | 1.0000           | 1.0000           |

## 10. Resumen multiclase secundario

| clasificador | representacion    | Accuracy_mean | Accuracy_std | Sensitivity_mean | Sensitivity_std | Specificity_mean | Specificity_std | Precision_mean | Precision_std | Score_mean | Score_std |
| ------------ | ----------------- | ------------- | ------------ | ---------------- | --------------- | ---------------- | --------------- | -------------- | ------------- | ---------- | --------- |
| SVM          | MFCC              | 0.9960        | 0.0024       | 0.9900           | 0.0061          | 0.9975           | 0.0015          | 0.9904         | 0.0058        | 0.9938     | 0.0038    |
| UjaNet       | MelSpectrogram    | 0.9904        | 0.0059       | 0.9760           | 0.0147          | 0.9940           | 0.0037          | 0.9768         | 0.0143        | 0.9850     | 0.0092    |
| SVM          | LogMelSpectrogram | 0.9900        | 0.0028       | 0.9750           | 0.0071          | 0.9938           | 0.0018          | 0.9755         | 0.0070        | 0.9844     | 0.0044    |
| KNN          | MFCC              | 0.9900        | 0.0032       | 0.9750           | 0.0079          | 0.9938           | 0.0020          | 0.9762         | 0.0069        | 0.9844     | 0.0049    |
| SVM          | MelSpectrogram    | 0.9892        | 0.0058       | 0.9730           | 0.0144          | 0.9933           | 0.0036          | 0.9738         | 0.0145        | 0.9831     | 0.0090    |
| UjaNet       | LogMelSpectrogram | 0.9884        | 0.0050       | 0.9710           | 0.0124          | 0.9928           | 0.0031          | 0.9719         | 0.0123        | 0.9819     | 0.0078    |
| SVM          | STFT              | 0.9884        | 0.0077       | 0.9710           | 0.0192          | 0.9928           | 0.0048          | 0.9719         | 0.0187        | 0.9819     | 0.0120    |
| UjaNet       | STFT              | 0.9884        | 0.0052       | 0.9710           | 0.0129          | 0.9927           | 0.0032          | 0.9722         | 0.0122        | 0.9819     | 0.0081    |
| UjaNet       | MFCC              | 0.9876        | 0.0061       | 0.9690           | 0.0152          | 0.9922           | 0.0038          | 0.9697         | 0.0146        | 0.9806     | 0.0095    |
| KNN          | LogMelSpectrogram | 0.9872        | 0.0030       | 0.9680           | 0.0076          | 0.9920           | 0.0019          | 0.9702         | 0.0073        | 0.9800     | 0.0047    |
| KNN          | MelSpectrogram    | 0.9828        | 0.0070       | 0.9570           | 0.0175          | 0.9892           | 0.0044          | 0.9581         | 0.0170        | 0.9731     | 0.0110    |
| KNN          | STFT              | 0.9820        | 0.0076       | 0.9550           | 0.0190          | 0.9887           | 0.0048          | 0.9557         | 0.0189        | 0.9719     | 0.0119    |
| UjaNet       | DeepONMF_H3       | 0.9756        | 0.0105       | 0.9390           | 0.0263          | 0.9848           | 0.0066          | 0.9404         | 0.0256        | 0.9619     | 0.0164    |
| SVM          | DeepONMF_H3       | 0.9540        | 0.0099       | 0.8850           | 0.0247          | 0.9713           | 0.0062          | 0.8893         | 0.0240        | 0.9281     | 0.0155    |
| UjaNet       | DeepONMF_W        | 0.9344        | 0.0078       | 0.8360           | 0.0195          | 0.9590           | 0.0049          | 0.8415         | 0.0187        | 0.8975     | 0.0122    |
| KNN          | DeepONMF_H3       | 0.8996        | 0.0131       | 0.7490           | 0.0327          | 0.9373           | 0.0082          | 0.7943         | 0.0271        | 0.8431     | 0.0204    |
| SVM          | DeepONMF_W        | 0.8820        | 0.0109       | 0.7050           | 0.0272          | 0.9262           | 0.0068          | 0.7087         | 0.0319        | 0.8156     | 0.0170    |
| KNN          | DeepONMF_W        | 0.8208        | 0.0114       | 0.5520           | 0.0284          | 0.8880           | 0.0071          | 0.5723         | 0.0322        | 0.7200     | 0.0178    |

## 11. Lectura para la defensa

La comparacion que mas importa es la binaria, porque reproduce la formulacion del articulo: detectar si el sonido es normal o patologico. Si `DeepONMF_H3` supera de forma consistente a `DeepONMF_W`, la conclusion es que la informacion temporal aprendida por Deep-ONMF captura mejor las alteraciones del ciclo cardiaco que la base espectral aislada.

Si algun clasificador no confirma exactamente la hipotesis, no invalida el TFG: indica que la separabilidad depende tambien del clasificador, del tamano de muestra y de como se vectoriza cada representacion. Por eso se guardan todos los folds, matrices de confusion y predicciones, de forma que el resultado sea auditable.