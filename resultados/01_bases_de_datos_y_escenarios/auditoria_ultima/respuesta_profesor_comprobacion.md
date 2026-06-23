# Comprobacion de resultados H1 vs Deep-ONMF

## Conclusion

No he encontrado un error en la alimentacion de UjaNet. La red esta recibiendo la matriz de activaciones final `H` (`h_final`) generada por ONMF/Deep-ONMF, no la matriz `W` ni el STFT original.

El resultado de que una capa sea competitiva con varias capas parece real dentro de este protocolo experimental. Aun asi, debe interpretarse con cuidado: H1 esta optimizada buscando bases de 8 a 32, y no es una configuracion debil fija. Ademas, UjaNet parece saturar bastante el dataset.

## Evidencias tecnicas

1. En la extraccion Deep-ONMF se guarda `resultado.h_final`:
   - `ULTIMA/codigo/ultima_final/extraccion.py`
   - Linea clave: `matrices_h.append(resultado.h_final.astype(np.float32))`

2. En la evaluacion, UjaNet recibe esa matriz `x` y la adapta solo con padding minimo si hace falta:
   - `ULTIMA/codigo/ultima_final/evaluacion.py`
   - Linea clave: `x_ujanet = adaptar_para_ujanet(np.asarray(x, dtype=np.float32))`
   - Despues se convierte a tensor como una imagen de un canal: `tensor_x = torch.tensor(x_norm[:, None, :, :], dtype=torch.float32)`

3. Las formas guardadas de las matrices `H` optimas son coherentes:

| Dataset | Modelo | Bases | Forma H guardada | Forma que registra UjaNet |
|---|---:|---:|---:|---:|
| Original | H1 | 10 | 1000 x 10 x 212 | 1000 x 10 x 212 |
| Original | H2 | 28-10 | 1000 x 10 x 212 | 1000 x 10 x 212 |
| Original | H3 | 16-10-6 | 1000 x 6 x 212 | 1000 x 6 x 212 |
| Original | H4 | 24-15-10-6 | 1000 x 6 x 212 | 1000 x 6 x 212 |
| SNR0db | H1 | 10 | 1000 x 10 x 212 | 1000 x 10 x 212 |
| SNR0db | H2 | 28-2 | 1000 x 2 x 212 | 1000 x 2 x 212 |
| SNR0db | H3 | 30-13-6 | 1000 x 6 x 212 | 1000 x 6 x 212 |
| SNR0db | H4 | 12-10-8-6 | 1000 x 6 x 212 | 1000 x 6 x 212 |

4. Las matrices no son copias entre si:

| Comparacion | Diferencia relativa Frobenius | Correlacion |
|---|---:|---:|
| Original H1 vs H2 | 1.2879 | 0.2288 |
| Original H3 vs H4 | 1.3148 | 0.2208 |
| SNR0db H3 vs H4 | 1.2653 | 0.1027 |

5. Las predicciones tampoco son identicas aunque las metricas sean parecidas:

| Comparacion | Predicciones iguales | Predicciones distintas |
|---|---:|---:|
| Original H1 vs H2 | 965 / 1000 | 35 |
| Original H1 vs H3 | 969 / 1000 | 31 |
| Original H3 vs H4 | 964 / 1000 | 36 |

6. Validacion de folds:
   - 1000 audios.
   - 5 clases con 200 audios por clase.
   - 5 folds estratificados.
   - Cada fold de test tiene 40 audios por clase.
   - Solape train/test por fold: 0.

## Lectura de los resultados

En dataset original:

| Modelo | Bases | Accuracy mean | Exactitud directa |
|---|---:|---:|---:|
| H1 | 10 | 0.9796 | 0.949 |
| H2 | 28-10 | 0.9796 | 0.949 |
| H3 | 16-10-6 | 0.9816 | 0.954 |
| H4 | 24-15-10-6 | 0.9768 | 0.942 |

En dataset SNR0db:

| Modelo | Bases | Accuracy mean | Exactitud directa |
|---|---:|---:|---:|
| H1 | 10 | 0.9672 | 0.918 |
| H2 | 28-2 | 0.9716 | 0.929 |
| H3 | 30-13-6 | 0.9672 | 0.918 |
| H4 | 12-10-8-6 | 0.9660 | 0.915 |

## Respuesta corta para Juan

He revisado el flujo y UjaNet esta recibiendo la matriz de activaciones final `H`. Las matrices `H1`, `H2`, `H3` y `H4` no son identicas, y las predicciones tampoco coinciden exactamente. Los folds no tienen solape entre entrenamiento y test.

Por tanto, no parece un fallo de alimentacion de la red. Lo que ocurre es que ONMF de una sola capa, cuando se optimiza el numero de bases entre 8 y 32, ya produce una representacion muy competitiva para este dataset. La mejora multicapa existe en el dataset original para H3, pero es pequena. En SNR0db, H2 es el mejor.

Si se quiere reforzar aun mas la fiabilidad, la siguiente prueba seria repetir el entrenamiento de UjaNet con varias semillas y reportar media y desviacion por configuracion optima.
