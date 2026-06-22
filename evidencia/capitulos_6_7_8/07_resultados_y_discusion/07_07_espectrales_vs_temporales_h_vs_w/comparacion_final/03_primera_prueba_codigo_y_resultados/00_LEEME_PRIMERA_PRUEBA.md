# Primera prueba: codigo y resultados

Esta carpeta es la copia congelada de la primera comparacion de Figura 11.
Aqui Deep ONMF entra con la representacion base de `7` SBV medios por audio.

## Contenido

- `codigo/01_codigos_ordenados`: copia del comparador usado para la primera
  prueba.
- `codigo/src/tfg_deep_onmf`: copia del codigo Deep ONMF necesario para
  entender de donde salen los SBV y las bases `W`.
- `resultados/02_resultados`: fotos, PDF, CSV e informe de la primera prueba.
- `resultados/origen_deep_onmf_base.zip`: resultado Deep ONMF base completo
  usado como origen de la prueba.

## Resultado que representa

La tabla de esta carpeta conserva la lectura de la primera prueba:

- Deep ONMF base: `silhouette t-SNE` aproximadamente `0.1014`.
- CNN local: `silhouette t-SNE` aproximadamente `0.2259`.

Por eso esta carpeta se mantiene separada: sirve para demostrar que el primer
enfoque con solo la media de los 7 SBV no era suficiente.

## Como repetir el flujo base

El lanzador de esta carpeta sigue llamando al comparador base:

```text
01_EJECUTAR_COMPARACION_ACTUAL.bat
```

Para estudiar el resultado sin ejecutar nada, empieza por:

```text
resultados/02_resultados/00_INFORME_RESULTADOS.md
resultados/02_resultados/02_pdf_comparativo
```
