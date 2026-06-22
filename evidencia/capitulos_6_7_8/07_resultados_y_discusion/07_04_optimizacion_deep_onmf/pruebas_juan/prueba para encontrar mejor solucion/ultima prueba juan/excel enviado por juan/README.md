# Prueba del Excel enviado por Juan

Esta carpeta contiene una prueba independiente para las 186 arquitecturas
decrecientes incluidas en `configuraciones_arquitecturas.xlsx` y sus 186
inversiones crecientes.

La evaluacion activa utiliza exclusivamente:

- clasificacion multiclase;
- UjaNet;
- activacion temporal H final;
- cinco folds sobre los 1000 audios;
- 60 iteraciones Deep-ONMF, penalizacion 0.05 y semilla 42.

Ejecucion completa:

```powershell
py ejecutar_excel_juan.py --workers 4
```

Regeneracion de documentos sin entrenar:

```powershell
py ejecutar_excel_juan.py --solo-informe
```

## Entregables

Los resultados finales se encuentran en `resultados`:

- `TODAS_LAS_COMPARACIONES_EXCEL_JUAN.pdf`: las 186 parejas en el orden
  original del Excel, siempre decreciente y despues creciente.
- `20_MEJORES_DECRECIENTES_Y_SUS_INVERSAS.pdf`: las 20 mejores
  decrecientes y la inversion de cada una.
- `20_MEJORES_CRECIENTES_Y_SUS_INVERSAS.pdf`: las 20 mejores crecientes
  y la decreciente correspondiente.
- `20_MEJORES_COMPARACION_GLOBAL.pdf`: las 20 mejores parejas unicas,
  eligiendo primero la mejor orientacion de cada pareja.

Cada documento de seleccion contiene una tabla de 40 filas y 20 paginas
posteriores con las matrices de confusion enfrentadas.

La carpeta `resultados/tablas_csv` conserva los valores utilizados en las
tablas. La carpeta `resultados/matrices_confusion` contiene las 372 matrices
en conteos y porcentajes. `resultados/auditoria_final.csv` recoge las
comprobaciones automaticas de filas, folds, predicciones, paginas y
coincidencia numerica entre los CSV y los PDF.
