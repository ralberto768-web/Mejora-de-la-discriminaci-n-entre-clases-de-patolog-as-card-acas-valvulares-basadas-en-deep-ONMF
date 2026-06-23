# ULTIMA - pruebas finales solicitadas por Juan

Esta carpeta es independiente del resto del proyecto. Todo lo que genera queda
dentro de `Ultima implementacio Juan/ULTIMA`.

## Prueba rapida

Desde esta carpeta:

```powershell
py ejecutar.py --rapido --limite-por-clase 2
```

La prueba rapida valida el flujo con pocos audios. Sus salidas quedan en
`auditoria/verificacion_rapida` y no son resultados cientificos.

## Ejecucion completa

```powershell
py ejecutar.py --datos-originales "..\..\Bases de Datos"
```

La ejecucion completa genera:

- `documentos_finales/optimizacion_dataset_original.pdf`
- `documentos_finales/optimizacion_dataset_SNR0db.pdf`
- `documentos_finales/Resultados_Optimizacion_original.pdf`
- `documentos_finales/Resultados_Optimizacion_SNR0db.pdf`

El proceso es reanudable: si una arquitectura ya tiene predicciones y resumen,
se reutiliza.

## Criterio de seleccion

El criterio de ranking es UjaNet multiclase sobre la activacion temporal final:
`DeepONMF_H1`, `DeepONMF_H2`, `DeepONMF_H3` o `DeepONMF_H4`.

Para 1 capa se evalua ONMF estandar con bases de 8 a 32 en saltos de 2. Para
2, 3 y 4 capas se usa la distribucion decreciente del Excel de Juan.
