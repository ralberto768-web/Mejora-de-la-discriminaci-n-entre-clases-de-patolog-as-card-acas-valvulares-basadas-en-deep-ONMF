# 04 - Matriz W por audio

Calcula una matriz `W` independiente por audio y genera una figura tipo 11 a partir de esas matrices.
Prueba `NNDSVD`, `NNDSVDa` y `NNDSVDar`.

## Ejecucion

```powershell
python ejecutar.py --datos "..\Bases de Datos"
```

## Verificacion rapida

```powershell
python verificar.py --datos "..\Bases de Datos"
```

La salida incluye NPZ con matrices `W`, CSV de rasgos, coordenadas t-SNE, metricas comparativas y validacion.
