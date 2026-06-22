# 03 - Deep-ONMF H mejorado con NNDSVD

Implementa Deep-ONMF con matriz `H` y tres inicializaciones: `NNDSVD`, `NNDSVDa` y `NNDSVDar`.
Tambien genera rasgos por audio basados en errores de reconstruccion y softmin F8.

## Ejecucion

```powershell
python ejecutar.py --datos "..\Bases de Datos"
```

## Verificacion rapida

```powershell
python verificar.py --datos "..\Bases de Datos"
```

La salida incluye matrices `W/H`, capas, rasgos F8 y una validacion CSV.
