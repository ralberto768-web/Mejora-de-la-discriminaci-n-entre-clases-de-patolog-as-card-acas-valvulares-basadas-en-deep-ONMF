# 02 - Deep-ONMF H rellenando audios

Implementa el flujo Deep-ONMF usando la matriz `H`, pero rellena con ceros los audios menores de 2 segundos.

## Ejecucion

```powershell
python ejecutar.py --datos "..\Bases de Datos"
```

Los resultados se guardan en `resultados`, salvo que se indique `--salida`.

## Verificacion rapida

```powershell
python verificar.py --datos "..\Bases de Datos"
```

La verificacion usa pocos audios por clase y comprueba que se generan matrices `H`, matrices `W`, resumen CSV y validacion.
