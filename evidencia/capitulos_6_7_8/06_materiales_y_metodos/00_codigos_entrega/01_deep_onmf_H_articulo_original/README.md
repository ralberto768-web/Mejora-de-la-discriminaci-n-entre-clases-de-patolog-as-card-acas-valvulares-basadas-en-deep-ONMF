# 01 - Deep-ONMF H articulo original

Implementa el flujo fiel del articulo usando la matriz `H` como representacion por audio.
En esta carpeta los audios menores de 2 segundos se descartan.

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
