# Datos externos

Este repositorio esta preparado para revision y verificacion. Para repetir todo desde cero se necesitan los datos externos originales.

## Estructura esperada

Colocar los datos asi:

```text
datos_externos/
  Yaseen/
    README_DATOS.txt
    audios_originales/
  AWGN/
    README_DATOS.txt
    audios_o_resultados_ruidosos/
```

## Por que no se suben directamente todos los audios

Los audios fuente y bases completas pueden tener restricciones de licencia, tamano elevado y no son adecuados para una subida normal a GitHub. El repositorio conserva resultados, configuraciones, tablas, figuras, codigo y manifiestos; los datos pesados se tratan como dependencia externa.

## Comprobacion

Despues de colocar los datos, ejecutar:

```powershell
python scripts\verificar_repositorio.py --modo rapido
```

Si se anaden datos externos, se recomienda generar un manifiesto propio de esos datos con SHA-256 antes de entregarlos o enlazarlos.
