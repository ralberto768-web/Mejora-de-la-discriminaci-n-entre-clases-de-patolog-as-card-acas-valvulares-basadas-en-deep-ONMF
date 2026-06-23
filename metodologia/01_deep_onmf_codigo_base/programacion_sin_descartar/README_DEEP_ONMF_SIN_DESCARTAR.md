# PROGRAMACION DEEP-ONMF SIN DESCARTAR

Esta es la programacion que vamos a usar cuando queramos trabajar con toda la base de datos disponible.

La diferencia importante frente al modo literal del articulo es esta:

- El articulo descarta audios menores de 2 segundos.
- En esta programacion no se descarta ningun WAV.
- Si un audio dura menos de 2 segundos, se rellena con ceros hasta alcanzar 2 segundos.
- Asi se conservan tambien las muestras cortas de MR, MS y MVP.

Archivos:

- `EJECUTAR_DEEP_ONMF_SIN_DESCARTAR.bat`: genera Figura 5, Tabla 2, Figura 7 y Figura 11D usando deep-ONMF y toda la base.
- `EJECUTAR_TESTEOS_FIGURA_11D_SIN_DESCARTAR.bat`: ejecuta varios testeos de la Figura 11D sin descartar audios cortos.

Resultados:

- El deep-ONMF principal se guarda en `resultados/resultadoN-deep_onmf_sin_descartar_menores_2s`.
- Los testeos de la Figura 11D se guardan en `resultados_deep_onmf/resultadoN-testeos_deep_onmf_figura11D_sin_descartar`.

Lectura correcta:

Esta programacion es mas adecuada para tu TFG si quieres justificar que la base disponible se usa completa. No es una copia literal del descarte del articulo, sino una adaptacion explicita y documentada para no perder muestras.
