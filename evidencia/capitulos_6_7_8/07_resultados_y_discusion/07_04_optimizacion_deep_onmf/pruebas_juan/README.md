# Tres pruebas de distribución Deep-ONMF para Juan

Esta carpeta implementa únicamente las tres comprobaciones solicitadas:

- `15-10-5`
- `10-6-4`
- `8-5-3`

La distribución original `9-8-7` se toma como referencia desde
`../resultados_punto3_validacion`. Los resultados originales no se modifican.

## Qué permanece constante

- 1000 audios de Yaseen: 200 por clase.
- Cinco folds estratificados idénticos a los originales.
- 800 audios de entrenamiento y 200 de test en cada fold.
- 60 iteraciones ONMF por capa.
- Penalización ortogonal `0.05`.
- Semilla `42`.
- Misma STFT, SVM, KNN y UjaNet.

Solo se modifica `rangos_deep_onmf`.

## Ejecución completa

Desde `Implementacion_last`:

```powershell
py ".\3 pruebas de juan\ejecutar_tres_pruebas.py" --datos "..\Bases de Datos"
```

También puede ejecutarse cada distribución por separado:

```powershell
py ".\3 pruebas de juan\prueba_15_10_5.py" --datos "..\Bases de Datos"
py ".\3 pruebas de juan\prueba_10_6_4.py" --datos "..\Bases de Datos"
py ".\3 pruebas de juan\prueba_8_5_3.py" --datos "..\Bases de Datos"
```

El sistema guarda cada fold nada más terminar. Si una ejecución se interrumpe,
el mismo comando continúa desde los resultados ya válidos.

## Verificación reducida

```powershell
py ".\3 pruebas de juan\verificar_pruebas.py"
```

La salida `resultados_verificacion` es una prueba técnica y no debe presentarse
como resultado científico.

## Salida final

El documento entregable se genera en:

`resultados/RESULTADOS_3_PRUEBAS_JUAN_TABLA_COMPLETA.pdf`

Los CSV conservan la precisión numérica completa. El PDF muestra cuatro
decimales y contiene un manifiesto SHA-256 que enlaza cada tabla con su CSV.

## Nota sobre UjaNet y 8-5-3

La arquitectura contiene dos MaxPool2D. Una dimensión de tamaño 3 no puede
atravesar ambos pooling sin llegar a tamaño cero. Para esa distribución se
añade una única fila o columna de ceros hasta tamaño 4. El procedimiento queda
registrado en `auditoria_adaptacion_ujanet.csv`, no inventa características y
mantiene intacta la arquitectura.

