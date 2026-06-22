# Busqueda de la mejor configuracion Deep-ONMF

Este proyecto es independiente de las tres pruebas anteriores. Busca
arquitecturas Deep-ONMF de dos a cinco capas, conservando el protocolo de
1000 audios, cinco folds, 60 iteraciones, penalizacion 0.05 y semilla 42.

Cada arquitectura evalua obligatoriamente su matriz W y su matriz temporal
final H con SVM, KNN y UjaNet, tanto en binario como en multiclase.

La ejecucion completa se inicia desde `Implementacion_last`:

```powershell
py ".\3 pruebas de juan\prueba para encontrar mejor solucion\ejecutar_busqueda_completa.py" --datos "..\Bases de Datos"
```

La busqueda es reanudable. Si se interrumpe, el mismo comando continua desde
la ultima configuracion o fold completado.

El PDF final se genera en:

`resultados/RESULTADOS_BUSQUEDA_MEJOR_CONFIGURACION_DEEP_ONMF.pdf`

Para verificar rapidamente el flujo sin producir resultados entregables:

```powershell
py ".\3 pruebas de juan\prueba para encontrar mejor solucion\ejecutar_busqueda_completa.py" --rapido --limite-por-clase 2
```
