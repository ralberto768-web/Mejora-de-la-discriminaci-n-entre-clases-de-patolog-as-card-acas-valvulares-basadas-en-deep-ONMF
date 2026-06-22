# Ultima prueba de Juan

Este proyecto amplia exclusivamente el analisis multiclase de Deep-ONMF.

Objetivos:

- completar 50 arquitecturas decrecientes de cuatro capas;
- completar 50 arquitecturas decrecientes de cinco capas;
- reunirlas con las 149 configuraciones anteriores;
- escoger diez configuraciones principales, incluyendo `9-8-7`;
- evaluar sus diez versiones crecientes;
- generar matrices de confusion agregadas sobre las 1000 predicciones fuera
  de muestra de los cinco folds.

Ejecucion completa desde `Implementacion_last`:

```powershell
py ".\3 pruebas de juan\prueba para encontrar mejor solucion\ultima prueba juan\ejecutar_ultima_prueba.py" --datos "..\Bases de Datos"
```

La ejecucion es reanudable. El modo rapido es solo una prueba tecnica:

```powershell
py ".\3 pruebas de juan\prueba para encontrar mejor solucion\ultima prueba juan\ejecutar_ultima_prueba.py" --rapido --limite-por-clase 2
```
