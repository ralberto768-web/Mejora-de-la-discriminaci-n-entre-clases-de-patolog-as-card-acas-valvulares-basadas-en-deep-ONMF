# Correspondencia entre índice del TFG y repositorio

Este documento relaciona la estructura del repositorio con los apartados finales de la memoria. La numeración `6`, `7` y `8` corresponde a los bloques de materiales y métodos, resultados y discusión, y conclusiones.

## 6. Materiales y métodos

Ruta:

```text
evidencia/capitulos_6_7_8/06_materiales_y_metodos/
```

Qué contiene:

- código de extracción de características;
- implementación y variantes de Deep-ONMF;
- generación de matrices `W` y `H`;
- material para explicar el uso de `H3` o `W` como entrada de clasificación;
- código y configuración asociados a UjaNet.

Uso en la memoria:

Este bloque sirve para explicar el método completo: desde el sonido cardíaco hasta la representación usada por el clasificador.

## 7. Resultados y discusión

Ruta:

```text
evidencia/capitulos_6_7_8/07_resultados_y_discusion/
```

### 7.1 Bases de datos

Ruta:

```text
07_01_bases_de_datos/
```

Incluye auditorías, particiones y material relacionado con el escenario sin ruido y los escenarios con ruido AWGN.

### 7.2 Métricas

Ruta:

```text
07_02_metricas/
```

Incluye métricas de clasificación, separabilidad y comparativas entre representaciones.

### 7.3 Metodología de evaluación k-fold

Ruta:

```text
07_03_metodologia_evaluacion_kfold/
```

Incluye particiones, configuraciones y evidencias de la validación cruzada usada para evaluar los modelos.

### 7.4 Optimización Deep-ONMF

Ruta:

```text
07_04_optimizacion_deep_onmf/
```

Incluye resultados de búsqueda de configuraciones por número de capas y dimensiones. Este bloque es el principal para justificar qué configuración Deep-ONMF se usa después.

### 7.5 Escenario real

Ruta:

```text
07_05_escenario_real/
```

Incluye resultados con datos sin ruido y comparativas entre representaciones clásicas y características Deep-ONMF.

### 7.6 Escenario ruidoso AWGN

Ruta:

```text
07_06_escenario_ruidoso_awgn/
```

Incluye resultados al añadir ruido AWGN y evaluar distintos niveles SNR.

### 7.7 Características espectrales frente a temporales

Ruta:

```text
07_07_espectrales_vs_temporales_h_vs_w/
```

Incluye evidencia para comparar `H3` y `W`, además de resultados que permiten discutir la utilidad de las características temporales frente a representaciones espectrales.

### 7.8 Discusión

Ruta:

```text
07_08_discusion/
```

Incluye material de apoyo para interpretar los resultados, separar rendimiento de clasificación, separabilidad visual y coste/dimensión de las representaciones.

## 8. Conclusiones y líneas futuras

Ruta:

```text
evidencia/capitulos_6_7_8/08_conclusiones_y_lineas_futuras/
```

Qué contiene:

- documentos finales;
- material para cerrar objetivos;
- limitaciones;
- posibles líneas futuras.

Uso en la memoria:

Este bloque no introduce experimentos nuevos. Sirve para cerrar la interpretación de los resultados y proponer mejoras posteriores.

