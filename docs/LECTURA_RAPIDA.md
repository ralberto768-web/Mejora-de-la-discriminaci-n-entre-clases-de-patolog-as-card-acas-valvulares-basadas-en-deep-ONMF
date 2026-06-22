# Lectura rápida del repositorio

Este documento está pensado para una persona que llega al repositorio desde GitHub y no tiene todavía delante la memoria del TFG.

## Qué es este trabajo

El repositorio acompaña al Trabajo Fin de Grado titulado:

**Mejora de la discriminación entre clases de patologías cardíacas valvulares basadas en deep-ONMF**

El trabajo estudia señales de sonido cardíaco y evalúa si una representación temporal obtenida mediante Deep-ONMF ayuda a diferenciar clases de patologías valvulares.

## Idea general

El flujo experimental es:

1. Partir de audios de sonidos cardíacos.
2. Preprocesar las señales.
3. Extraer características mediante Deep-ONMF.
4. Usar matrices internas del modelo, principalmente `H3`, como entrada de clasificadores.
5. Comparar esas características temporales con representaciones clásicas.
6. Evaluar en escenario sin ruido y en escenario ruidoso con AWGN.
7. Analizar métricas, matrices de confusión, figuras y tablas.

## Qué significan W, H y H3

En la factorización usada por Deep-ONMF aparecen matrices internas:

- `W`: matriz asociada a bases o patrones aprendidos.
- `H`: matriz asociada a activaciones temporales.
- `H3`: representación obtenida en la tercera capa del modelo Deep-ONMF y usada como una de las entradas principales de clasificación.

La comparación `H3` frente a `W` es importante porque permite valorar si las activaciones temporales aportan más información discriminativa que las bases.

## Qué es UjaNet

UjaNet es la arquitectura CNN usada en el trabajo para clasificar las representaciones generadas. En este repositorio aparecen resultados con UjaNet y con otros clasificadores de referencia.

## Qué puede comprobar el lector

Sin bases de datos externas:

- revisar el código usado;
- revisar tablas y resultados ya generados;
- abrir el documento global de evaluación;
- comprobar manifiestos, tamaños y hashes;
- verificar que los resultados principales están presentes.

Con bases de datos externas:

- repetir extracciones;
- regenerar representaciones;
- volver a entrenar/evaluar clasificadores;
- comparar nuevos resultados con los resultados esperados.

## Ruta recomendada de lectura

1. `README.md`
2. `documento_global/DOCUMENTO_EVALUACION_CAPITULOS_6_7_8.pdf`
3. `docs/MAPA_INDICE_TFG.md`
4. `docs/RESULTADOS_ESPERADOS.md`
5. `docs/REPRODUCIBILIDAD.md`
6. `evidencia/capitulos_6_7_8/00_README_GENERAL.md`

## Qué no contiene directamente

El repositorio no fuerza la inclusión de todos los audios fuente originales. Esos datos deben colocarse en `datos_externos/` si se quiere reproducir todo desde cero. Esta decisión evita mezclar código y resultados con datos externos pesados o sujetos a restricciones de distribución.

