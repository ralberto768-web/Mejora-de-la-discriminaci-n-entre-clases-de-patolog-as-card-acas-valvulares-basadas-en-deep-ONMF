from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write(rel: str, text: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> int:
    write(
        "README.md",
        """
# Mejora de la discriminacion entre clases de patologias cardiacas valvulares basadas en Deep-ONMF

Este repositorio contiene el codigo, las configuraciones, los resultados y la documentacion tecnica de un Trabajo Fin de Grado sobre clasificacion automatica de sonidos cardiacos. El trabajo estudia representaciones temporales obtenidas con Deep-ONMF y su uso junto con clasificadores de aprendizaje automatico para discriminar clases de patologias cardiacas valvulares.

El repositorio esta preparado para que un lector externo pueda entender que se ha hecho, que resultados se han obtenido y que archivos necesita revisar o ejecutar, sin depender de carpetas privadas ni de documentos locales del autor.

## Que incluye

- `metodologia/`: codigo, configuraciones y material tecnico del flujo Deep-ONMF y del sistema de clasificacion.
- `resultados/`: resultados organizados por tipo de experimento: bases de datos, metricas, validacion cruzada, optimizacion, escenario sin ruido, escenario con ruido AWGN y comparacion entre representaciones `H3` y `W`.
- `conclusiones/`: evidencias usadas para cerrar objetivos, limitaciones y lineas futuras.
- `informe_general/`: informe PDF y Markdown con una lectura integrada del trabajo y de los resultados principales.
- `docs/`: guias de lectura, reproducibilidad, datos externos y resultados esperados.
- `scripts/`: comprobaciones automaticas de entorno, integridad del repositorio y resumen de evidencias.
- `datos_externos/`: plantilla para colocar bases de datos que no se distribuyen directamente en GitHub.
- `github/`: manifiestos del repositorio, lista de archivos grandes y estado de la entrega.
- `verificacion/`: manifiestos y resumen de integridad de la evidencia experimental incluida.

## Lectura recomendada

Para entender el contenido sin ejecutar nada:

1. `docs/LECTURA_RAPIDA.md`
2. `informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf`
3. `docs/GUIA_ESTRUCTURA_REPOSITORIO.md`
4. `docs/RESULTADOS_ESPERADOS.md`

Para comprobar que la descarga esta completa:

```powershell
.\\run_all.bat todo
```

Para preparar una reproduccion completa de los experimentos:

```text
docs/DATOS_EXTERNOS.md
docs/REPRODUCIBILIDAD.md
```

La reproduccion completa requiere disponer de las bases de datos externas de sonidos cardiacos. Los audios fuente no se incluyen directamente en el repositorio por tamano y por posibles restricciones de licencia.

## Flujo experimental

El trabajo sigue este esquema general:

1. Preparacion de senales de fonocardiograma.
2. Extraccion de caracteristicas temporales mediante Deep-ONMF.
3. Obtencion de matrices internas del modelo, principalmente `H3` y `W`.
4. Clasificacion mediante UjaNet y comparacion con representaciones clasicas.
5. Evaluacion con validacion cruzada k-fold.
6. Analisis en escenario sin ruido y con ruido AWGN.
7. Comparacion entre informacion temporal y espectral, incluyendo el contraste `H3` frente a `W`.
8. Discusion de coste, dimension de entrada, robustez y limitaciones.

## Estructura de resultados

| Ruta | Contenido |
|---|---|
| `resultados/01_bases_de_datos_y_escenarios/` | Evidencia sobre bases de datos y escenarios de evaluacion. |
| `resultados/02_metricas_de_evaluacion/` | Definiciones, tablas y salidas relacionadas con metricas. |
| `resultados/03_validacion_cruzada_kfold/` | Particiones, validaciones y resultados de k-fold. |
| `resultados/04_optimizacion_deep_onmf/` | Configuraciones optimizadas por capas y dimensiones. |
| `resultados/04_optimizacion_deep_onmf_historico/` | Resultados historicos complementarios de la busqueda Deep-ONMF. |
| `resultados/05_escenario_sin_ruido/` | Evaluacion sobre datos limpios o escenario optimo. |
| `resultados/06_escenario_ruidoso_awgn/` | Evaluacion con ruido AWGN y distintos niveles SNR. |
| `resultados/07_comparacion_temporal_espectral_h3_w/` | Comparacion entre representaciones temporales y espectrales, incluyendo `H3` y `W`. |
| `resultados/08_discusion_resultados/` | Material para interpretar los resultados principales. |

## Comprobacion rapida

En Windows:

```powershell
.\\run_all.bat todo
```

Ese comando comprueba dependencias basicas, verifica archivos obligatorios y resume evidencias clave. Tambien se puede ejecutar por partes:

```powershell
python scripts\\comprobar_entorno.py
python scripts\\verificar_repositorio.py --modo rapido
python scripts\\resumen_resultados.py
```

## Git LFS

El repositorio usa Git LFS para ficheros grandes como `.npy`, `.npz`, `.pt`, `.zip`, `.pdf` y resultados pesados. Despues de clonar:

```powershell
git lfs install
git lfs pull
```

La lista de archivos grandes esta en `github/ARCHIVOS_GRANDES_GIT_LFS.csv`.

## Estado de la entrega

- Paquete preparado para GitHub con estructura publica y nombres descriptivos.
- Codigo, resultados, figuras, tablas, configuraciones y manifiestos incluidos.
- Documentacion redactada para lectura autonoma por parte de un tribunal o lector externo.
- Comprobacion automatica disponible mediante `run_all.bat todo`.
""",
    )

    write(
        "docs/LECTURA_RAPIDA.md",
        """
# Lectura rapida

Este documento resume como leer el repositorio sin conocer la organizacion interna usada durante el desarrollo del TFG.

## Objetivo del repositorio

El proyecto evalua un flujo de clasificacion de sonidos cardiacos basado en caracteristicas temporales extraidas con Deep-ONMF. La evidencia incluida permite revisar el metodo, los resultados experimentales, la comparacion entre representaciones y la reproducibilidad basica del paquete.

## Orden recomendado de lectura

1. `README.md`: vision general del proyecto y estructura de carpetas.
2. `informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf`: informe integrado con metodologia, resultados y discusion.
3. `docs/GUIA_ESTRUCTURA_REPOSITORIO.md`: relacion entre carpetas y bloques tecnicos.
4. `docs/RESULTADOS_ESPERADOS.md`: que deberia encontrar el lector en cada bloque de resultados.
5. `docs/REPRODUCIBILIDAD.md`: como comprobar el paquete y que hace falta para repetir ejecuciones completas.

## Que revisar primero

- Metodologia Deep-ONMF y clasificacion: `metodologia/`.
- Resultados de optimizacion: `resultados/04_optimizacion_deep_onmf/` y `resultados/04_optimizacion_deep_onmf_historico/`.
- Escenario sin ruido: `resultados/05_escenario_sin_ruido/`.
- Escenario con ruido AWGN: `resultados/06_escenario_ruidoso_awgn/`.
- Comparacion `H3` frente a `W`: `resultados/07_comparacion_temporal_espectral_h3_w/`.
- Integridad de archivos: `verificacion/` y `github/MANIFIESTO_REPOSITORIO.csv`.

## Comprobacion minima

Ejecuta desde la raiz del repositorio:

```powershell
.\\run_all.bat todo
```

Si el comando termina con verificacion correcta, la descarga contiene los archivos obligatorios y las evidencias principales estan localizables.
""",
    )

    write(
        "docs/GUIA_ESTRUCTURA_REPOSITORIO.md",
        """
# Guia de estructura del repositorio

Esta guia explica que contiene cada bloque del repositorio y para que se usa. Los nombres estan pensados para lectura publica: cada carpeta describe su funcion tecnica sin depender de rutas locales ni de documentos privados.

## Bloques principales

| Carpeta | Funcion |
|---|---|
| `metodologia/` | Material tecnico del metodo: Deep-ONMF, extraccion de matrices `W` y `H`, configuraciones y arquitectura de clasificacion. |
| `resultados/` | Evidencia experimental organizada por bases de datos, metricas, validacion, optimizacion, escenarios y comparaciones. |
| `conclusiones/` | Material de cierre: objetivos alcanzados, limitaciones y lineas futuras. |
| `informe_general/` | Informe PDF/Markdown que integra metodologia, resultados y discusion en un unico documento. |
| `docs/` | Guias de lectura para un evaluador externo. |
| `scripts/` | Comprobaciones automaticas y resumen de archivos clave. |
| `datos_externos/` | Carpeta preparada para colocar bases de datos externas no distribuidas en GitHub. |
| `verificacion/` | Manifiestos de integridad de evidencia experimental. |
| `github/` | Estado del paquete, manifiesto del repositorio y archivos gestionados con Git LFS. |

## Resultados experimentales

| Carpeta | Contenido esperado |
|---|---|
| `resultados/01_bases_de_datos_y_escenarios/` | Descripcion y auditoria de bases de datos, escenarios limpios y escenarios con AWGN. |
| `resultados/02_metricas_de_evaluacion/` | Accuracy, precision, recall, F1-score, matrices de confusion y metricas de separabilidad cuando aparecen. |
| `resultados/03_validacion_cruzada_kfold/` | Particiones y salidas de validacion cruzada. |
| `resultados/04_optimizacion_deep_onmf/` | Mejores configuraciones Deep-ONMF por numero de capas y dimensiones. |
| `resultados/04_optimizacion_deep_onmf_historico/` | Ejecuciones historicas complementarias conservadas para trazabilidad. |
| `resultados/05_escenario_sin_ruido/` | Resultados de clasificacion en condiciones limpias. |
| `resultados/06_escenario_ruidoso_awgn/` | Resultados bajo ruido AWGN y diferentes niveles SNR. |
| `resultados/07_comparacion_temporal_espectral_h3_w/` | Comparacion entre caracteristicas temporales y espectrales, con enfasis en `H3` frente a `W`. |
| `resultados/08_discusion_resultados/` | Graficas, tablas y notas utiles para interpretar los resultados. |

## Lectura tecnica sugerida

1. Revisar `metodologia/README.md` para entender el flujo de extraccion y clasificacion.
2. Revisar `resultados/README.md` y despues cada subcarpeta de resultados.
3. Consultar el informe `informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf` para ver una narrativa unificada.
4. Ejecutar `run_all.bat todo` para comprobar que la descarga conserva los archivos principales.
""",
    )

    write(
        "docs/GUIA_TRIBUNAL.md",
        """
# Guia para evaluacion externa

Esta guia esta pensada para una revision rapida por parte de un tribunal, tutor o lector externo. El repositorio puede revisarse sin ejecutar entrenamientos completos, porque incluye resultados ya generados, tablas, figuras y manifiestos de verificacion.

## Revision recomendada

1. Leer `README.md` para entender el objetivo y la estructura.
2. Abrir `informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf` para una vision completa del metodo y los resultados.
3. Consultar `docs/RESULTADOS_ESPERADOS.md` para saber donde esta cada tipo de evidencia.
4. Revisar `resultados/04_optimizacion_deep_onmf/`, `resultados/05_escenario_sin_ruido/`, `resultados/06_escenario_ruidoso_awgn/` y `resultados/07_comparacion_temporal_espectral_h3_w/`.
5. Ejecutar `run_all.bat todo` si se quiere comprobar que el repositorio descargado conserva los archivos obligatorios.

## Que se puede comprobar directamente

- Codigo y configuraciones del flujo Deep-ONMF.
- Resultados de optimizacion por capas y dimensiones.
- Resultados en escenario sin ruido.
- Resultados en escenario con ruido AWGN.
- Comparacion entre representaciones temporales y espectrales.
- Comparacion entre matrices `H3` y `W`.
- Figuras, tablas, matrices de confusion, CSV de metricas y documentos explicativos.

## Que no se incluye directamente

Los audios fuente y bases de datos completas no se suben al repositorio por tamano y por posibles restricciones externas. Para repetir ejecuciones completas deben colocarse manualmente en `datos_externos/`, siguiendo `docs/DATOS_EXTERNOS.md`.

## Separacion entre evidencia e interpretacion

- Las carpetas de `resultados/` contienen evidencia experimental: metricas, tablas, figuras y salidas generadas.
- `informe_general/` y `docs/` contienen la explicacion y la orientacion de lectura.
- `conclusiones/` contiene material de cierre y lineas futuras.

Esta separacion facilita revisar los datos primero y valorar la interpretacion despues.
""",
    )

    write(
        "docs/RESULTADOS_ESPERADOS.md",
        """
# Resultados esperados

Este documento indica que deberia encontrar un lector al revisar cada bloque experimental.

## Optimizacion Deep-ONMF

Ruta principal: `resultados/04_optimizacion_deep_onmf/`

Contenido esperado:

- configuraciones probadas por numero de capas;
- dimensiones empleadas en las matrices internas;
- tablas con mejores configuraciones;
- resultados historicos complementarios en `resultados/04_optimizacion_deep_onmf_historico/`;
- figuras y documentos de apoyo cuando estan disponibles.

## Escenario sin ruido

Ruta principal: `resultados/05_escenario_sin_ruido/`

Contenido esperado:

- resultados de clasificacion sobre datos limpios;
- comparacion con representaciones clasicas cuando existe evidencia disponible;
- metricas agregadas y matrices de confusion;
- ficheros CSV, imagenes y documentos de interpretacion.

## Escenario ruidoso con AWGN

Ruta principal: `resultados/06_escenario_ruidoso_awgn/`

Contenido esperado:

- resultados por nivel de SNR;
- comparacion de robustez frente a ruido;
- tablas de metricas y figuras resumen;
- salidas intermedias necesarias para auditar la evaluacion.

## Comparacion temporal-espectral y H3-W

Ruta principal: `resultados/07_comparacion_temporal_espectral_h3_w/`

Contenido esperado:

- comparacion de representaciones temporales frente a espectrales;
- evidencias de separabilidad mediante nubes de puntos o proyecciones;
- resultados comparando matrices `H3` y `W`;
- tablas CSV con metricas de apoyo.

## Discusion y cierre

Rutas principales:

- `resultados/08_discusion_resultados/`
- `conclusiones/`
- `informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf`

Contenido esperado:

- sintesis de resultados relevantes;
- ventajas y limitaciones de las caracteristicas temporales;
- relacion entre dimension de entrada, coste computacional y rendimiento;
- lineas futuras razonables a partir de la evidencia incluida.
""",
    )

    write(
        "docs/REPRODUCIBILIDAD.md",
        """
# Reproducibilidad

El repositorio permite dos niveles de comprobacion: revision de resultados ya generados y reproduccion completa con datos externos.

## Comprobacion del paquete descargado

Desde la raiz del repositorio:

```powershell
.\\run_all.bat todo
```

Este comando comprueba:

- dependencias basicas de Python;
- existencia de archivos obligatorios;
- coherencia con el manifiesto del repositorio;
- localizacion de evidencias principales.

## Reproduccion completa

Para repetir entrenamientos o extracciones desde cero se necesita:

1. Instalar dependencias de `requirements.txt` o `environment.yml`.
2. Descargar correctamente archivos gestionados por Git LFS.
3. Colocar las bases de datos externas en `datos_externos/`.
4. Revisar configuraciones y scripts en `metodologia/` y `resultados/`.
5. Ejecutar los flujos correspondientes segun el experimento que se quiera repetir.

## Datos externos

Los audios fuente no estan incluidos directamente en GitHub. La carpeta `datos_externos/` contiene la plantilla de colocacion y `docs/DATOS_EXTERNOS.md` explica como prepararla.

## Integridad

- `github/MANIFIESTO_REPOSITORIO.csv` lista archivos versionados y tamanos.
- `github/ARCHIVOS_GRANDES_GIT_LFS.csv` lista archivos grandes esperados en Git LFS.
- `verificacion/` conserva manifiestos de la evidencia experimental incluida.

La comprobacion rapida no reentrena modelos. Su objetivo es confirmar que el paquete descargado esta completo y que la documentacion apunta a archivos existentes.
""",
    )

    write(
        "docs/DATOS_EXTERNOS.md",
        """
# Datos externos

Las bases de datos completas de sonidos cardiacos no se distribuyen dentro del repositorio. Esta decision evita subir audios pesados y respeta posibles restricciones de licencia o redistribucion.

## Uso de la carpeta `datos_externos/`

Coloca aqui las bases de datos necesarias para repetir extracciones o entrenamientos completos. La estructura concreta puede depender de la fuente de datos disponible, pero debe mantenerse separada del codigo y de los resultados ya generados.

Ejemplo de organizacion:

```text
datos_externos/
  yaseen/
    audios/
    etiquetas/
  ruido_awgn/
    configuraciones/
```

## Revision sin datos externos

No hace falta disponer de los audios fuente para revisar el repositorio. Las carpetas `resultados/`, `informe_general/`, `docs/` y `verificacion/` contienen la evidencia ya generada.

## Repeticion de experimentos

Para repetir un experimento completo:

1. Coloca los datos externos en esta carpeta.
2. Comprueba dependencias con `python scripts\\comprobar_entorno.py`.
3. Revisa la configuracion del experimento en `metodologia/` o `resultados/`.
4. Ejecuta el script correspondiente.
""",
    )

    write(
        "docs/GUIA_CONTENIDO.md",
        """
# Guia de contenido

El repositorio agrupa el trabajo en cuatro bloques de lectura.

## 1. Metodo

`metodologia/` contiene el material necesario para explicar el sistema: extraccion Deep-ONMF, matrices `W` y `H`, seleccion de `H3` como representacion temporal de referencia y clasificacion con UjaNet u otros modelos.

## 2. Resultados

`resultados/` contiene la evidencia experimental separada por escenario y comparacion. Esta carpeta debe usarse para localizar metricas, figuras, tablas, matrices de confusion, predicciones y configuraciones.

## 3. Informe integrado

`informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf` ofrece una lectura continua del trabajo. Sirve como documento de apoyo para revisar el repositorio sin abrir manualmente cada subcarpeta.

## 4. Verificacion

`run_all.bat todo`, `github/MANIFIESTO_REPOSITORIO.csv` y `verificacion/` permiten comprobar que el paquete conserva los archivos esperados.
""",
    )

    write(
        "metodologia/README.md",
        """
# Metodologia

Esta carpeta recoge el material tecnico usado para describir el sistema de clasificacion basado en Deep-ONMF.

## Contenido

- Codigo y configuraciones para extraccion de caracteristicas temporales.
- Evidencias relacionadas con matrices `W`, `H` y `H3`.
- Material de apoyo para explicar la arquitectura de clasificacion.
- Documentos y resultados intermedios necesarios para entender el flujo experimental.

## Uso recomendado

Lee esta carpeta antes de revisar `resultados/`. Aqui se encuentra el contexto tecnico que permite interpretar las metricas, figuras y tablas experimentales.
""",
    )

    write(
        "resultados/README.md",
        """
# Resultados

Esta carpeta contiene los resultados experimentales organizados por tipo de evaluacion.

## Carpetas

- `01_bases_de_datos_y_escenarios/`: bases de datos y escenarios de evaluacion.
- `02_metricas_de_evaluacion/`: metricas y salidas de medida.
- `03_validacion_cruzada_kfold/`: validacion cruzada y particiones.
- `04_optimizacion_deep_onmf/`: resultados principales de optimizacion.
- `04_optimizacion_deep_onmf_historico/`: ejecuciones historicas complementarias.
- `05_escenario_sin_ruido/`: resultados en condiciones limpias.
- `06_escenario_ruidoso_awgn/`: resultados con ruido AWGN.
- `07_comparacion_temporal_espectral_h3_w/`: comparacion temporal-espectral y `H3` frente a `W`.
- `08_discusion_resultados/`: material de apoyo para la discusion.

Cada subcarpeta conserva nombres originales de archivos cuando son utiles para trazabilidad.
""",
    )

    write(
        "conclusiones/README.md",
        """
# Conclusiones y lineas futuras

Esta carpeta agrupa evidencias y documentos usados para cerrar el trabajo.

## Contenido

- Relacion entre objetivos y resultados obtenidos.
- Limitaciones detectadas durante la evaluacion.
- Posibles lineas futuras de mejora.
- Material complementario para la discusion final.
""",
    )

    write(
        "verificacion/README.md",
        """
# Verificacion

Esta carpeta contiene manifiestos y resumenes usados para comprobar la integridad de la evidencia experimental incluida.

## Archivos principales

- `MANIFIESTO_ARCHIVOS.csv`: relacion de archivos de evidencia con tamano y hash cuando esta disponible.
- `RESUMEN_COPIA.md`: resumen de la copia de evidencia.

La comprobacion publica del repositorio se ejecuta desde la raiz con:

```powershell
.\\run_all.bat todo
```
""",
    )

    write(
        "informe_general/VERIFICACION_INFORME.md",
        """
# Verificacion del informe general

El informe principal del repositorio es:

- `informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.pdf`
- `informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.md`

El PDF se genera a partir de una version Markdown equivalente y se mantiene como documento de lectura rapida para evaluacion externa. Las figuras y tablas principales usadas como apoyo se conservan en `figuras_principales/` y `tablas_principales/`.
""",
    )

    nested = {
        "metodologia/03_codigo_y_configuracion/README.md": "Codigo y configuracion",
        "conclusiones/01_cierre_objetivos/README.md": "Cierre de objetivos",
        "conclusiones/02_limitaciones/README.md": "Limitaciones",
        "conclusiones/03_lineas_futuras/README.md": "Lineas futuras",
    }
    for rel, title in nested.items():
        if (ROOT / rel).exists():
            write(
                rel,
                f"""
# {title}

Carpeta de apoyo del repositorio para conservar evidencia, codigo o documentos relacionados con este bloque. Los archivos mantienen sus nombres originales cuando ayudan a rastrear de donde procede cada resultado.
""",
            )

    write(
        "informe_general/INFORME_GENERAL_RESULTADOS_DEEP_ONMF.md",
        """
# Informe general de resultados Deep-ONMF

## 1. Proposito del repositorio

Este informe resume el material incluido en el repositorio para evaluar un sistema de clasificacion de sonidos cardiacos basado en Deep-ONMF. El objetivo es revisar de forma ordenada el metodo, las evidencias experimentales y las conclusiones tecnicas sin depender de rutas locales ni de documentos privados.

## 2. Metodo Deep-ONMF y clasificacion

El flujo experimental parte de senales de fonocardiograma. A partir de ellas se extraen caracteristicas temporales mediante Deep-ONMF, obteniendo matrices internas como `W`, `H` y `H3`. La representacion `H3` se usa como entrada temporal principal para el sistema de clasificacion, mientras que `W` se conserva para comparaciones de separabilidad y rendimiento.

El repositorio conserva codigo, configuraciones y resultados intermedios en `metodologia/`. Esta carpeta debe revisarse para entender como se generan las representaciones y como se conectan con la arquitectura de clasificacion.

## 3. Bases de datos y escenarios

La evaluacion distingue dos condiciones principales. La primera corresponde a un escenario sin ruido, usado como referencia de funcionamiento. La segunda introduce ruido AWGN con distintos niveles SNR para estudiar robustez. La evidencia asociada esta organizada en `resultados/01_bases_de_datos_y_escenarios/`, `resultados/05_escenario_sin_ruido/` y `resultados/06_escenario_ruidoso_awgn/`.

Los audios fuente completos no se distribuyen en GitHub. La reproduccion desde cero requiere colocarlos en `datos_externos/` siguiendo la guia correspondiente.

## 4. Metricas y metodologia de evaluacion

Las salidas experimentales incluyen metricas de clasificacion como accuracy, precision, recall y F1-score, junto con matrices de confusion y tablas resumen cuando estan disponibles. La evaluacion se organiza con validacion cruzada k-fold para reducir dependencia de una unica particion.

Las carpetas relevantes son `resultados/02_metricas_de_evaluacion/` y `resultados/03_validacion_cruzada_kfold/`.

## 5. Optimizacion Deep-ONMF

La optimizacion estudia el efecto del numero de capas y de las dimensiones internas del modelo. La evidencia principal se encuentra en `resultados/04_optimizacion_deep_onmf/`. Las ejecuciones historicas complementarias se conservan en `resultados/04_optimizacion_deep_onmf_historico/` para trazabilidad.

En la revision de resultados conviene separar dos aspectos: por un lado, las configuraciones objetivamente evaluadas; por otro, la interpretacion sobre que configuracion es mas defendible para el sistema final.

## 6. Resultados en escenario sin ruido

El escenario sin ruido permite comparar el comportamiento del enfoque temporal Deep-ONMF frente a representaciones clasicas. La evidencia se guarda en `resultados/05_escenario_sin_ruido/` e incluye tablas, figuras y salidas de clasificacion cuando estan disponibles.

Este bloque sirve como referencia para valorar si la representacion temporal conserva informacion discriminativa suficiente con menor dimension de entrada.

## 7. Resultados en escenario ruidoso

El escenario con AWGN analiza la estabilidad del sistema ante degradacion artificial de la senal. La carpeta `resultados/06_escenario_ruidoso_awgn/` agrupa resultados por nivel de ruido y permite comparar el comportamiento relativo de las representaciones.

La interpretacion debe centrarse en tendencias de robustez y no solo en un valor puntual de una metrica.

## 8. Comparacion temporal-espectral y H3-W

La carpeta `resultados/07_comparacion_temporal_espectral_h3_w/` contiene la evidencia usada para comparar caracteristicas temporales frente a espectrales y para contrastar el uso de `H3` frente a `W`.

Esta comparacion tiene dos lecturas complementarias: representabilidad visual mediante proyecciones o nubes de puntos, y rendimiento cuantitativo mediante metricas de clasificacion. Ambas lecturas deben mantenerse separadas para evitar conclusiones excesivas.

## 9. Discusion tecnica

Los resultados deben interpretarse considerando rendimiento, dimension de entrada, coste computacional y robustez. Aunque las representaciones temporales no siempre superen a las representaciones clasicas en todas las metricas, pueden aportar una entrada de menor dimension y un coste potencialmente menor.

La carpeta `resultados/08_discusion_resultados/` conserva material util para esta lectura.

## 10. Conclusiones y lineas futuras

El repositorio permite defender un flujo completo de extraccion temporal, clasificacion y evaluacion bajo condiciones limpias y ruidosas. Las lineas futuras pasan por ampliar bases de datos, refinar la optimizacion de capas y dimensiones, estudiar otras arquitecturas de clasificacion y mejorar la validacion externa.

El material de cierre esta en `conclusiones/`.

## 11. Verificacion

La integridad del paquete puede comprobarse con:

```powershell
.\\run_all.bat todo
```

Los manifiestos estan en `github/` y `verificacion/`. El objetivo de esas comprobaciones no es reentrenar modelos, sino confirmar que el repositorio conserva los archivos obligatorios y las evidencias principales.
""",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
