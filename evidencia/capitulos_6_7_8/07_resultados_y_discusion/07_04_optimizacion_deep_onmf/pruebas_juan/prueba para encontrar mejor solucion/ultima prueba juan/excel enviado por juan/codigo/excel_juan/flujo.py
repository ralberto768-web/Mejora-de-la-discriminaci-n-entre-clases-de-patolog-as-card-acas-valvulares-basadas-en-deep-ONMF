from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import fitz
import numpy as np
import pandas as pd

from codigo.clasificadores import crear_folds, etiquetas_numericas
from codigo.configuracion import cargar_configuracion

from .arquitecturas import construir_plan, convertir_etiqueta, clave
from .evaluacion import (
    buscar_fuente_anterior,
    evaluar_configuracion,
    predicciones_completas,
    preparar_cache_stft,
    reutilizar_fuente,
)
from .informes import generar_pdf_resumen, generar_pdf_todas
from .resultados import (
    carpeta_configuracion,
    consolidar_resultados,
    generar_matrices_csv,
    seleccionar_resumenes,
)


RAIZ_EXCEL = Path(__file__).resolve().parents[2]
RAIZ_ULTIMA = RAIZ_EXCEL.parent
RAIZ_BUSQUEDA = RAIZ_ULTIMA.parent
RAIZ_PRUEBAS = RAIZ_BUSQUEDA.parent
RAIZ_IMPLEMENTACION = RAIZ_PRUEBAS.parent
RAIZ_TFG = RAIZ_IMPLEMENTACION.parent
CONFIGURACION = RAIZ_IMPLEMENTACION / "configuracion_experimento.json"
RESULTADOS_PUNTO3 = RAIZ_IMPLEMENTACION / "resultados_punto3_validacion"
EXCEL_LOCAL = RAIZ_EXCEL / "configuraciones_arquitecturas.xlsx"


def _metadatos_maestros() -> pd.DataFrame:
    ruta = (
        RESULTADOS_PUNTO3
        / "representaciones"
        / "DeepONMF_W"
        / "metadatos.csv"
    )
    tabla = pd.read_csv(ruta, encoding="utf-8-sig")
    if len(tabla) != 1000 or tabla["clase"].value_counts().to_dict() != {
        "N": 200,
        "AS": 200,
        "MR": 200,
        "MS": 200,
        "MVP": 200,
    }:
        raise AssertionError("Los metadatos maestros no contienen 1000 audios")
    return tabla


def preparar(
    carpeta_resultados: Path,
    carpeta_datos: Path,
) -> tuple[pd.DataFrame, Path, pd.DataFrame]:
    config = cargar_configuracion(CONFIGURACION)
    if (
        config.iteraciones_onmf != 60
        or abs(config.penalizacion_ortogonal - 0.05) > 1e-12
        or config.semilla != 42
        or config.folds != 5
    ):
        raise AssertionError("La configuracion base no coincide con el encargo")
    plan = construir_plan(EXCEL_LOCAL)
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    (carpeta_resultados / "tablas_csv").mkdir(parents=True, exist_ok=True)
    plan.to_csv(
        carpeta_resultados / "tablas_csv" / "plan_372_orientaciones.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadatos = _metadatos_maestros()
    ruta_stft, metadatos_stft = preparar_cache_stft(
        carpeta_datos,
        metadatos,
        config,
        carpeta_resultados / "cache_compartida",
    )
    claves_maestras = (
        metadatos["clase"].astype(str)
        + "/"
        + metadatos["archivo"].astype(str)
    ).tolist()
    claves_stft = (
        metadatos_stft["clase"].astype(str)
        + "/"
        + metadatos_stft["archivo"].astype(str)
    ).tolist()
    if claves_maestras != claves_stft:
        raise AssertionError("La cache STFT no conserva el orden maestro")
    auditoria = pd.DataFrame(
        [
            {"comprobacion": "arquitecturas_excel", "valor": 186},
            {"comprobacion": "orientaciones_totales", "valor": len(plan)},
            {"comprobacion": "audios", "valor": len(metadatos)},
            {"comprobacion": "folds", "valor": config.folds},
            {
                "comprobacion": "iteraciones_onmf",
                "valor": config.iteraciones_onmf,
            },
            {
                "comprobacion": "penalizacion_ortogonal",
                "valor": config.penalizacion_ortogonal,
            },
            {"comprobacion": "semilla", "valor": config.semilla},
        ]
    )
    auditoria.to_csv(
        carpeta_resultados / "auditoria_preparacion.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return plan, ruta_stft, metadatos


def ejecutar_worker(
    indice_worker: int,
    numero_workers: int,
    carpeta_resultados: Path,
    carpeta_datos: Path,
) -> None:
    plan, ruta_stft, metadatos = preparar(carpeta_resultados, carpeta_datos)
    config = cargar_configuracion(CONFIGURACION)
    _, y_multi = etiquetas_numericas(metadatos)
    folds = crear_folds(y_multi, config)
    # El plan maestro alterna decreciente/creciente. Agrupar primero por
    # sentido antes de repartir evita que unos workers reciban solo un
    # sentido, que puede ser sensiblemente mas costoso que el otro.
    plan_trabajo = pd.concat(
        [
            plan.loc[plan["sentido"].eq("decreciente")],
            plan.loc[plan["sentido"].eq("creciente")],
        ],
        ignore_index=True,
    )
    tareas = plan_trabajo.iloc[
        [
            indice
            for indice in range(len(plan_trabajo))
            if indice % numero_workers == indice_worker
        ]
    ]
    estado_ruta = carpeta_resultados / f"estado_worker_{indice_worker}.json"
    completadas = 0
    reutilizadas = 0
    calculadas = 0
    for tarea in tareas.itertuples(index=False):
        rangos = convertir_etiqueta(str(tarea.distribucion))
        carpeta = carpeta_configuracion(
            carpeta_resultados,
            str(tarea.sentido),
            str(tarea.distribucion),
        )
        carpeta_pred = (
            carpeta
            / "pred"
            / "UjaNet"
            / f"H{len(rangos)}"
        )
        ruta_resumen = (
            carpeta / "metricas" / "resumen_metricas_multiclase.csv"
        )
        if ruta_resumen.exists() and predicciones_completas(carpeta_pred):
            completadas += 1
        else:
            fuente = buscar_fuente_anterior(rangos, RAIZ_ULTIMA)
            if fuente is not None:
                reutilizar_fuente(fuente, rangos, carpeta)
                reutilizadas += 1
            else:
                evaluar_configuracion(
                    rangos,
                    ruta_stft,
                    metadatos,
                    folds,
                    config,
                    carpeta,
                )
                calculadas += 1
            completadas += 1
        estado_ruta.write_text(
            json.dumps(
                {
                    "worker": indice_worker,
                    "numero_workers": numero_workers,
                    "asignadas": len(tareas),
                    "completadas": completadas,
                    "reutilizadas_en_esta_ejecucion": reutilizadas,
                    "calculadas_en_esta_ejecucion": calculadas,
                    "ultima": str(tarea.distribucion),
                    "sentido": str(tarea.sentido),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[worker {indice_worker}] {completadas}/{len(tareas)} "
            f"{tarea.sentido} {tarea.distribucion}"
        )


def lanzar_workers(
    numero_workers: int,
    carpeta_resultados: Path,
    carpeta_datos: Path,
) -> None:
    procesos = []
    archivos = []
    entorno = os.environ.copy()
    entorno.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    script = RAIZ_EXCEL / "ejecutar_worker_excel_juan.py"
    for indice in range(numero_workers):
        ruta_log = carpeta_resultados / f"worker_{indice}.log"
        archivo = ruta_log.open("a", encoding="utf-8")
        archivos.append(archivo)
        procesos.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(script),
                    "--indice",
                    str(indice),
                    "--workers",
                    str(numero_workers),
                    "--salida",
                    str(carpeta_resultados),
                    "--datos",
                    str(carpeta_datos),
                ],
                stdout=archivo,
                stderr=subprocess.STDOUT,
                env=entorno,
                cwd=str(RAIZ_EXCEL),
            )
        )
    errores = []
    for indice, proceso in enumerate(procesos):
        codigo = proceso.wait()
        if codigo != 0:
            errores.append((indice, codigo))
    for archivo in archivos:
        archivo.close()
    if errores:
        raise RuntimeError(f"Fallaron trabajadores: {errores}")


def _tabla_pdf(resumen: pd.DataFrame) -> pd.DataFrame:
    tabla = resumen.copy()
    tabla["posicion"] = tabla["pareja"]
    return tabla


def generar_entregables(carpeta_resultados: Path) -> list[Path]:
    plan = pd.read_csv(
        carpeta_resultados / "tablas_csv" / "plan_372_orientaciones.csv",
        encoding="utf-8-sig",
    )
    resumen, _ = consolidar_resultados(plan, carpeta_resultados)
    selecciones = seleccionar_resumenes(resumen, carpeta_resultados)
    manifiesto = generar_matrices_csv(plan, carpeta_resultados)
    pdfs = [
        generar_pdf_todas(
            carpeta_resultados / "TODAS_LAS_COMPARACIONES_EXCEL_JUAN.pdf",
            _tabla_pdf(resumen),
        ),
        generar_pdf_resumen(
            carpeta_resultados
            / "20_MEJORES_DECRECIENTES_Y_SUS_INVERSAS.pdf",
            selecciones["decrecientes"],
            manifiesto,
            "20 mejores decrecientes y sus inversas",
        ),
        generar_pdf_resumen(
            carpeta_resultados
            / "20_MEJORES_CRECIENTES_Y_SUS_INVERSAS.pdf",
            selecciones["crecientes"],
            manifiesto,
            "20 mejores crecientes y sus inversas",
        ),
        generar_pdf_resumen(
            carpeta_resultados / "20_MEJORES_COMPARACION_GLOBAL.pdf",
            selecciones["global"],
            manifiesto,
            "20 mejores parejas de la comparacion global",
        ),
    ]
    auditar_entregables(
        pdfs,
        resumen,
        selecciones,
        manifiesto,
        carpeta_resultados,
    )
    return pdfs


def auditar_entregables(
    pdfs: list[Path],
    resumen: pd.DataFrame,
    selecciones: dict[str, pd.DataFrame],
    manifiesto: pd.DataFrame,
    carpeta_resultados: Path,
) -> None:
    metricas_pdf = (
        "Score_mean",
        "Accuracy_mean",
        "Sensitivity_mean",
        "Specificity_mean",
        "Precision_mean",
    )

    def fila_presente(texto: str, fila: pd.Series) -> bool:
        valores = [
            str(fila["distribucion"]),
            *(f"{float(fila[columna]):.4f}" for columna in metricas_pdf),
        ]
        return all(valor in texto for valor in valores)

    comprobaciones: dict[str, bool] = {
        "resultados_372": len(resumen) == 372,
        "parejas_186": resumen["pareja"].nunique() == 186,
        "matrices_372": len(manifiesto) == 372,
        "matrices_1000_predicciones": manifiesto[
            "predicciones"
        ].eq(1000).all(),
        "cuatro_pdf": len(pdfs) == 4 and all(ruta.exists() for ruta in pdfs),
    }
    for nombre, tabla in selecciones.items():
        comprobaciones[f"{nombre}_40_filas"] = len(tabla) == 40
        comprobaciones[f"{nombre}_20_parejas"] = (
            tabla["pareja"].nunique() == 20
        )
    for indice, ruta in enumerate(pdfs):
        with fitz.open(ruta) as documento:
            texto = "\n".join(pagina.get_text() for pagina in documento)
            comprobaciones[f"pdf_{indice}_horizontal"] = all(
                pagina.rect.width > pagina.rect.height
                for pagina in documento
            )
            comprobaciones[f"pdf_{indice}_solo_ujanet_h"] = (
                "SVM" not in texto
                and "KNN" not in texto
                and "DeepONMF_W" not in texto
                and "binaria" not in texto.lower()
            )
            if indice == 0:
                comprobaciones["pdf_todas_cifras_iguales_csv"] = all(
                    fila_presente(
                        documento[posicion // 24].get_text(),
                        fila,
                    )
                    for posicion, (_, fila) in enumerate(
                        resumen.iterrows()
                    )
                )
                comprobaciones["pdf_todas_sin_matrices"] = (
                    sum(
                        len(pagina.get_images(full=True))
                        for pagina in documento
                    )
                    == 0
                )
                comprobaciones["pdf_todas_paginas"] = len(documento) == math.ceil(
                    372 / 24
                )
            else:
                comprobaciones[f"pdf_{indice}_22_paginas"] = (
                    len(documento) == 22
                )
                comprobaciones[f"pdf_{indice}_40_matrices"] = (
                    sum(
                        len(pagina.get_images(full=True))
                        for pagina in documento
                    )
                    == 40
                )
                nombre = ("decrecientes", "crecientes", "global")[
                    indice - 1
                ]
                tabla_seleccion = selecciones[nombre]
                comprobaciones[
                    f"pdf_{indice}_cifras_iguales_csv"
                ] = all(
                    fila_presente(
                        documento[posicion // 20].get_text(),
                        fila,
                    )
                    for posicion, (_, fila) in enumerate(
                        tabla_seleccion.iterrows()
                    )
                )
    comprobaciones["matrices_filas_200"] = True
    comprobaciones["matrices_porcentajes_100"] = True
    for fila in manifiesto.itertuples(index=False):
        conteos = pd.read_csv(
            Path(str(fila.csv_conteos)),
            index_col=0,
            encoding="utf-8-sig",
        )
        porcentajes = pd.read_csv(
            Path(str(fila.csv_porcentajes)),
            index_col=0,
            encoding="utf-8-sig",
        )
        if not conteos.sum(axis=1).eq(200).all():
            comprobaciones["matrices_filas_200"] = False
        if not np.allclose(
            porcentajes.sum(axis=1).to_numpy(dtype=float),
            100.0,
            atol=1e-9,
        ):
            comprobaciones["matrices_porcentajes_100"] = False
    tabla = pd.DataFrame(
        [
            {"comprobacion": nombre, "correcto": bool(valor)}
            for nombre, valor in comprobaciones.items()
        ]
    )
    tabla.to_csv(
        carpeta_resultados / "auditoria_final.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not all(comprobaciones.values()):
        raise AssertionError(f"Auditoria incorrecta: {comprobaciones}")


def ejecutar(
    carpeta_resultados: Path,
    carpeta_datos: Path,
    numero_workers: int,
    solo_informe: bool,
) -> list[Path]:
    inicio = time.perf_counter()
    preparar(carpeta_resultados, carpeta_datos)
    if not solo_informe:
        lanzar_workers(numero_workers, carpeta_resultados, carpeta_datos)
    pdfs = generar_entregables(carpeta_resultados)
    nombre_tiempo = (
        "resumen_tiempo_generacion_informes.txt"
        if solo_informe
        else "resumen_tiempo.txt"
    )
    (carpeta_resultados / nombre_tiempo).write_text(
        f"Tiempo total: {(time.perf_counter() - inicio) / 60:.2f} minutos\n",
        encoding="utf-8",
    )
    return pdfs
