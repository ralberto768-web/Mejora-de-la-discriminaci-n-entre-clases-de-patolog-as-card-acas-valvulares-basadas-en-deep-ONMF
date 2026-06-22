from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from codigo.configuracion import CLASES

from .arquitecturas import clave, convertir_etiqueta


METRICAS = ("Accuracy", "Score", "Sensitivity", "Specificity", "Precision")


def carpeta_configuracion(
    carpeta_resultados: Path,
    sentido: str,
    distribucion: str,
) -> Path:
    return (
        carpeta_resultados
        / "configuraciones"
        / sentido
        / clave(convertir_etiqueta(distribucion))
    )


def consolidar_resultados(
    plan: pd.DataFrame,
    carpeta_resultados: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filas_resumen = []
    filas_fold = []
    faltantes = []
    for fila in plan.itertuples(index=False):
        carpeta = carpeta_configuracion(
            carpeta_resultados,
            str(fila.sentido),
            str(fila.distribucion),
        )
        ruta_resumen = (
            carpeta / "metricas" / "resumen_metricas_multiclase.csv"
        )
        ruta_fold = (
            carpeta / "metricas" / "metricas_multiclase_por_fold.csv"
        )
        if not ruta_resumen.exists() or not ruta_fold.exists():
            faltantes.append(f"{fila.sentido}:{fila.distribucion}")
            continue
        resumen = pd.read_csv(ruta_resumen, encoding="utf-8-sig")
        por_fold = pd.read_csv(ruta_fold, encoding="utf-8-sig")
        if len(resumen) != 1 or len(por_fold) != 5:
            faltantes.append(f"{fila.sentido}:{fila.distribucion}")
            continue
        for tabla in (resumen, por_fold):
            tabla["pareja"] = int(fila.pareja)
            tabla["sentido"] = str(fila.sentido)
            tabla["orden_excel"] = int(fila.orden_excel)
            tabla["dimensiones_repetidas"] = bool(
                fila.dimensiones_repetidas
            )
        filas_resumen.append(resumen)
        filas_fold.append(por_fold)
    if faltantes:
        raise RuntimeError(
            f"Faltan {len(faltantes)} orientaciones completas. "
            f"Primeras: {faltantes[:10]}"
        )
    resumen = pd.concat(filas_resumen, ignore_index=True).sort_values(
        "orden_excel"
    )
    por_fold = pd.concat(filas_fold, ignore_index=True).sort_values(
        ["orden_excel", "fold"]
    )
    if len(resumen) != 372 or len(por_fold) != 1860:
        raise AssertionError("La consolidacion no contiene 372 x 5 resultados")
    tablas = carpeta_resultados / "tablas_csv"
    tablas.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(
        tablas / "resultados_372_orientaciones.csv",
        index=False,
        encoding="utf-8-sig",
    )
    por_fold.to_csv(
        tablas / "metricas_372_orientaciones_por_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return resumen.reset_index(drop=True), por_fold.reset_index(drop=True)


def ordenar(tabla: pd.DataFrame) -> pd.DataFrame:
    columnas = [f"{metrica}_mean" for metrica in METRICAS]
    ordenada = tabla.copy()
    columnas_orden = []
    for columna in columnas:
        columna_orden = f"_orden_{columna}"
        # Evita que residuos de coma flotante (por ejemplo, 0.98725 frente
        # a 0.9872499999999998) decidan un desempate que matematicamente
        # corresponde a la metrica siguiente.
        ordenada[columna_orden] = ordenada[columna].round(12)
        columnas_orden.append(columna_orden)
    ordenada = ordenada.sort_values(
        [*columnas_orden, "distribucion"],
        ascending=[False] * len(columnas) + [True],
        kind="mergesort",
    ).drop(columns=columnas_orden)
    return ordenada.reset_index(drop=True)


def _parejas_presentacion(
    seleccion: pd.DataFrame,
    resumen: pd.DataFrame,
    tipo_seleccion: str,
) -> pd.DataFrame:
    filas = []
    for posicion, candidata in enumerate(
        seleccion.itertuples(index=False),
        start=1,
    ):
        pareja = int(candidata.pareja)
        bloque = resumen[resumen["pareja"].eq(pareja)].copy()
        bloque["posicion"] = posicion
        bloque["sentido_seleccionado"] = str(candidata.sentido)
        bloque["seleccionada"] = bloque["sentido"].eq(
            str(candidata.sentido)
        )
        bloque["tipo_seleccion"] = tipo_seleccion
        orden = pd.Categorical(
            bloque["sentido"],
            categories=["decreciente", "creciente"],
            ordered=True,
        )
        bloque = bloque.assign(_orden=orden).sort_values("_orden").drop(
            columns="_orden"
        )
        filas.append(bloque)
    resultado = pd.concat(filas, ignore_index=True)
    if len(resultado) != 40:
        raise AssertionError("Cada resumen debe contener 40 filas")
    return resultado


def seleccionar_resumenes(
    resumen: pd.DataFrame,
    carpeta_resultados: Path,
) -> dict[str, pd.DataFrame]:
    decrecientes = ordenar(
        resumen[resumen["sentido"].eq("decreciente")]
    ).head(20)
    crecientes = ordenar(
        resumen[resumen["sentido"].eq("creciente")]
    ).head(20)
    candidatos_globales = []
    for _, bloque in resumen.groupby("pareja", sort=False):
        candidatos_globales.append(ordenar(bloque).iloc[0])
    globales = ordenar(pd.DataFrame(candidatos_globales)).head(20)
    selecciones = {
        "decrecientes": _parejas_presentacion(
            decrecientes,
            resumen,
            "mejor_decreciente",
        ),
        "crecientes": _parejas_presentacion(
            crecientes,
            resumen,
            "mejor_creciente",
        ),
        "global": _parejas_presentacion(
            globales,
            resumen,
            "mejor_global_de_la_pareja",
        ),
    }
    carpeta = carpeta_resultados / "tablas_csv"
    for nombre, tabla in selecciones.items():
        tabla.to_csv(
            carpeta / f"seleccion_20_{nombre}_40_filas.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return selecciones


def leer_predicciones(
    carpeta_resultados: Path,
    sentido: str,
    distribucion: str,
) -> pd.DataFrame:
    numero_capas = len(convertir_etiqueta(distribucion))
    carpeta = (
        carpeta_configuracion(carpeta_resultados, sentido, distribucion)
        / "pred"
        / "UjaNet"
        / f"H{numero_capas}"
    )
    archivos = [carpeta / f"f{fold}_pred.csv" for fold in range(1, 6)]
    if not all(ruta.exists() for ruta in archivos):
        raise FileNotFoundError(f"Faltan predicciones en {carpeta}")
    tabla = pd.concat(
        [pd.read_csv(ruta, encoding="utf-8-sig") for ruta in archivos],
        ignore_index=True,
    )
    if len(tabla) != 1000:
        raise AssertionError(f"{distribucion}: no contiene 1000 predicciones")
    return tabla


def matriz_confusion(
    predicciones: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conteos = pd.crosstab(
        pd.Categorical(predicciones["clase"], categories=CLASES),
        pd.Categorical(
            predicciones["pred_multiclase"],
            categories=CLASES,
        ),
        dropna=False,
    )
    conteos.index = list(CLASES)
    conteos.columns = list(CLASES)
    porcentajes = conteos.div(conteos.sum(axis=1), axis=0) * 100.0
    return conteos, porcentajes


def generar_matrices_csv(
    plan: pd.DataFrame,
    carpeta_resultados: Path,
) -> pd.DataFrame:
    filas = []
    raiz = carpeta_resultados / "matrices_confusion"
    for fila in plan.itertuples(index=False):
        predicciones = leer_predicciones(
            carpeta_resultados,
            str(fila.sentido),
            str(fila.distribucion),
        )
        conteos, porcentajes = matriz_confusion(predicciones)
        carpeta = (
            raiz
            / f"pareja_{int(fila.pareja):03d}"
            / str(fila.sentido)
        )
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta_conteos = carpeta / "conteos.csv"
        ruta_porcentajes = carpeta / "porcentajes.csv"
        conteos.to_csv(ruta_conteos, encoding="utf-8-sig")
        porcentajes.to_csv(ruta_porcentajes, encoding="utf-8-sig")
        errores = conteos.to_numpy(copy=True)
        np.fill_diagonal(errores, 0)
        indice = np.unravel_index(int(np.argmax(errores)), errores.shape)
        filas.append(
            {
                "pareja": int(fila.pareja),
                "sentido": str(fila.sentido),
                "distribucion": str(fila.distribucion),
                "predicciones": len(predicciones),
                "aciertos": int(np.trace(conteos.to_numpy())),
                "mayor_error_real": CLASES[indice[0]],
                "mayor_error_predicha": CLASES[indice[1]],
                "mayor_error_cantidad": int(errores[indice]),
                "csv_conteos": str(ruta_conteos.resolve()),
                "csv_porcentajes": str(ruta_porcentajes.resolve()),
            }
        )
    manifiesto = pd.DataFrame(filas).sort_values(["pareja", "sentido"])
    if len(manifiesto) != 372 or not manifiesto["predicciones"].eq(1000).all():
        raise AssertionError("Las 372 matrices no son completas")
    manifiesto.to_csv(
        raiz / "manifiesto_372_matrices.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return manifiesto.reset_index(drop=True)
