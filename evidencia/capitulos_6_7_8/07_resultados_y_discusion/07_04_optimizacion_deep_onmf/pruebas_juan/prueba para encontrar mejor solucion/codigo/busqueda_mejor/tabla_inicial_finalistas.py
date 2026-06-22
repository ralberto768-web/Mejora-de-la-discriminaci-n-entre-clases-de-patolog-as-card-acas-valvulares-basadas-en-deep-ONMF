from __future__ import annotations

from pathlib import Path

import fitz
import pandas as pd

from .informe import (
    ALTO,
    ANCHO,
    AZUL,
    BLANCO,
    GRIS,
    MARGEN_X,
    MARGEN_Y,
    NEGRO,
    _texto_celda,
)


COLUMNAS = [
    "clasificador",
    "representacion",
    "distribucion",
    "Score_mean",
    "Accuracy_mean",
    "Sensitivity_mean",
    "Specificity_mean",
    "Precision_mean",
]

ETIQUETAS = [
    "Clasificador",
    "Representacion",
    "Distribucion",
    "Score",
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Precision",
]

PESOS = [1.05, 1.65, 1.05, 0.86, 0.94, 1.04, 1.04, 0.94]


def _normalizar_finalistas(tabla: pd.DataFrame) -> pd.DataFrame:
    return tabla.rename(
        columns={
            "Score_mean": "Score_mean",
            "Accuracy_mean": "Accuracy_mean",
            "Sensitivity_mean": "Sensitivity_mean",
            "Specificity_mean": "Specificity_mean",
            "Precision_mean": "Precision_mean",
        }
    )[COLUMNAS].copy()


def construir_tablas_66(
    carpeta_tablas_antiguas: Path,
    carpeta_tablas_busqueda: Path,
) -> dict[str, pd.DataFrame]:
    finalistas = pd.read_csv(
        carpeta_tablas_busqueda / "tabla_compacta_busqueda.csv",
        encoding="utf-8-sig",
    )
    cinco = set(finalistas.head(5)["distribucion"].astype(str))
    tablas: dict[str, pd.DataFrame] = {}

    for tipo in ("binaria", "multiclase"):
        antigua = pd.read_csv(
            carpeta_tablas_antiguas / f"tabla_{tipo}_ampliada.csv",
            encoding="utf-8-sig",
        )[COLUMNAS]
        resumen = pd.read_csv(
            carpeta_tablas_busqueda
            / (
                "todas_resumen_metricas_binarias.csv"
                if tipo == "binaria"
                else "todas_resumen_metricas_multiclase.csv"
            ),
            encoding="utf-8-sig",
        )
        nuevas = resumen[
            resumen["distribucion"].astype(str).isin(cinco)
        ].copy()
        nuevas = _normalizar_finalistas(nuevas)
        orden_finalistas = {
            distribucion: indice
            for indice, distribucion in enumerate(
                finalistas.head(5)["distribucion"].astype(str)
            )
        }
        orden_clasificadores = {"SVM": 0, "KNN": 1, "UjaNet": 2}
        nuevas["_orden_finalista"] = nuevas["distribucion"].map(
            orden_finalistas
        )
        nuevas["_orden_clasificador"] = nuevas["clasificador"].map(
            orden_clasificadores
        )
        nuevas["_orden_representacion"] = nuevas["representacion"].map(
            lambda valor: 0 if str(valor) == "DeepONMF_W" else 1
        )
        nuevas = nuevas.sort_values(
            [
                "_orden_finalista",
                "_orden_clasificador",
                "_orden_representacion",
            ],
            kind="mergesort",
        ).drop(
            columns=[
                "_orden_finalista",
                "_orden_clasificador",
                "_orden_representacion",
            ]
        )
        combinada = pd.concat([antigua, nuevas], ignore_index=True)
        if len(antigua) != 36 or len(nuevas) != 30 or len(combinada) != 66:
            raise AssertionError(
                f"Tabla {tipo}: se esperaban 36 + 30 = 66 filas"
            )
        combinada = combinada.sort_values(
            ["Accuracy_mean", "Score_mean"],
            ascending=[False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        combinada.attrs["distribuciones_finalistas"] = cinco
        filas_antiguas_recuperadas = combinada.merge(
            antigua.drop_duplicates(),
            on=COLUMNAS,
            how="inner",
        )
        if len(filas_antiguas_recuperadas) != len(antigua):
            raise AssertionError(
                f"Tabla {tipo}: no se conservaron las 36 filas iniciales"
            )
        tablas[tipo] = combinada
        combinada.to_csv(
            carpeta_tablas_busqueda
            / f"tabla_{tipo}_inicial_mas_5_mejores_66_filas.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return tablas


def _formatear(valor: object, columna: str) -> str:
    if columna.endswith("_mean"):
        return f"{float(valor):.4f}"
    return str(valor)


def _dibujar_tabla_66(
    documento: fitz.Document,
    tabla: pd.DataFrame,
    tipo: str,
) -> None:
    distribuciones_finalistas = set(
        tabla.attrs.get("distribuciones_finalistas", ())
    )
    filas_por_pagina = 22
    indice = 0
    ancho_util = ANCHO - 2 * MARGEN_X
    anchos = [ancho_util * peso / sum(PESOS) for peso in PESOS]
    while indice < len(tabla):
        pagina = documento.new_page(width=ANCHO, height=ALTO)
        titulo = (
            f"Tabla {tipo}: representaciones iniciales y cinco mejores"
            if indice == 0
            else f"Tabla {tipo} (continuacion)"
        )
        pagina.insert_text(
            (MARGEN_X, MARGEN_Y + 14),
            titulo,
            fontname="hebo",
            fontsize=13,
            color=AZUL,
        )
        y = MARGEN_Y + 22
        alto_cabecera = 19.0
        alto_fila = (ALTO - MARGEN_Y - y - alto_cabecera) / min(
            filas_por_pagina,
            len(tabla) - indice,
        )

        x = MARGEN_X
        for etiqueta, ancho in zip(ETIQUETAS, anchos):
            rect = fitz.Rect(x, y, x + ancho, y + alto_cabecera)
            pagina.draw_rect(rect, color=AZUL, fill=AZUL, width=0.3)
            _texto_celda(
                pagina,
                rect,
                etiqueta,
                tamano=6.8,
                negrita=True,
                color=BLANCO,
            )
            x += ancho
        y += alto_cabecera

        fin = min(indice + filas_por_pagina, len(tabla))
        for numero in range(indice, fin):
            fila = tabla.iloc[numero]
            fondo = BLANCO if numero % 2 == 0 else GRIS
            if str(fila["distribucion"]) in distribuciones_finalistas:
                fondo = (0.88, 0.95, 0.89)
            x = MARGEN_X
            for columna, ancho in zip(COLUMNAS, anchos):
                rect = fitz.Rect(x, y, x + ancho, y + alto_fila)
                pagina.draw_rect(
                    rect,
                    color=(0.68, 0.72, 0.75),
                    fill=fondo,
                    width=0.25,
                )
                _texto_celda(
                    pagina,
                    rect,
                    _formatear(fila[columna], columna),
                    tamano=6.8,
                    color=NEGRO,
                )
                x += ancho
            y += alto_fila
        indice = fin


def generar_pdf_tabla_inicial_mas_cinco(
    ruta_salida: Path,
    carpeta_tablas_antiguas: Path,
    carpeta_tablas_busqueda: Path,
) -> Path:
    tablas = construir_tablas_66(
        carpeta_tablas_antiguas,
        carpeta_tablas_busqueda,
    )
    documento = fitz.open()
    _dibujar_tabla_66(documento, tablas["binaria"], "binaria")
    _dibujar_tabla_66(documento, tablas["multiclase"], "multiclase")
    documento.set_metadata(
        {
            "title": "Tablas iniciales y cinco mejores configuraciones",
            "author": "TFG de Alberto",
            "subject": "Comparacion binaria y multiclase de 66 filas",
        }
    )
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta_salida.with_name(ruta_salida.stem + "_temporal.pdf")
    documento.save(temporal, garbage=4, deflate=True)
    documento.close()
    temporal.replace(ruta_salida)
    auditar_pdf_tablas_66(ruta_salida, tablas)
    return ruta_salida


def auditar_pdf_tablas_66(
    ruta_pdf: Path,
    tablas: dict[str, pd.DataFrame],
) -> None:
    documento = fitz.open(ruta_pdf)
    texto = "\n".join(pagina.get_text() for pagina in documento)
    if not all(pagina.rect.width > pagina.rect.height for pagina in documento):
        raise AssertionError("El PDF de tablas no es completamente horizontal")
    if "+/-" in texto or "std" in texto.lower():
        raise AssertionError("El PDF contiene desviaciones estandar")
    for tipo, tabla in tablas.items():
        if len(tabla) != 66:
            raise AssertionError(f"La tabla {tipo} no contiene 66 filas")
        for _, fila in tabla.iterrows():
            firma = "\n".join(
                _formatear(fila[columna], columna) for columna in COLUMNAS
            )
            if firma not in texto:
                raise AssertionError(
                    f"Falta una fila {tipo} en el PDF: "
                    f"{fila['clasificador']} {fila['representacion']} "
                    f"{fila['distribucion']}"
                )
    documento.close()
