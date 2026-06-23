from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import fitz
import pandas as pd


ANCHO = 841.89
ALTO = 595.28
MARGEN = 28
AZUL = (0.08, 0.22, 0.31)
AZUL_CLARO = (0.88, 0.93, 0.96)
VERDE = (0.86, 0.94, 0.87)
GRIS = (0.95, 0.96, 0.97)
BLANCO = (1.0, 1.0, 1.0)
NEGRO = (0.08, 0.08, 0.08)


def _texto(
    pagina: fitz.Page,
    rect: fitz.Rect,
    valor: object,
    tamano: float = 7.0,
    negrita: bool = False,
    color=NEGRO,
    alinear: int = 1,
) -> None:
    pagina.insert_textbox(
        rect,
        str(valor),
        fontname="hebo" if negrita else "helv",
        fontsize=tamano,
        color=color,
        align=alinear,
    )


def _pagina_titulo(
    documento: fitz.Document,
    titulo: str,
    lineas: list[str],
) -> None:
    pagina = documento.new_page(width=ANCHO, height=ALTO)
    pagina.draw_rect(
        fitz.Rect(0, 0, ANCHO, 76),
        color=AZUL,
        fill=AZUL,
    )
    pagina.insert_text(
        (MARGEN, 47),
        titulo,
        fontname="hebo",
        fontsize=22,
        color=BLANCO,
    )
    y = 115
    for linea in lineas:
        for fragmento in wrap(linea, 112):
            pagina.insert_text(
                (MARGEN, y),
                fragmento,
                fontname="helv",
                fontsize=11,
                color=NEGRO,
            )
            y += 18
        y += 8


def _tabla(
    documento: fitz.Document,
    titulo: str,
    tabla: pd.DataFrame,
    columnas: list[tuple[str, str, float]],
    filas_por_pagina: int = 22,
    resaltar: str | None = None,
) -> None:
    pesos = [columna[2] for columna in columnas]
    ancho_util = ANCHO - 2 * MARGEN
    anchos = [ancho_util * peso / sum(pesos) for peso in pesos]
    for inicio in range(0, len(tabla), filas_por_pagina):
        bloque = tabla.iloc[inicio : inicio + filas_por_pagina]
        pagina = documento.new_page(width=ANCHO, height=ALTO)
        encabezado = titulo if inicio == 0 else f"{titulo} (continuacion)"
        pagina.insert_text(
            (MARGEN, 27),
            encabezado,
            fontname="hebo",
            fontsize=12,
            color=AZUL,
        )
        y = 36
        alto_cabecera = 20
        alto_fila = (ALTO - y - MARGEN - alto_cabecera) / max(
            filas_por_pagina,
            len(bloque),
        )
        x = MARGEN
        for _, etiqueta, ancho in zip(
            [valor[0] for valor in columnas],
            [valor[1] for valor in columnas],
            anchos,
        ):
            rect = fitz.Rect(x, y, x + ancho, y + alto_cabecera)
            pagina.draw_rect(rect, color=AZUL, fill=AZUL, width=0.2)
            _texto(
                pagina,
                rect,
                etiqueta,
                tamano=6.5,
                negrita=True,
                color=BLANCO,
            )
            x += ancho
        y += alto_cabecera
        for numero, (_, fila) in enumerate(bloque.iterrows(), start=inicio):
            fondo = BLANCO if numero % 2 == 0 else GRIS
            if resaltar and bool(fila.get(resaltar, False)):
                fondo = VERDE
            x = MARGEN
            for (columna, _, _), ancho in zip(columnas, anchos):
                rect = fitz.Rect(x, y, x + ancho, y + alto_fila)
                pagina.draw_rect(
                    rect,
                    color=(0.72, 0.75, 0.78),
                    fill=fondo,
                    width=0.2,
                )
                valor = fila[columna]
                if isinstance(valor, float):
                    valor = f"{valor:.4f}"
                _texto(pagina, rect, valor, tamano=6.2)
                x += ancho
            y += alto_fila


def _pagina_matrices(
    documento: fitz.Document,
    pareja: int,
    clasificador: str,
    decreciente: pd.Series,
    creciente: pd.Series,
) -> None:
    pagina = documento.new_page(width=ANCHO, height=ALTO)
    pagina.insert_text(
        (MARGEN, 27),
        (
            f"Pareja {pareja}: {decreciente['distribucion']} frente a "
            f"{creciente['distribucion']} - {clasificador}"
        ),
        fontname="hebo",
        fontsize=12,
        color=AZUL,
    )
    rect_izq = fitz.Rect(MARGEN, 42, ANCHO / 2 - 8, 485)
    rect_der = fitz.Rect(ANCHO / 2 + 8, 42, ANCHO - MARGEN, 485)
    pagina.insert_image(rect_izq, filename=str(decreciente["ruta_png"]))
    pagina.insert_image(rect_der, filename=str(creciente["ruta_png"]))
    texto_izq = (
        f"Mayor error decreciente: {decreciente['mayor_error_real']} -> "
        f"{decreciente['mayor_error_predicha']} "
        f"({int(decreciente['mayor_error_cantidad'])} señales)."
    )
    texto_der = (
        f"Mayor error creciente: {creciente['mayor_error_real']} -> "
        f"{creciente['mayor_error_predicha']} "
        f"({int(creciente['mayor_error_cantidad'])} señales)."
    )
    _texto(
        pagina,
        fitz.Rect(MARGEN, 500, ANCHO / 2 - 8, 550),
        texto_izq,
        tamano=8.3,
        alinear=0,
    )
    _texto(
        pagina,
        fitz.Rect(ANCHO / 2 + 8, 500, ANCHO - MARGEN, 550),
        texto_der,
        tamano=8.3,
        alinear=0,
    )


def _dibujar_tabla_en_pagina(
    pagina: fitz.Page,
    tabla: pd.DataFrame,
    columnas: list[tuple[str, str, float]],
    y_inicial: float,
    y_final: float,
) -> None:
    pesos = [columna[2] for columna in columnas]
    ancho_util = ANCHO - 2 * MARGEN
    anchos = [ancho_util * peso / sum(pesos) for peso in pesos]
    alto_cabecera = 20
    alto_fila = (y_final - y_inicial - alto_cabecera) / len(tabla)
    x = MARGEN
    for (_, etiqueta, _), ancho in zip(columnas, anchos):
        rect = fitz.Rect(x, y_inicial, x + ancho, y_inicial + alto_cabecera)
        pagina.draw_rect(rect, color=AZUL, fill=AZUL, width=0.2)
        _texto(
            pagina,
            rect,
            etiqueta,
            tamano=6.5,
            negrita=True,
            color=BLANCO,
        )
        x += ancho
    y = y_inicial + alto_cabecera
    for numero, (_, fila) in enumerate(tabla.iterrows()):
        fondo = BLANCO if numero % 2 == 0 else GRIS
        x = MARGEN
        for (columna, _, _), ancho in zip(columnas, anchos):
            rect = fitz.Rect(x, y, x + ancho, y + alto_fila)
            pagina.draw_rect(
                rect,
                color=(0.72, 0.75, 0.78),
                fill=fondo,
                width=0.2,
            )
            valor = fila[columna]
            if isinstance(valor, float):
                valor = f"{valor:.4f}"
            _texto(pagina, rect, valor, tamano=6.2)
            x += ancho
        y += alto_fila


def _pagina_resumen_ujanet(
    documento: fitz.Document,
    tabla: pd.DataFrame,
) -> None:
    pagina = documento.new_page(width=ANCHO, height=ALTO)
    pagina.insert_text(
        (MARGEN, 26),
        "Comparacion multiclase UjaNet: arquitecturas decrecientes e invertidas",
        fontname="hebo",
        fontsize=13,
        color=AZUL,
    )
    pagina.insert_text(
        (MARGEN, 46),
        (
            "Cada pareja aparece de forma consecutiva: primero la arquitectura "
            "decreciente y despues su inversion creciente."
        ),
        fontname="helv",
        fontsize=8.5,
        color=NEGRO,
    )
    pagina.insert_text(
        (MARGEN, 61),
        (
            "Resultados de UjaNet sobre la activacion temporal H final, "
            "obtenidos mediante validacion cruzada de cinco folds."
        ),
        fontname="helv",
        fontsize=8.5,
        color=NEGRO,
    )
    _dibujar_tabla_en_pagina(
        pagina,
        tabla,
        [
            ("pareja", "Pareja", 0.48),
            ("sentido", "Sentido", 0.78),
            ("distribucion", "Distribucion", 0.88),
            ("representacion", "Matriz H final", 0.95),
            ("Score_mean", "Score", 0.68),
            ("Accuracy_mean", "Accuracy", 0.72),
            ("Sensitivity_mean", "Sensitivity", 0.78),
            ("Specificity_mean", "Specificity", 0.78),
            ("Precision_mean", "Precision", 0.72),
        ],
        y_inicial=74,
        y_final=ALTO - 18,
    )


def _pagina_pareja_ujanet(
    documento: fitz.Document,
    pareja: int,
    metricas: pd.DataFrame,
    matriz_decreciente: pd.Series,
    matriz_creciente: pd.Series,
) -> None:
    pagina = documento.new_page(width=ANCHO, height=ALTO)
    pagina.insert_text(
        (MARGEN, 27),
        (
            f"Pareja {pareja}: {matriz_decreciente['distribucion']} frente a "
            f"{matriz_creciente['distribucion']} - UjaNet"
        ),
        fontname="hebo",
        fontsize=12,
        color=AZUL,
    )
    _dibujar_tabla_en_pagina(
        pagina,
        metricas,
        [
            ("sentido", "Sentido", 0.78),
            ("distribucion", "Distribucion", 0.9),
            ("representacion", "Matriz H final", 0.95),
            ("Score_mean", "Score", 0.7),
            ("Accuracy_mean", "Accuracy", 0.75),
            ("Sensitivity_mean", "Sensitivity", 0.8),
            ("Specificity_mean", "Specificity", 0.8),
            ("Precision_mean", "Precision", 0.75),
        ],
        y_inicial=38,
        y_final=102,
    )
    rect_izq = fitz.Rect(MARGEN, 112, ANCHO / 2 - 8, 505)
    rect_der = fitz.Rect(ANCHO / 2 + 8, 112, ANCHO - MARGEN, 505)
    pagina.insert_image(
        rect_izq,
        filename=str(matriz_decreciente["ruta_png"]),
        keep_proportion=True,
    )
    pagina.insert_image(
        rect_der,
        filename=str(matriz_creciente["ruta_png"]),
        keep_proportion=True,
    )
    texto_izq = (
        "Decreciente. Mayor error: "
        f"{matriz_decreciente['mayor_error_real']} -> "
        f"{matriz_decreciente['mayor_error_predicha']} "
        f"({int(matriz_decreciente['mayor_error_cantidad'])} senales)."
    )
    texto_der = (
        "Creciente. Mayor error: "
        f"{matriz_creciente['mayor_error_real']} -> "
        f"{matriz_creciente['mayor_error_predicha']} "
        f"({int(matriz_creciente['mayor_error_cantidad'])} senales)."
    )
    _texto(
        pagina,
        fitz.Rect(MARGEN, 515, ANCHO / 2 - 8, 552),
        texto_izq,
        tamano=8.0,
        alinear=0,
    )
    _texto(
        pagina,
        fitz.Rect(ANCHO / 2 + 8, 515, ANCHO - MARGEN, 552),
        texto_der,
        tamano=8.0,
        alinear=0,
    )


def generar_pdf_completo(
    ruta: Path,
    ranking: pd.DataFrame,
    detalle: pd.DataFrame,
    resumen_profundidad: pd.DataFrame,
    seleccion: pd.DataFrame,
) -> Path:
    ranking_vista = ranking.copy()
    ranking_vista["seleccion_texto"] = ranking_vista["seleccionada"].map(
        {True: "Si", False: "No"}
    )
    seleccion_vista = seleccion.copy()
    seleccion_vista["criterio"] = seleccion_vista[
        "forzada_como_referencia"
    ].map({True: "Referencia 9-8-7", False: "Ranking"})
    documento = fitz.open()
    _pagina_titulo(
        documento,
        "Resultados completos multiclase - ultima prueba de Juan",
        [
            (
                f"Se presentan {ranking['distribucion'].nunique()} "
                "arquitecturas Deep-ONMF evaluadas exclusivamente sobre las "
                "cinco clases N, AS, MR, MS y MVP."
            ),
            (
                "Cada fila del ranking es un resultado real de un clasificador. "
                "No se promedian SVM, KNN y UjaNet. Se ordena por Accuracy y "
                "despues por Score."
            ),
            (
                "Los resultados anteriores se reutilizan sin modificar. Las "
                "nuevas pruebas completan 50 configuraciones de cuatro capas y "
                "50 configuraciones de cinco capas."
            ),
            (
                "Accuracy, Sensitivity, Specificity, Precision y Score "
                "conservan exactamente la definicion macro uno-contra-rest "
                "utilizada en las tablas multiclase anteriores."
            ),
            (
                "La seleccion final contiene las nueve primeras del ranking y "
                "el modelo 9-8-7 como referencia obligatoria cuando no aparece "
                "entre ellas. Su posicion global se conserva de forma visible."
            ),
        ],
    )
    _tabla(
        documento,
        "Resumen por profundidad",
        resumen_profundidad,
        [
            ("numero_capas", "Capas", 0.7),
            ("configuraciones", "Configuraciones", 1.2),
            ("accuracy_maxima", "Accuracy H maxima", 1.2),
            ("score_maximo", "Score H maximo", 1.1),
        ],
        filas_por_pagina=15,
    )
    _tabla(
        documento,
        "Ranking completo de resultados H por clasificador",
        ranking_vista,
        [
            ("posicion_resultado", "Pos.", 0.45),
            ("distribucion", "Distribucion", 1.0),
            ("numero_capas", "Capas", 0.5),
            ("clasificador", "Clasificador", 0.75),
            ("representacion", "Representacion", 0.95),
            ("Score", "Score", 0.7),
            ("Accuracy", "Accuracy", 0.75),
            ("Sensitivity", "Sensitivity", 0.75),
            ("Specificity", "Specificity", 0.75),
            ("Precision", "Precision", 0.7),
            ("origen", "Origen", 0.9),
            ("seleccion_texto", "Seleccionada", 0.7),
        ],
        resaltar="seleccionada",
    )
    _tabla(
        documento,
        "Resultados completos: W y H final",
        detalle,
        [
            ("clasificador", "Clasificador", 0.75),
            ("representacion", "Representacion", 1.15),
            ("distribucion", "Distribucion", 0.85),
            ("numero_capas", "Capas", 0.42),
            ("Score_mean", "Score", 0.65),
            ("Accuracy_mean", "Accuracy", 0.7),
            ("Sensitivity_mean", "Sensitivity", 0.72),
            ("Specificity_mean", "Specificity", 0.72),
            ("Precision_mean", "Precision", 0.7),
            ("origen", "Origen", 0.75),
        ],
    )
    _tabla(
        documento,
        "Diez configuraciones principales seleccionadas",
        seleccion_vista,
        [
            ("posicion_seleccion", "Sel.", 0.45),
            ("posicion_resultado", "Pos. resultado", 0.7),
            ("posicion_arquitectura", "Pos. arquitectura", 0.8),
            ("distribucion", "Decreciente", 1.0),
            ("distribucion_invertida", "Creciente", 1.0),
            ("numero_capas", "Capas", 0.5),
            ("clasificador", "Clasificador", 0.7),
            ("Accuracy", "Accuracy", 0.8),
            ("Score", "Score", 0.75),
            ("origen", "Origen", 0.8),
            ("criterio", "Criterio", 1.0),
        ],
        filas_por_pagina=15,
    )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.stem + "_temporal.pdf")
    documento.save(temporal, garbage=4, deflate=True)
    documento.close()
    temporal.replace(ruta)
    return ruta


def generar_pdf_comparacion(
    ruta: Path,
    seleccion: pd.DataFrame,
    detalle: pd.DataFrame,
    diferencias: pd.DataFrame,
    matrices: pd.DataFrame,
    conclusiones: list[str],
) -> Path:
    del diferencias, conclusiones
    if len(seleccion) != 10:
        raise AssertionError(f"Se esperaban diez parejas, hay {len(seleccion)}")
    detalle_ujanet_h = detalle[
        detalle["clasificador"].eq("UjaNet")
        & detalle["tipo_matriz"].eq("H")
    ].copy()
    orden_sentido = pd.CategoricalDtype(
        ["decreciente", "creciente"],
        ordered=True,
    )
    detalle_ujanet_h["sentido"] = detalle_ujanet_h["sentido"].astype(
        orden_sentido
    )
    detalle_ujanet_h = detalle_ujanet_h.sort_values(
        ["pareja", "sentido"],
        kind="mergesort",
    ).reset_index(drop=True)
    matrices_ujanet = matrices[
        matrices["clasificador"].eq("UjaNet")
    ].copy()
    matrices_ujanet["sentido"] = matrices_ujanet["sentido"].astype(
        orden_sentido
    )
    matrices_ujanet = matrices_ujanet.sort_values(
        ["pareja", "sentido"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(detalle_ujanet_h) != 20:
        raise AssertionError(
            f"Se esperaban 20 filas UjaNet-H, hay {len(detalle_ujanet_h)}"
        )
    if len(matrices_ujanet) != 20:
        raise AssertionError(
            f"Se esperaban 20 matrices UjaNet-H, hay {len(matrices_ujanet)}"
        )
    documento = fitz.open()
    _pagina_resumen_ujanet(documento, detalle_ujanet_h)
    for pareja in range(1, 11):
        metricas = detalle_ujanet_h[
            detalle_ujanet_h["pareja"].eq(pareja)
        ].copy()
        bloque = matrices_ujanet[matrices_ujanet["pareja"].eq(pareja)]
        matriz_decreciente = bloque[
            bloque["sentido"].eq("decreciente")
        ].iloc[0]
        matriz_creciente = bloque[
            bloque["sentido"].eq("creciente")
        ].iloc[0]
        _pagina_pareja_ujanet(
            documento,
            pareja,
            metricas,
            matriz_decreciente,
            matriz_creciente,
        )
    if len(documento) != 11:
        raise AssertionError(f"El PDF debe tener 11 paginas, tiene {len(documento)}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.stem + "_temporal.pdf")
    documento.save(temporal, garbage=4, deflate=True)
    documento.close()
    temporal.replace(ruta)
    return ruta
