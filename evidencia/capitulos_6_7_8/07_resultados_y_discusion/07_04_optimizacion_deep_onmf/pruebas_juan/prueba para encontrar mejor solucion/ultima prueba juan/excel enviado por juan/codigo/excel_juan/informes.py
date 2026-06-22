from __future__ import annotations

from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANCHO = 841.89
ALTO = 595.28
MARGEN = 26
AZUL = (0.08, 0.22, 0.31)
BLANCO = (1.0, 1.0, 1.0)
GRIS = (0.95, 0.96, 0.97)
VERDE = (0.86, 0.94, 0.87)
NEGRO = (0.08, 0.08, 0.08)
CLASES = ("N", "AS", "MR", "MS", "MVP")


def _texto(
    pagina: fitz.Page,
    rect: fitz.Rect,
    valor: object,
    tamano: float = 6.2,
    negrita: bool = False,
    color=NEGRO,
) -> None:
    pagina.insert_textbox(
        rect,
        str(valor),
        fontname="hebo" if negrita else "helv",
        fontsize=tamano,
        color=color,
        align=1,
    )


def _tabla_pagina(
    documento: fitz.Document,
    titulo: str,
    tabla: pd.DataFrame,
    primera: bool,
    resaltar: bool,
) -> None:
    pagina = documento.new_page(width=ANCHO, height=ALTO)
    pagina.insert_text(
        (MARGEN, 25),
        titulo,
        fontname="hebo",
        fontsize=12 if primera else 10.5,
        color=AZUL,
    )
    columnas = [
        ("posicion", "Pos.", 0.42),
        ("pareja", "Pareja", 0.48),
        ("sentido", "Sentido", 0.72),
        ("distribucion", "Distribucion", 0.88),
        ("numero_capas", "Capas", 0.46),
        ("representacion", "H final", 0.88),
        ("Score_mean", "Score", 0.65),
        ("Accuracy_mean", "Accuracy", 0.7),
        ("Sensitivity_mean", "Sensitivity", 0.75),
        ("Specificity_mean", "Specificity", 0.75),
        ("Precision_mean", "Precision", 0.7),
    ]
    if "posicion" not in tabla:
        tabla = tabla.copy()
        tabla["posicion"] = tabla["pareja"]
    pesos = [columna[2] for columna in columnas]
    anchos = [
        (ANCHO - 2 * MARGEN) * peso / sum(pesos) for peso in pesos
    ]
    y = 35
    alto_cabecera = 19
    alto_fila = (ALTO - y - MARGEN - alto_cabecera) / len(tabla)
    x = MARGEN
    for (_, etiqueta, _), ancho in zip(columnas, anchos):
        rect = fitz.Rect(x, y, x + ancho, y + alto_cabecera)
        pagina.draw_rect(rect, color=AZUL, fill=AZUL, width=0.2)
        _texto(
            pagina,
            rect,
            etiqueta,
            tamano=6.1,
            negrita=True,
            color=BLANCO,
        )
        x += ancho
    y += alto_cabecera
    for indice, (_, fila) in enumerate(tabla.iterrows()):
        fondo = BLANCO if indice % 2 == 0 else GRIS
        if resaltar and bool(fila.get("seleccionada", False)):
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
            _texto(pagina, rect, valor, tamano=5.8)
            x += ancho
        y += alto_fila


def generar_pdf_todas(
    ruta: Path,
    tabla: pd.DataFrame,
) -> Path:
    documento = fitz.open()
    filas_por_pagina = 24
    for inicio in range(0, len(tabla), filas_por_pagina):
        bloque = tabla.iloc[inicio : inicio + filas_por_pagina].copy()
        _tabla_pagina(
            documento,
            (
                "Todas las comparaciones del Excel enviado por Juan"
                if inicio == 0
                else "Todas las comparaciones (continuacion)"
            ),
            bloque,
            primera=inicio == 0,
            resaltar=False,
        )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.stem + "_temporal.pdf")
    documento.save(temporal, garbage=4, deflate=True)
    documento.close()
    temporal.replace(ruta)
    return ruta


def _png_matriz(
    conteos: pd.DataFrame,
    porcentajes: pd.DataFrame,
    titulo: str,
    ruta: Path,
) -> Path:
    if ruta.exists():
        return ruta
    figura, eje = plt.subplots(figsize=(6.6, 5.2))
    imagen = eje.imshow(porcentajes.to_numpy(), cmap="Blues", vmin=0, vmax=100)
    eje.set_xticks(range(5), CLASES)
    eje.set_yticks(range(5), CLASES)
    eje.set_xlabel("Clase predicha")
    eje.set_ylabel("Clase real")
    eje.set_title(titulo, fontsize=10.5, fontweight="bold")
    for fila in range(5):
        for columna in range(5):
            porcentaje = float(porcentajes.iloc[fila, columna])
            eje.text(
                columna,
                fila,
                f"{int(conteos.iloc[fila, columna])}\n{porcentaje:.1f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if porcentaje >= 55 else "black",
            )
    barra = figura.colorbar(imagen, ax=eje, fraction=0.046, pad=0.04)
    barra.set_label("% dentro de la clase real")
    figura.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=170, bbox_inches="tight")
    plt.close(figura)
    return ruta


def _pagina_matrices(
    documento: fitz.Document,
    posicion: int,
    bloque: pd.DataFrame,
    manifiesto: pd.DataFrame,
) -> None:
    decreciente = bloque[bloque["sentido"].eq("decreciente")].iloc[0]
    creciente = bloque[bloque["sentido"].eq("creciente")].iloc[0]
    matrices = manifiesto[manifiesto["pareja"].eq(int(decreciente["pareja"]))]
    datos_dec = matrices[matrices["sentido"].eq("decreciente")].iloc[0]
    datos_inc = matrices[matrices["sentido"].eq("creciente")].iloc[0]
    rutas_png = []
    for fila, datos in ((decreciente, datos_dec), (creciente, datos_inc)):
        conteos = pd.read_csv(datos["csv_conteos"], index_col=0)
        porcentajes = pd.read_csv(datos["csv_porcentajes"], index_col=0)
        ruta_png = (
            Path(datos["csv_conteos"]).parent / "matriz_confusion.png"
        )
        rutas_png.append(
            _png_matriz(
                conteos,
                porcentajes,
                (
                    f"{fila['distribucion']} - "
                    f"{fila['representacion']} - UjaNet"
                ),
                ruta_png,
            )
        )
    pagina = documento.new_page(width=ANCHO, height=ALTO)
    pagina.insert_text(
        (MARGEN, 25),
        (
            f"Pareja {posicion}: {decreciente['distribucion']} frente a "
            f"{creciente['distribucion']}"
        ),
        fontname="hebo",
        fontsize=12,
        color=AZUL,
    )
    tabla = bloque.copy()
    tabla["posicion"] = posicion
    columnas = [
        ("sentido", "Sentido", 0.75),
        ("distribucion", "Distribucion", 0.9),
        ("representacion", "H final", 0.85),
        ("Score_mean", "Score", 0.68),
        ("Accuracy_mean", "Accuracy", 0.72),
        ("Sensitivity_mean", "Sensitivity", 0.76),
        ("Specificity_mean", "Specificity", 0.76),
        ("Precision_mean", "Precision", 0.72),
    ]
    pesos = [columna[2] for columna in columnas]
    anchos = [
        (ANCHO - 2 * MARGEN) * peso / sum(pesos) for peso in pesos
    ]
    y = 36
    x = MARGEN
    for (_, etiqueta, _), ancho in zip(columnas, anchos):
        rect = fitz.Rect(x, y, x + ancho, y + 19)
        pagina.draw_rect(rect, color=AZUL, fill=AZUL, width=0.2)
        _texto(pagina, rect, etiqueta, 6.2, True, BLANCO)
        x += ancho
    y += 19
    for numero, (_, fila) in enumerate(tabla.iterrows()):
        x = MARGEN
        fondo = VERDE if bool(fila["seleccionada"]) else (
            BLANCO if numero == 0 else GRIS
        )
        for (columna, _, _), ancho in zip(columnas, anchos):
            rect = fitz.Rect(x, y, x + ancho, y + 25)
            pagina.draw_rect(
                rect,
                color=(0.72, 0.75, 0.78),
                fill=fondo,
                width=0.2,
            )
            valor = fila[columna]
            if isinstance(valor, float):
                valor = f"{valor:.4f}"
            _texto(pagina, rect, valor, 6.2)
            x += ancho
        y += 25
    pagina.insert_image(
        fitz.Rect(MARGEN, 112, ANCHO / 2 - 8, 510),
        filename=str(rutas_png[0]),
        keep_proportion=True,
    )
    pagina.insert_image(
        fitz.Rect(ANCHO / 2 + 8, 112, ANCHO - MARGEN, 510),
        filename=str(rutas_png[1]),
        keep_proportion=True,
    )
    _texto(
        pagina,
        fitz.Rect(MARGEN, 518, ANCHO / 2 - 8, 555),
        (
            f"Mayor error: {datos_dec['mayor_error_real']} -> "
            f"{datos_dec['mayor_error_predicha']} "
            f"({int(datos_dec['mayor_error_cantidad'])} senales)"
        ),
        7.8,
    )
    _texto(
        pagina,
        fitz.Rect(ANCHO / 2 + 8, 518, ANCHO - MARGEN, 555),
        (
            f"Mayor error: {datos_inc['mayor_error_real']} -> "
            f"{datos_inc['mayor_error_predicha']} "
            f"({int(datos_inc['mayor_error_cantidad'])} senales)"
        ),
        7.8,
    )


def generar_pdf_resumen(
    ruta: Path,
    tabla: pd.DataFrame,
    manifiesto: pd.DataFrame,
    titulo: str,
) -> Path:
    documento = fitz.open()
    for inicio in (0, 20):
        _tabla_pagina(
            documento,
            titulo if inicio == 0 else f"{titulo} (continuacion)",
            tabla.iloc[inicio : inicio + 20],
            primera=inicio == 0,
            resaltar=True,
        )
    for posicion in range(1, 21):
        _pagina_matrices(
            documento,
            posicion,
            tabla[tabla["posicion"].eq(posicion)],
            manifiesto,
        )
    if len(documento) != 22:
        raise AssertionError("Cada resumen debe tener 22 paginas")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.stem + "_temporal.pdf")
    documento.save(temporal, garbage=4, deflate=True)
    documento.close()
    temporal.replace(ruta)
    return ruta

