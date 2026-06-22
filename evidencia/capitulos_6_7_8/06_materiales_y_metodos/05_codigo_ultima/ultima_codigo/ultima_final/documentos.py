from __future__ import annotations

from pathlib import Path
import textwrap

import fitz
import pandas as pd


ANCHO = 841.89
ALTO = 595.28
MARGEN = 28
AZUL = (0.07, 0.20, 0.30)
GRIS = (0.94, 0.95, 0.96)
BLANCO = (1.0, 1.0, 1.0)
NEGRO = (0.08, 0.08, 0.08)


def _texto(
    pagina: fitz.Page,
    rect: fitz.Rect,
    texto: object,
    tamano: float = 8.0,
    negrita: bool = False,
    color=NEGRO,
    align: int = 0,
) -> None:
    pagina.insert_textbox(
        rect,
        str(texto),
        fontname="hebo" if negrita else "helv",
        fontsize=tamano,
        color=color,
        align=align,
    )


def _portada(documento: fitz.Document, titulo: str, subtitulo: str) -> None:
    pagina = documento.new_page(width=ANCHO, height=ALTO)
    pagina.draw_rect(fitz.Rect(0, 0, ANCHO, 82), color=AZUL, fill=AZUL)
    _texto(
        pagina,
        fitz.Rect(MARGEN, 22, ANCHO - MARGEN, 60),
        titulo,
        tamano=20,
        negrita=True,
        color=BLANCO,
    )
    _texto(
        pagina,
        fitz.Rect(MARGEN, 115, ANCHO - MARGEN, 210),
        subtitulo,
        tamano=11,
    )


def _tabla(
    documento: fitz.Document,
    titulo: str,
    tabla: pd.DataFrame,
    columnas: list[tuple[str, str, float]],
    filas_por_pagina: int = 18,
) -> None:
    if tabla.empty:
        pagina = documento.new_page(width=ANCHO, height=ALTO)
        _texto(pagina, fitz.Rect(MARGEN, MARGEN, ANCHO - MARGEN, 60), titulo, 14, True, AZUL)
        _texto(pagina, fitz.Rect(MARGEN, 80, ANCHO - MARGEN, 110), "Sin datos disponibles.", 10)
        return
    pesos = [columna[2] for columna in columnas]
    anchos = [(ANCHO - 2 * MARGEN) * peso / sum(pesos) for peso in pesos]
    for inicio in range(0, len(tabla), filas_por_pagina):
        pagina = documento.new_page(width=ANCHO, height=ALTO)
        _texto(pagina, fitz.Rect(MARGEN, 18, ANCHO - MARGEN, 48), titulo, 14, True, AZUL)
        y = 60
        alto_cabecera = 28
        alto_fila = 24
        x = MARGEN
        for (_, etiqueta, _), ancho in zip(columnas, anchos):
            rect = fitz.Rect(x, y, x + ancho, y + alto_cabecera)
            pagina.draw_rect(rect, color=AZUL, fill=AZUL, width=0.3)
            _texto(pagina, rect, etiqueta, 7.6, True, BLANCO, 1)
            x += ancho
        y += alto_cabecera
        bloque = tabla.iloc[inicio : inicio + filas_por_pagina]
        for numero, (_, fila) in enumerate(bloque.iterrows()):
            x = MARGEN
            fondo = BLANCO if numero % 2 == 0 else GRIS
            for indice_columna, (columna, _, _) in enumerate(columnas):
                ancho = anchos[indice_columna]
                rect = fitz.Rect(x, y, x + ancho, y + alto_fila)
                pagina.draw_rect(rect, color=(0.74, 0.76, 0.78), fill=fondo, width=0.25)
                valor = fila.get(columna, "")
                if isinstance(valor, float):
                    valor = f"{valor:.4f}"
                _texto(pagina, rect, valor, 6.8, align=1)
                x += ancho
            y += alto_fila


def _matrices(documento: fitz.Document, titulo: str, tabla: pd.DataFrame) -> None:
    if tabla.empty or "ruta_matriz_png" not in tabla.columns:
        return
    for inicio in range(0, len(tabla), 4):
        pagina = documento.new_page(width=ANCHO, height=ALTO)
        _texto(pagina, fitz.Rect(MARGEN, 16, ANCHO - MARGEN, 42), titulo, 13, True, AZUL)
        bloque = tabla.iloc[inicio : inicio + 4]
        ancho = (ANCHO - 2 * MARGEN - 18) / 2
        alto = (ALTO - 70 - MARGEN - 14) / 2
        for indice, (_, fila) in enumerate(bloque.iterrows()):
            col = indice % 2
            fil = indice // 2
            x0 = MARGEN + col * (ancho + 18)
            y0 = 58 + fil * (alto + 14)
            etiqueta = f"{fila.get('distribucion', '')} / {fila.get('representacion', '')}"
            _texto(pagina, fitz.Rect(x0, y0, x0 + ancho, y0 + 18), etiqueta, 8, True, AZUL, 1)
            ruta = Path(str(fila["ruta_matriz_png"]))
            if ruta.exists():
                pagina.insert_image(
                    fitz.Rect(x0 + 12, y0 + 20, x0 + ancho - 12, y0 + alto),
                    filename=str(ruta),
                    keep_proportion=True,
                )


def guardar_pdf(documento: fitz.Document, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.stem + "_temporal.pdf")
    documento.save(temporal, garbage=4, deflate=True)
    documento.close()
    temporal.replace(ruta)
    return ruta


def generar_pdf_optimizacion(
    titulo: str,
    carpeta_optimizacion: Path,
    ruta_pdf: Path,
) -> Path:
    tabla = pd.read_csv(carpeta_optimizacion / "tablas_csv" / "mejores_por_capas.csv", encoding="utf-8-sig")
    documento = fitz.open()
    _portada(
        documento,
        titulo,
        "Ranking por UjaNet multiclase sobre la activacion temporal final. "
        "Se muestran hasta cuatro configuraciones por profundidad.",
    )
    columnas = [
        ("numero_capas", "Capas", 0.7),
        ("distribucion", "Bases", 1.4),
        ("representacion", "Representacion", 1.2),
        ("Accuracy_mean", "Accuracy", 1.0),
        ("Score_mean", "Score", 1.0),
        ("Sensitivity_mean", "Sensitivity", 1.0),
        ("Specificity_mean", "Specificity", 1.0),
        ("Precision_mean", "Precision", 1.0),
        ("exactitud_directa", "Exactitud directa", 1.1),
    ]
    _tabla(documento, "Mejores configuraciones por numero de capas", tabla, columnas, filas_por_pagina=18)
    _matrices(documento, "Matrices de confusion UjaNet multiclase", tabla)
    return guardar_pdf(documento, ruta_pdf)


def generar_pdf_resultados(
    titulo: str,
    carpeta_resultados: Path,
    ruta_pdf: Path,
) -> Path:
    tabla = pd.read_csv(carpeta_resultados / "tablas_csv" / "resultados_por_snr.csv", encoding="utf-8-sig")
    tabla["distribucion"] = tabla["distribucion"].fillna("").astype(str).str.strip()
    tabla = tabla.drop_duplicates(
        subset=["base", "representacion", "distribucion"],
        keep="last",
    ).reset_index(drop=True)
    tabla.to_csv(carpeta_resultados / "tablas_csv" / "resultados_por_snr.csv", index=False, encoding="utf-8-sig")
    documento = fitz.open()
    _portada(
        documento,
        titulo,
        "Comparativa de bases ruidosas con STFT, MFCC, Mel, LogMel y las cuatro "
        "configuraciones Deep-ONMF optimas seleccionadas.",
    )
    columnas = [
        ("base", "Base", 1.0),
        ("snr_db", "SNR", 0.7),
        ("representacion", "Representacion", 1.6),
        ("distribucion", "Bases Deep", 1.3),
        ("Accuracy_mean", "Accuracy", 1.0),
        ("Score_mean", "Score", 1.0),
        ("Sensitivity_mean", "Sensitivity", 1.0),
        ("Specificity_mean", "Specificity", 1.0),
        ("Precision_mean", "Precision", 1.0),
    ]
    _tabla(documento, "Resultados por SNR", tabla, columnas, filas_por_pagina=18)
    resumen = []
    for base, grupo in tabla.groupby("base"):
        mejor = grupo.sort_values(["Accuracy_mean", "Score_mean"], ascending=False).iloc[0]
        resumen.append(
            f"{base}: mejor representacion {mejor['representacion']} "
            f"(Accuracy={float(mejor['Accuracy_mean']):.4f}, Score={float(mejor['Score_mean']):.4f})."
        )
    pagina = documento.new_page(width=ANCHO, height=ALTO)
    _texto(pagina, fitz.Rect(MARGEN, 20, ANCHO - MARGEN, 48), "Resumen de lectura", 14, True, AZUL)
    y = 70
    for linea in resumen:
        for trozo in textwrap.wrap(linea, width=115):
            _texto(pagina, fitz.Rect(MARGEN, y, ANCHO - MARGEN, y + 18), trozo, 9)
            y += 18
        y += 6
    return guardar_pdf(documento, ruta_pdf)
