from __future__ import annotations

from pathlib import Path

import fitz
import pandas as pd


ANCHO = 841.89
ALTO = 595.28
MARGEN_X = 24.0
MARGEN_Y = 20.0
AZUL = (0.16, 0.31, 0.45)
GRIS = (0.95, 0.96, 0.97)
BLANCO = (1.0, 1.0, 1.0)
NEGRO = (0.10, 0.12, 0.14)

COLUMNAS_TABLA = [
    "clasificador",
    "representacion",
    "distribucion",
    "Score_mean",
    "Accuracy_mean",
    "Sensitivity_mean",
    "Specificity_mean",
    "Precision_mean",
]

NOMBRES_COLUMNAS = {
    "clasificador": "Clasificador",
    "representacion": "Representación",
    "distribucion": "Distribución",
    "Score_mean": "Score",
    "Accuracy_mean": "Accuracy",
    "Sensitivity_mean": "Sensitivity",
    "Specificity_mean": "Specificity",
    "Precision_mean": "Precision",
}


def _formatear(valor: object, columna: str) -> str:
    if columna.endswith("_mean"):
        return f"{float(valor):.4f}"
    return str(valor)


def _texto_celda(
    pagina: fitz.Page,
    rect: fitz.Rect,
    texto: str,
    fuente: str,
    tamano: float,
    color: tuple[float, float, float],
) -> None:
    tamano_real = tamano
    ancho_disponible = rect.width - 4
    while (
        fitz.get_text_length(texto, fontname=fuente, fontsize=tamano_real)
        > ancho_disponible
        and tamano_real > 4.5
    ):
        tamano_real -= 0.2
    ancho_texto = fitz.get_text_length(
        texto,
        fontname=fuente,
        fontsize=tamano_real,
    )
    x = rect.x0 + max(2.0, (rect.width - ancho_texto) / 2.0)
    y = rect.y0 + (rect.height + tamano_real * 0.72) / 2.0
    pagina.insert_text(
        (x, y),
        texto,
        fontname=fuente,
        fontsize=tamano_real,
        color=color,
    )


def _insertar_imagen(
    pagina: fitz.Page,
    ruta: Path,
    rect: fitz.Rect,
) -> None:
    pagina.insert_image(rect, filename=str(ruta), keep_proportion=True)


def _anchos_columnas() -> list[float]:
    ancho_util = ANCHO - 2 * MARGEN_X
    pesos = [1.05, 1.65, 1.05, 0.86, 0.94, 1.04, 1.04, 0.94]
    return [ancho_util * peso / sum(pesos) for peso in pesos]


def _dibujar_cabecera_tabla(
    pagina: fitz.Page,
    y: float,
    alto: float,
) -> float:
    x = MARGEN_X
    for columna, ancho_columna in zip(COLUMNAS_TABLA, _anchos_columnas()):
        rect = fitz.Rect(x, y, x + ancho_columna, y + alto)
        pagina.draw_rect(rect, color=AZUL, fill=AZUL, width=0.35)
        _texto_celda(
            pagina,
            rect,
            NOMBRES_COLUMNAS[columna],
            "hebo",
            7.0,
            BLANCO,
        )
        x += ancho_columna
    return y + alto


def _dibujar_fila_tabla(
    pagina: fitz.Page,
    fila: pd.Series,
    y: float,
    alto: float,
    indice: int,
) -> float:
    x = MARGEN_X
    fondo = BLANCO if indice % 2 == 0 else GRIS
    for columna, ancho_columna in zip(COLUMNAS_TABLA, _anchos_columnas()):
        rect = fitz.Rect(x, y, x + ancho_columna, y + alto)
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
            "helv",
            7.2,
            NEGRO,
        )
        x += ancho_columna
    return y + alto


def _preparar_tabla(
    tablas: dict[str, pd.DataFrame],
    tipo_evaluacion: str,
) -> pd.DataFrame:
    clave = (
        "tabla_global_binaria_completa_36_filas"
        if tipo_evaluacion == "binaria"
        else "tabla_global_multiclase_completa_36_filas"
    )
    tabla = tablas[clave].copy()
    tabla = tabla.sort_values(
        ["Score_mean", "Accuracy_mean"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(tabla) != 36:
        raise AssertionError(
            f"La tabla {tipo_evaluacion} ampliada debe tener 36 filas, no {len(tabla)}"
        )
    return tabla


def _firmas_tabla(tabla: pd.DataFrame, tipo_evaluacion: str) -> pd.DataFrame:
    filas = []
    for indice, fila in tabla.iterrows():
        firma = "\n".join(_formatear(fila[columna], columna) for columna in COLUMNAS_TABLA)
        filas.append(
            {
                "evaluacion": tipo_evaluacion,
                "fila": indice + 1,
                "firma_esperada": firma,
            }
        )
    return pd.DataFrame(filas)


def generar_pdf(
    ruta_pdf: Path,
    tablas: dict[str, pd.DataFrame],
    graficas: list[Path],
    auditoria_base: pd.DataFrame,
    protocolo: pd.DataFrame,
    carpeta_tablas: Path,
) -> pd.DataFrame:
    """Genera los bloques binario y multiclase con sus figuras y tablas."""

    del auditoria_base, protocolo
    documento = fitz.open()

    auditorias: list[pd.DataFrame] = []
    for tipo_evaluacion in ("binaria", "multiclase"):
        tabla = _preparar_tabla(tablas, tipo_evaluacion)
        rutas_por_clasificador = {
            clasificador: next(
                ruta
                for ruta in graficas
                if tipo_evaluacion in ruta.stem and clasificador in ruta.stem
            )
            for clasificador in ("SVM", "KNN", "UjaNet")
        }

        # Primera página del bloque: SVM y KNN.
        pagina = documento.new_page(width=ANCHO, height=ALTO)
        separacion = 8.0
        alto_imagen = (ALTO - 2 * MARGEN_Y - separacion) / 2
        _insertar_imagen(
            pagina,
            rutas_por_clasificador["SVM"],
            fitz.Rect(
                MARGEN_X,
                MARGEN_Y,
                ANCHO - MARGEN_X,
                MARGEN_Y + alto_imagen,
            ),
        )
        _insertar_imagen(
            pagina,
            rutas_por_clasificador["KNN"],
            fitz.Rect(
                MARGEN_X,
                MARGEN_Y + alto_imagen + separacion,
                ANCHO - MARGEN_X,
                ALTO - MARGEN_Y,
            ),
        )

        # Segunda página del bloque: UjaNet y comienzo de la tabla.
        pagina = documento.new_page(width=ANCHO, height=ALTO)
        alto_ujanet = 250.0
        _insertar_imagen(
            pagina,
            rutas_por_clasificador["UjaNet"],
            fitz.Rect(
                MARGEN_X,
                MARGEN_Y,
                ANCHO - MARGEN_X,
                MARGEN_Y + alto_ujanet,
            ),
        )
        y = MARGEN_Y + alto_ujanet + 5
        pagina.insert_text(
            (MARGEN_X, y + 11),
            f"Tabla {tipo_evaluacion} ampliada",
            fontname="hebo",
            fontsize=11,
            color=AZUL,
        )
        y += 17
        alto_cabecera = 19.0
        alto_fila = 21.5
        y = _dibujar_cabecera_tabla(pagina, y, alto_cabecera)

        indice = 0
        while indice < len(tabla) and y + alto_fila <= ALTO - MARGEN_Y:
            y = _dibujar_fila_tabla(pagina, tabla.iloc[indice], y, alto_fila, indice)
            indice += 1

        # Página(s) siguientes: continuación inmediata de la tabla.
        while indice < len(tabla):
            pagina = documento.new_page(width=ANCHO, height=ALTO)
            y = MARGEN_Y
            y = _dibujar_cabecera_tabla(pagina, y, alto_cabecera)
            while indice < len(tabla) and y + alto_fila <= ALTO - MARGEN_Y:
                y = _dibujar_fila_tabla(
                    pagina,
                    tabla.iloc[indice],
                    y,
                    alto_fila,
                    indice,
                )
                indice += 1

        tabla[COLUMNAS_TABLA].to_csv(
            carpeta_tablas / f"tabla_{tipo_evaluacion}_ampliada.csv",
            index=False,
            encoding="utf-8-sig",
        )
        auditorias.append(_firmas_tabla(tabla, tipo_evaluacion))

    documento.set_metadata(
        {
            "title": "Comparaciones W-H3 y tablas binaria y multiclase",
            "author": "TFG de Alberto",
            "subject": "Distribuciones 9-8-7, 15-10-5, 10-6-4 y 8-5-3",
        }
    )
    ruta_pdf.parent.mkdir(parents=True, exist_ok=True)
    ruta_temporal = ruta_pdf.with_name(f"{ruta_pdf.stem}_temporal.pdf")
    documento.save(ruta_temporal, garbage=4, deflate=True)
    documento.close()
    ruta_temporal.replace(ruta_pdf)

    auditoria = pd.concat(auditorias, ignore_index=True)
    auditoria.to_csv(
        carpeta_tablas / "auditoria_filas_incluidas_pdf.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return auditoria


def auditar_pdf(
    ruta_pdf: Path,
    auditoria_filas: pd.DataFrame,
    carpeta: Path,
) -> pd.DataFrame:
    documento = fitz.open(ruta_pdf)
    texto = "\n".join(pagina.get_text() for pagina in documento)
    filas: list[dict[str, object]] = []

    firmas_ausentes = [
        f"{fila['evaluacion']}:{int(fila['fila'])}"
        for _, fila in auditoria_filas.iterrows()
        if str(fila["firma_esperada"]) not in texto
    ]
    todas_horizontales = all(pagina.rect.width > pagina.rect.height for pagina in documento)
    terminos_obligatorios = [
        "Tabla binaria ampliada",
        "Tabla multiclase ampliada",
        "9-8-7",
        "15-10-5",
        "10-6-4",
        "8-5-3",
        "DeepONMF_W",
        "DeepONMF_H3",
        "SVM",
        "KNN",
        "UjaNet",
    ]
    terminos_prohibidos = [
        "Metodología",
        "Auditoría",
        "Conclusiones",
        "TP, TN, FP y FN",
        "SHA-256",
    ]

    filas.append(
        {
            "comprobacion": "72_filas_csv_pdf",
            "correcto": len(auditoria_filas) == 72 and not firmas_ausentes,
            "detalle": f"filas={len(auditoria_filas)}; ausentes={firmas_ausentes}",
        }
    )
    filas.append(
        {
            "comprobacion": "orientacion_horizontal",
            "correcto": todas_horizontales,
            "detalle": f"paginas={len(documento)}",
        }
    )
    filas.append(
        {
            "comprobacion": "contenido_obligatorio",
            "correcto": all(termino in texto for termino in terminos_obligatorios),
            "detalle": ", ".join(
                termino for termino in terminos_obligatorios if termino not in texto
            )
            or "completo",
        }
    )
    filas.append(
        {
            "comprobacion": "sin_contenido_adicional",
            "correcto": all(termino not in texto for termino in terminos_prohibidos),
            "detalle": ", ".join(
                termino for termino in terminos_prohibidos if termino in texto
            )
            or "correcto",
        }
    )

    documento.close()
    resultado = pd.DataFrame(filas)
    resultado.to_csv(carpeta / "auditoria_pdf.csv", index=False, encoding="utf-8-sig")
    if not resultado["correcto"].all():
        fallos = resultado.loc[~resultado["correcto"], "comprobacion"].tolist()
        raise AssertionError(f"El PDF no supera la auditoría: {fallos}")
    return resultado
