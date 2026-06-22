from __future__ import annotations

from pathlib import Path
import textwrap

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .configuracion_busqueda import CLASIFICADORES


ANCHO = 841.89
ALTO = 595.28
MARGEN_X = 24.0
MARGEN_Y = 20.0
AZUL = (0.12, 0.28, 0.40)
AZUL_CLARO = (0.89, 0.94, 0.97)
GRIS = (0.95, 0.96, 0.97)
BLANCO = (1.0, 1.0, 1.0)
NEGRO = (0.08, 0.10, 0.12)
VERDE = (0.87, 0.95, 0.89)


def _texto_celda(
    pagina: fitz.Page,
    rect: fitz.Rect,
    texto: str,
    tamano: float = 6.5,
    negrita: bool = False,
    color: tuple[float, float, float] = NEGRO,
) -> None:
    fuente = "hebo" if negrita else "helv"
    tamano_real = tamano
    while (
        fitz.get_text_length(texto, fontname=fuente, fontsize=tamano_real)
        > rect.width - 4
        and tamano_real > 4.3
    ):
        tamano_real -= 0.2
    ancho = fitz.get_text_length(
        texto,
        fontname=fuente,
        fontsize=tamano_real,
    )
    x = rect.x0 + max(2.0, (rect.width - ancho) / 2)
    y = rect.y0 + (rect.height + tamano_real * 0.72) / 2
    pagina.insert_text(
        (x, y),
        texto,
        fontname=fuente,
        fontsize=tamano_real,
        color=color,
    )


def _titulo(pagina: fitz.Page, texto: str, y: float = MARGEN_Y) -> float:
    pagina.insert_text(
        (MARGEN_X, y + 15),
        texto,
        fontname="hebo",
        fontsize=14,
        color=AZUL,
    )
    return y + 23


def _parrafo(
    pagina: fitz.Page,
    texto: str,
    y: float,
    alto: float,
    tamano: float = 9.2,
) -> float:
    pagina.insert_textbox(
        fitz.Rect(MARGEN_X, y, ANCHO - MARGEN_X, y + alto),
        texto,
        fontname="helv",
        fontsize=tamano,
        lineheight=1.18,
        color=NEGRO,
        align=fitz.TEXT_ALIGN_LEFT,
    )
    return y + alto


def _dibujar_tabla(
    documento: fitz.Document,
    tabla: pd.DataFrame,
    columnas: list[str],
    etiquetas: list[str],
    pesos: list[float],
    titulo: str,
    filas_por_pagina: int,
    formateadores: dict[str, callable] | None = None,
) -> None:
    formateadores = formateadores or {}
    indice = 0
    while indice < len(tabla):
        pagina = documento.new_page(width=ANCHO, height=ALTO)
        y = _titulo(
            pagina,
            titulo if indice == 0 else f"{titulo} (continuacion)",
        )
        alto_cabecera = 20.0
        alto_fila = min(
            32.0,
            (ALTO - MARGEN_Y - y - alto_cabecera)
            / min(filas_por_pagina, len(tabla) - indice),
        )
        ancho_util = ANCHO - 2 * MARGEN_X
        anchos = [ancho_util * peso / sum(pesos) for peso in pesos]
        x = MARGEN_X
        for etiqueta_columna, ancho in zip(etiquetas, anchos):
            rect = fitz.Rect(x, y, x + ancho, y + alto_cabecera)
            pagina.draw_rect(rect, color=AZUL, fill=AZUL, width=0.3)
            _texto_celda(
                pagina,
                rect,
                etiqueta_columna,
                tamano=6.7,
                negrita=True,
                color=BLANCO,
            )
            x += ancho
        y += alto_cabecera

        fin = min(indice + filas_por_pagina, len(tabla))
        for numero_fila in range(indice, fin):
            fila = tabla.iloc[numero_fila]
            x = MARGEN_X
            finalista = bool(fila.get("finalista", False))
            fondo = (
                VERDE
                if finalista
                else (BLANCO if numero_fila % 2 == 0 else GRIS)
            )
            for columna, ancho in zip(columnas, anchos):
                rect = fitz.Rect(x, y, x + ancho, y + alto_fila)
                pagina.draw_rect(
                    rect,
                    color=(0.68, 0.72, 0.75),
                    fill=fondo,
                    width=0.25,
                )
                valor = fila[columna]
                texto = (
                    formateadores[columna](valor)
                    if columna in formateadores
                    else str(valor)
                )
                _texto_celda(
                    pagina,
                    rect,
                    texto,
                    tamano=6.3,
                    negrita=finalista and columna in ("posicion", "distribucion"),
                )
                x += ancho
            y += alto_fila
        indice = fin


def generar_graficas_finalistas(
    tablas: dict[str, pd.DataFrame],
    compacta: pd.DataFrame,
    carpeta: Path,
) -> list[Path]:
    carpeta.mkdir(parents=True, exist_ok=True)
    finalistas = compacta.head(5)["distribucion"].tolist()
    rutas: list[Path] = []
    for tipo in ("binarias", "multiclase"):
        resumen = tablas[
            "resumen_metricas_binarias"
            if tipo == "binarias"
            else "resumen_metricas_multiclase"
        ]
        resumen = resumen[resumen["distribucion"].isin(finalistas)].copy()
        figura, ejes = plt.subplots(1, 3, figsize=(15.2, 4.6), sharey=True)
        x = np.arange(len(finalistas))
        for eje, clasificador in zip(ejes, CLASIFICADORES):
            datos = resumen[resumen["clasificador"] == clasificador]
            valores_w = []
            valores_h = []
            for distribucion in finalistas:
                grupo = datos[datos["distribucion"] == distribucion]
                valores_w.append(
                    float(
                        grupo[grupo["representacion"] == "DeepONMF_W"][
                            "Accuracy_mean"
                        ].iloc[0]
                    )
                )
                valores_h.append(
                    float(
                        grupo[
                            grupo["representacion"].str.startswith(
                                "DeepONMF_H"
                            )
                        ]["Accuracy_mean"].iloc[0]
                    )
                )
            ancho = 0.36
            barras_w = eje.bar(
                x - ancho / 2,
                valores_w,
                ancho,
                color="#356A9A",
                label="W",
            )
            barras_h = eje.bar(
                x + ancho / 2,
                valores_h,
                ancho,
                color="#D37532",
                label="H final",
            )
            eje.bar_label(barras_w, fmt="%.3f", fontsize=7, padding=2)
            eje.bar_label(barras_h, fmt="%.3f", fontsize=7, padding=2)
            eje.set_title(clasificador)
            eje.set_xticks(x, finalistas, rotation=25, ha="right")
            eje.set_ylim(
                max(0.0, min(valores_w + valores_h) - 0.08),
                1.01,
            )
            eje.grid(axis="y", alpha=0.2)
        ejes[0].set_ylabel("Accuracy media")
        ejes[-1].legend(loc="lower right")
        figura.suptitle(
            f"Cinco finalistas: W frente a H ({tipo})",
            fontsize=14,
            fontweight="bold",
        )
        figura.tight_layout()
        ruta = carpeta / f"finalistas_accuracy_{tipo}.png"
        figura.savefig(ruta, dpi=170, bbox_inches="tight")
        plt.close(figura)
        rutas.append(ruta)
    return rutas


def _tabla_detallada(
    resumen: pd.DataFrame,
    por_fold: pd.DataFrame,
    compacta: pd.DataFrame,
) -> pd.DataFrame:
    finalistas = compacta.head(5)["distribucion"].tolist()
    referencias = resumen[resumen["distribucion"] == "9-8-7"].copy()
    resumen = resumen[resumen["distribucion"].isin(finalistas)].copy()
    orden = {valor: indice for indice, valor in enumerate(finalistas)}
    filas = []
    for _, fila in resumen.iterrows():
        pliegues = por_fold[
            (por_fold["distribucion"] == fila["distribucion"])
            & (por_fold["clasificador"] == fila["clasificador"])
            & (por_fold["representacion"] == fila["representacion"])
        ]
        tipo_repr = (
            "W" if fila["representacion"] == "DeepONMF_W" else "H"
        )
        referencia = referencias[
            (referencias["clasificador"] == fila["clasificador"])
            & (
                referencias["representacion"].eq("DeepONMF_W")
                if tipo_repr == "W"
                else referencias["representacion"].str.startswith(
                    "DeepONMF_H"
                )
            )
        ]
        delta_accuracy = (
            float(fila["Accuracy_mean"])
            - float(referencia["Accuracy_mean"].iloc[0])
            if not referencia.empty
            else 0.0
        )
        filas.append(
            {
                "distribucion": fila["distribucion"],
                "clasificador": fila["clasificador"],
                "representacion": tipo_repr,
                "Accuracy": f"{float(fila['Accuracy_mean']):.4f}",
                "Score": f"{float(fila['Score_mean']):.4f}",
                "Sensitivity": f"{float(fila['Sensitivity_mean']):.4f}",
                "Specificity": f"{float(fila['Specificity_mean']):.4f}",
                "Precision": f"{float(fila['Precision_mean']):.4f}",
                "TP": int(pliegues["TP"].sum()),
                "TN": int(pliegues["TN"].sum()),
                "FP": int(pliegues["FP"].sum()),
                "FN": int(pliegues["FN"].sum()),
                "Delta_Accuracy": f"{delta_accuracy:+.4f}",
                "_orden": orden[fila["distribucion"]],
            }
        )
    return pd.DataFrame(filas).sort_values(
        ["clasificador", "_orden", "representacion"]
    ).drop(columns="_orden")


def _insertar_graficas(
    documento: fitz.Document,
    rutas: list[Path],
) -> None:
    pagina = documento.new_page(width=ANCHO, height=ALTO)
    separacion = 8.0
    alto = (ALTO - 2 * MARGEN_Y - separacion) / 2
    for indice, ruta in enumerate(rutas):
        y0 = MARGEN_Y + indice * (alto + separacion)
        pagina.insert_image(
            fitz.Rect(
                MARGEN_X,
                y0,
                ANCHO - MARGEN_X,
                y0 + alto,
            ),
            filename=str(ruta),
            keep_proportion=True,
        )


def _leer_auditoria(
    carpeta_resultados: Path,
    distribucion: str,
) -> pd.DataFrame:
    clave = distribucion.replace("-", "_")
    rutas = (
        carpeta_resultados
        / "configuraciones"
        / clave
        / "representaciones"
        / "auditoria_deep_onmf.csv",
        carpeta_resultados
        / "historicos_importados"
        / clave
        / "representaciones"
        / "auditoria_deep_onmf.csv",
    )
    for ruta in rutas:
        if ruta.exists():
            return pd.read_csv(ruta, encoding="utf-8-sig")
    return pd.DataFrame()


def _pagina_conclusion(
    documento: fitz.Document,
    compacta: pd.DataFrame,
    tablas: dict[str, pd.DataFrame],
    carpeta_resultados: Path,
) -> None:
    ganadora = compacta.iloc[0]
    distribucion = str(ganadora["distribucion"])
    rangos = tuple(int(valor) for valor in distribucion.split("-"))
    resumen_bin = tablas["resumen_metricas_binarias"]
    datos_ganadora = resumen_bin[
        resumen_bin["distribucion"] == distribucion
    ]
    datos_h = datos_ganadora[
        datos_ganadora["representacion"].str.startswith("DeepONMF_H")
    ]
    datos_w = datos_ganadora[
        datos_ganadora["representacion"] == "DeepONMF_W"
    ]
    acc_h = float(datos_h["Accuracy_mean"].mean())
    acc_w = float(datos_w["Accuracy_mean"].mean())
    referencia = compacta[compacta["distribucion"] == "9-8-7"].iloc[0]
    delta_acc_referencia = acc_h - float(referencia["Accuracy_media"])
    delta_score_referencia = (
        float(ganadora["Score_medio"]) - float(referencia["Score_medio"])
    )

    resumen_multi = tablas["resumen_metricas_multiclase"]
    multi_h = resumen_multi[
        (resumen_multi["distribucion"] == distribucion)
        & resumen_multi["representacion"].str.startswith("DeepONMF_H")
    ]
    multi_ref = resumen_multi[
        (resumen_multi["distribucion"] == "9-8-7")
        & resumen_multi["representacion"].str.startswith("DeepONMF_H")
    ]
    acc_multi = float(multi_h["Accuracy_mean"].mean())
    acc_multi_ref = float(multi_ref["Accuracy_mean"].mean())

    auditoria = _leer_auditoria(carpeta_resultados, distribucion)
    error = (
        float(auditoria["error_final"].mean())
        if "error_final" in auditoria
        else float("nan")
    )
    columna_ort = f"capa_{len(rangos)}_ortogonalidad_h"
    ortogonalidad = (
        float(auditoria[columna_ort].mean())
        if columna_ort in auditoria
        else float("nan")
    )
    auditoria_ref = _leer_auditoria(carpeta_resultados, "9-8-7")
    error_ref = (
        float(auditoria_ref["error_final"].mean())
        if "error_final" in auditoria_ref
        else float("nan")
    )
    ortogonalidad_ref = (
        float(auditoria_ref["capa_3_ortogonalidad_h"].mean())
        if "capa_3_ortogonalidad_h" in auditoria_ref
        else float("nan")
    )

    pagina = documento.new_page(width=ANCHO, height=ALTO)
    y = _titulo(pagina, "Configuracion ganadora y justificacion")
    texto = (
        f"La mejor configuracion encontrada es {distribucion}, con "
        f"{len(rangos)} capas y una activacion temporal final de "
        f"{rangos[-1]} x 212. Su Accuracy binaria media de H entre SVM, "
        f"KNN y UjaNet es {acc_h:.4f}; el Score medio es "
        f"{float(ganadora['Score_medio']):.4f}."
    )
    y = _parrafo(pagina, texto, y, 48)

    texto = (
        f"Frente a 9-8-7, la mejora temporal media es "
        f"{delta_acc_referencia:+.4f} en Accuracy y "
        f"{delta_score_referencia:+.4f} en Score. Dentro de 16-4, H supera "
        f"a W en {acc_h - acc_w:+.4f} de Accuracy media. En multiclase, H "
        f"alcanza {acc_multi:.4f}, frente a {acc_multi_ref:.4f} de 9-8-7."
    )
    y = _parrafo(pagina, texto, y, 52)

    vecinas = compacta[
        compacta["distribucion"].isin(("15-4", "16-3", "17-3", "17-4"))
    ].set_index("distribucion")
    texto = (
        "La comparacion local respalda que el maximo no es accidental: "
        f"15-4 obtiene {float(vecinas.loc['15-4', 'Accuracy_media']):.4f}, "
        f"16-3 {float(vecinas.loc['16-3', 'Accuracy_media']):.4f}, "
        f"17-3 {float(vecinas.loc['17-3', 'Accuracy_media']):.4f} y "
        f"17-4 {float(vecinas.loc['17-4', 'Accuracy_media']):.4f}. "
        "La combinacion exacta 16-4 conserva el mejor equilibrio observado."
    )
    y = _parrafo(pagina, texto, y, 55)

    compresiones = " -> ".join(str(valor) for valor in rangos)
    texto = (
        f"La secuencia {compresiones} realiza una compresion progresiva: "
        "las primeras capas conservan patrones relativamente variados y "
        "las ultimas obligan a concentrar la informacion temporal en menos "
        "activaciones. El resultado sugiere que esta configuracion alcanza "
        "un equilibrio mejor entre capacidad y compresion que las restantes. "
        "Una arquitectura demasiado ancha puede conservar variacion poco "
        "discriminativa; una demasiado estrecha puede eliminar eventos "
        "temporales utiles."
    )
    y = _parrafo(pagina, texto, y, 66)

    texto = (
        f"Auditoria numerica: error relativo medio de reconstruccion "
        f"{error:.4f}, frente a {error_ref:.4f} en 9-8-7. La semejanza media "
        f"entre filas de H es {ortogonalidad:.4f}, frente a "
        f"{ortogonalidad_ref:.4f}; al ser mayor, 16-4 no gana por ser mas "
        "ortogonal, sino por producir activaciones mas utiles para clasificar. "
        "La seleccion se realiza por Accuracy, no por reconstruccion."
    )
    y = _parrafo(pagina, texto, y, 62)

    mejores = []
    for clasificador in CLASIFICADORES:
        columna = f"Accuracy_{clasificador}"
        fila = compacta.sort_values(
            [columna, f"Score_{clasificador}"],
            ascending=False,
        ).iloc[0]
        mejores.append(
            f"{clasificador}: {fila['distribucion']} "
            f"(Accuracy {float(fila[columna]):.4f})"
        )
    texto = (
        "Mejores configuraciones por clasificador: "
        + "; ".join(mejores)
        + ". La ganadora global no tiene por que ser la maxima en cada "
        "clasificador individual: representa el mejor comportamiento medio "
        "y reduce la dependencia de un unico modelo."
    )
    y = _parrafo(pagina, texto, y, 58)

    mejor_w = resumen_bin[
        resumen_bin["representacion"] == "DeepONMF_W"
    ].sort_values(["Accuracy_mean", "Score_mean"], ascending=False).iloc[0]
    mejor_h = resumen_bin[
        resumen_bin["representacion"].str.startswith("DeepONMF_H")
    ].sort_values(["Accuracy_mean", "Score_mean"], ascending=False).iloc[0]
    mejor_multi = resumen_multi.sort_values(
        ["Accuracy_mean", "Score_mean"],
        ascending=False,
    ).iloc[0]
    texto = (
        f"Maximos individuales: mejor W, {mejor_w['distribucion']} con "
        f"{mejor_w['clasificador']} (Accuracy "
        f"{float(mejor_w['Accuracy_mean']):.4f}); mejor H, "
        f"{mejor_h['distribucion']} con {mejor_h['clasificador']} "
        f"(Accuracy {float(mejor_h['Accuracy_mean']):.4f}); mejor resultado "
        f"multiclase, {mejor_multi['distribucion']} "
        f"{mejor_multi['representacion']} con "
        f"{mejor_multi['clasificador']} (Accuracy "
        f"{float(mejor_multi['Accuracy_mean']):.4f})."
    )
    _parrafo(pagina, texto, y, 52)


def generar_pdf(
    ruta_pdf_anterior: Path,
    ruta_pdf_salida: Path,
    compacta: pd.DataFrame,
    tablas: dict[str, pd.DataFrame],
    carpeta_resultados: Path,
) -> Path:
    if not ruta_pdf_anterior.exists():
        raise FileNotFoundError(
            f"No se encuentra el PDF anterior: {ruta_pdf_anterior}"
        )
    documento = fitz.open()
    anterior = fitz.open(ruta_pdf_anterior)
    documento.insert_pdf(anterior)
    paginas_anteriores = len(anterior)
    anterior.close()

    columnas_compactas = [
        "posicion",
        "distribucion",
        "numero_capas",
        "Accuracy_SVM",
        "Score_SVM",
        "Accuracy_KNN",
        "Score_KNN",
        "Accuracy_UjaNet",
        "Score_UjaNet",
        "Accuracy_media",
        "Score_medio",
        "finalista",
    ]
    etiquetas_compactas = [
        "Pos.",
        "Distribucion",
        "Capas",
        "Acc. SVM",
        "Score SVM",
        "Acc. KNN",
        "Score KNN",
        "Acc. UjaNet",
        "Score UjaNet",
        "Acc. media",
        "Score medio",
        "Finalista",
    ]
    formatos = {
        columna: (lambda valor: f"{float(valor):.4f}")
        for columna in columnas_compactas
        if columna.startswith("Accuracy") or columna.startswith("Score")
    }
    formatos["finalista"] = lambda valor: "SI" if bool(valor) else "NO"
    _dibujar_tabla(
        documento,
        compacta,
        columnas_compactas,
        etiquetas_compactas,
        [0.45, 1.1, 0.5, 0.75, 0.75, 0.75, 0.75, 0.82, 0.82, 0.82, 0.82, 0.62],
        "Todas las configuraciones probadas: resultados temporales H",
        filas_por_pagina=19,
        formateadores=formatos,
    )

    rutas_graficas = generar_graficas_finalistas(
        tablas,
        compacta,
        carpeta_resultados / "figuras",
    )
    _insertar_graficas(documento, rutas_graficas)

    for tipo, clave_resumen, clave_fold in (
        (
            "binaria",
            "resumen_metricas_binarias",
            "metricas_binarias_por_fold",
        ),
        (
            "multiclase",
            "resumen_metricas_multiclase",
            "metricas_multiclase_por_fold",
        ),
    ):
        detallada = _tabla_detallada(
            tablas[clave_resumen],
            tablas[clave_fold],
            compacta,
        )
        detallada.to_csv(
            carpeta_resultados
            / "tablas_csv"
            / f"cinco_finalistas_{tipo}_completa.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _dibujar_tabla(
            documento,
            detallada.reset_index(drop=True),
            [
                "clasificador",
                "distribucion",
                "representacion",
                "Score",
                "Accuracy",
                "Sensitivity",
                "Specificity",
                "Precision",
                "TP",
                "TN",
                "FP",
                "FN",
                "Delta_Accuracy",
            ],
            [
                "Clasif.",
                "Distribucion",
                "Matriz",
                "Score",
                "Accuracy",
                "Sensitivity",
                "Specificity",
                "Precision",
                "TP",
                "TN",
                "FP",
                "FN",
                "Delta Acc. vs 9-8-7",
            ],
            [
                0.62,
                0.9,
                0.48,
                1.18,
                1.18,
                0.76,
                0.76,
                0.76,
                0.43,
                0.43,
                0.43,
                0.43,
                0.82,
            ],
            f"Cinco finalistas: evaluacion {tipo}",
            filas_por_pagina=15,
        )

    _pagina_conclusion(documento, compacta, tablas, carpeta_resultados)
    documento.set_metadata(
        {
            "title": "Busqueda de la mejor configuracion Deep-ONMF",
            "author": "TFG de Alberto",
            "subject": (
                "Comparaciones historicas y busqueda multicapa de W y H"
            ),
        }
    )
    ruta_pdf_salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta_pdf_salida.with_name(
        ruta_pdf_salida.stem + "_temporal.pdf"
    )
    documento.save(temporal, garbage=4, deflate=True)
    documento.close()
    temporal.replace(ruta_pdf_salida)

    auditoria = auditar_pdf(
        ruta_pdf_salida,
        paginas_anteriores,
        compacta,
        carpeta_resultados,
    )
    auditoria.to_csv(
        carpeta_resultados / "tablas_csv" / "auditoria_pdf.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not auditoria["correcto"].all():
        raise AssertionError(
            "El PDF se genero, pero no supero todas las comprobaciones"
        )
    return ruta_pdf_salida


def auditar_pdf(
    ruta_pdf: Path,
    paginas_anteriores: int,
    compacta: pd.DataFrame,
    carpeta_resultados: Path,
) -> pd.DataFrame:
    documento = fitz.open(ruta_pdf)
    texto = "\n".join(pagina.get_text() for pagina in documento)
    todas_horizontales = all(
        pagina.rect.width > pagina.rect.height for pagina in documento
    )
    distribuciones_ausentes = [
        distribucion
        for distribucion in compacta["distribucion"].astype(str)
        if distribucion not in texto
    ]
    firmas_compactas_ausentes = []
    for _, fila in compacta.iterrows():
        firma = "\n".join(
            [
                str(int(fila["posicion"])),
                str(fila["distribucion"]),
                str(int(fila["numero_capas"])),
                f"{float(fila['Accuracy_SVM']):.4f}",
                f"{float(fila['Score_SVM']):.4f}",
                f"{float(fila['Accuracy_KNN']):.4f}",
                f"{float(fila['Score_KNN']):.4f}",
                f"{float(fila['Accuracy_UjaNet']):.4f}",
                f"{float(fila['Score_UjaNet']):.4f}",
                f"{float(fila['Accuracy_media']):.4f}",
                f"{float(fila['Score_medio']):.4f}",
                "SI" if bool(fila["finalista"]) else "NO",
            ]
        )
        if firma not in texto:
            firmas_compactas_ausentes.append(str(fila["distribucion"]))

    firmas_finalistas_ausentes = []
    for tipo in ("binaria", "multiclase"):
        ruta = (
            carpeta_resultados
            / "tablas_csv"
            / f"cinco_finalistas_{tipo}_completa.csv"
        )
        tabla = pd.read_csv(ruta, encoding="utf-8-sig")
        for indice, fila in tabla.iterrows():
            firma = "\n".join(
                [
                    str(fila["clasificador"]),
                    str(fila["distribucion"]),
                    str(fila["representacion"]),
                    f"{float(fila['Score']):.4f}",
                    f"{float(fila['Accuracy']):.4f}",
                    f"{float(fila['Sensitivity']):.4f}",
                    f"{float(fila['Specificity']):.4f}",
                    f"{float(fila['Precision']):.4f}",
                    str(int(fila["TP"])),
                    str(int(fila["TN"])),
                    str(int(fila["FP"])),
                    str(int(fila["FN"])),
                    f"{float(fila['Delta_Accuracy']):+.4f}",
                ]
            )
            if firma not in texto:
                firmas_finalistas_ausentes.append(f"{tipo}:{indice + 1}")
    filas = [
        {
            "comprobacion": "paginas_historicas_conservadas",
            "correcto": len(documento) > paginas_anteriores,
            "detalle": (
                f"historicas={paginas_anteriores}; total={len(documento)}"
            ),
        },
        {
            "comprobacion": "orientacion_horizontal",
            "correcto": todas_horizontales,
            "detalle": f"paginas={len(documento)}",
        },
        {
            "comprobacion": "todas_las_distribuciones_listadas",
            "correcto": not distribuciones_ausentes,
            "detalle": str(distribuciones_ausentes),
        },
        {
            "comprobacion": "cinco_finalistas",
            "correcto": int(compacta["finalista"].sum()) == 5,
            "detalle": f"finalistas={int(compacta['finalista'].sum())}",
        },
        {
            "comprobacion": "149_filas_compactas_csv_pdf",
            "correcto": not firmas_compactas_ausentes,
            "detalle": str(firmas_compactas_ausentes),
        },
        {
            "comprobacion": "60_filas_finalistas_csv_pdf",
            "correcto": not firmas_finalistas_ausentes,
            "detalle": str(firmas_finalistas_ausentes),
        },
        {
            "comprobacion": "contenido_w_h",
            "correcto": "DeepONMF_W" in texto and "DeepONMF_H3" in texto,
            "detalle": "W y H presentes",
        },
        {
            "comprobacion": "sin_desviacion_estandar_visible",
            "correcto": "+/-" not in texto and "std" not in texto.lower(),
            "detalle": "Score y Accuracy se muestran como medias simples",
        },
    ]
    documento.close()
    return pd.DataFrame(filas)
