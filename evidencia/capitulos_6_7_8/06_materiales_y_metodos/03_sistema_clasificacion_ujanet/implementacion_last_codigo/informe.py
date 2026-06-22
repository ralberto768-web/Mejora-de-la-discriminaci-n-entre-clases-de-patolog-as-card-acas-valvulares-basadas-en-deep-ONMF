from __future__ import annotations

from pathlib import Path
import textwrap

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .configuracion import ConfiguracionExperimento


def _tabla_markdown(df: pd.DataFrame, max_filas: int = 100) -> str:
    tabla = df.head(max_filas).copy()
    for columna in tabla.columns:
        if pd.api.types.is_float_dtype(tabla[columna]):
            tabla[columna] = tabla[columna].map(lambda valor: f"{valor:.4f}")
    cabecera = "| " + " | ".join(map(str, tabla.columns)) + " |"
    separador = "| " + " | ".join(["---"] * len(tabla.columns)) + " |"
    filas = [
        "| " + " | ".join(map(str, fila)) + " |"
        for fila in tabla.itertuples(index=False, name=None)
    ]
    return "\n".join([cabecera, separador, *filas])


def crear_figuras_resumen(resumen_binario: pd.DataFrame, carpeta: Path) -> list[Path]:
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas: list[Path] = []
    for clasificador, grupo in resumen_binario.groupby("clasificador"):
        datos = grupo.sort_values("Score_mean", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.barh(
            datos["representacion"],
            datos["Score_mean"],
            xerr=datos["Score_std"].fillna(0),
            color="#3f6f8f",
        )
        ax.set_title(f"Score binario por representación - {clasificador}")
        ax.set_xlabel("Score = (Sensitivity + Specificity) / 2")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.35)
        ax.invert_yaxis()
        for borde in ax.spines.values():
            borde.set_visible(False)
        ruta = carpeta / f"score_binario_{clasificador}.png"
        fig.tight_layout()
        fig.savefig(ruta, dpi=220, facecolor="white")
        plt.close(fig)
        rutas.append(ruta)
    return rutas


def redactar_informe_markdown(
    config: ConfiguracionExperimento,
    auditoria: pd.DataFrame,
    resumen_binario: pd.DataFrame,
    resumen_multiclase: pd.DataFrame,
    carpeta: Path,
) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    mejores = (
        resumen_binario.sort_values("Score_mean", ascending=False)
        .groupby("clasificador", as_index=False)
        .first()[
            [
                "clasificador",
                "representacion",
                "Score_mean",
                "Sensitivity_mean",
                "Specificity_mean",
            ]
        ]
    )
    lineas = [
        "# Informe punto 3: validación automática de representaciones",
        "",
        "## 1. Resumen",
        "",
        "Este documento cubre el punto 3 solicitado: comparar seis representaciones "
        "mediante SVM, KNN y UjaNet. La evaluación principal distingue sonidos "
        "normales frente a anómalos y se añade una evaluación multiclase secundaria.",
        "",
        "## 2. Protocolo",
        "",
        f"Se utilizan {config.folds} folds estratificados. En la ejecución completa "
        "cada fold contiene 800 señales de entrenamiento y 200 de prueba, con 40 "
        "señales de cada clase en test.",
        "",
        f"Deep-ONMF usa rangos {config.rangos_deep_onmf}, "
        f"{config.iteraciones_onmf} iteraciones por capa, penalización ortogonal "
        f"{config.penalizacion_ortogonal} y semilla {config.semilla}.",
        "",
        "## 3. Auditoría de la base de datos",
        "",
        _tabla_markdown(auditoria),
        "",
        "## 4. Representaciones",
        "",
        "- `STFT`: espectrograma de magnitud normalizado.",
        "- `MFCC`: 13 coeficientes cepstrales sobre banco Mel.",
        "- `MelSpectrogram`: energía proyectada sobre filtros Mel.",
        "- `LogMelSpectrogram`: versión logarítmica del Mel-Spectrogram.",
        "- `DeepONMF_W`: matriz espectral final de Deep-ONMF.",
        "- `DeepONMF_H3`: activaciones temporales de la tercera capa.",
        "",
        "## 5. Clasificadores y métricas",
        "",
        "SVM utiliza kernel RBF y pesos balanceados. KNN utiliza cinco vecinos "
        "ponderados por distancia. UjaNet reproduce la arquitectura convolucional "
        "del artículo. Se calculan TP, TN, FP, FN, Accuracy, Sensitivity, "
        "Specificity, Precision y Score.",
        "",
        "## 6. Resultados binarios",
        "",
        _tabla_markdown(resumen_binario.sort_values("Score_mean", ascending=False)),
        "",
    ]
    for clasificador in ("SVM", "KNN", "UjaNet"):
        tabla = resumen_binario.loc[
            resumen_binario["clasificador"] == clasificador
        ].sort_values("Score_mean", ascending=False)
        lineas.extend(
            [
                f"### {clasificador}",
                "",
                _tabla_markdown(tabla),
                "",
            ]
        )
    lineas.extend(
        [
            "## 7. Mejor representación por clasificador",
            "",
            _tabla_markdown(mejores),
            "",
            "## 8. Resultados multiclase",
            "",
            _tabla_markdown(
                resumen_multiclase.sort_values("Score_mean", ascending=False)
            ),
            "",
            "## 9. Interpretación",
            "",
            "La comparación central es DeepONMF_H3 frente a DeepONMF_W. H3 representa "
            "la evolución temporal de los componentes y W sus patrones espectrales. "
            "Las conclusiones deben leerse por clasificador y apoyarse en las cinco "
            "particiones, las matrices de confusión y las predicciones guardadas.",
        ]
    )
    ruta = carpeta / "informe_punto3_validacion.md"
    ruta.write_text("\n".join(lineas), encoding="utf-8")
    return ruta


class PDFSimple:
    def __init__(self, ruta: Path) -> None:
        self.ruta = ruta
        self.doc = fitz.open()
        self.pagina = self.doc.new_page(width=595, height=842)
        self.y = 52.0
        self.pagina.insert_text(
            (42, self.y),
            "Punto 3: validación automática de representaciones",
            fontsize=18,
        )
        self.y += 40

    def _nueva_pagina(self) -> None:
        self.pagina = self.doc.new_page(width=595, height=842)
        self.y = 45.0

    def _asegurar(self, alto: float) -> None:
        if self.y + alto > 800:
            self._nueva_pagina()

    def encabezado(self, texto: str) -> None:
        self._asegurar(30)
        self.pagina.insert_text((42, self.y), texto, fontsize=13)
        self.y += 24

    def texto(self, texto: str) -> None:
        lineas = textwrap.wrap(texto, width=92, break_long_words=False)
        self._asegurar(len(lineas) * 13 + 8)
        for linea in lineas:
            self.pagina.insert_text((42, self.y), linea, fontsize=9.5)
            self.y += 12.5
        self.y += 6

    def tabla(self, df: pd.DataFrame, titulo: str, max_filas: int) -> None:
        tabla = df.head(max_filas).copy()
        for columna in tabla.columns:
            if pd.api.types.is_float_dtype(tabla[columna]):
                tabla[columna] = tabla[columna].map(lambda valor: f"{valor:.3f}")
        lineas = tabla.to_string(index=False).splitlines()
        self._asegurar(35 + len(lineas) * 9)
        self.encabezado(titulo)
        for linea in lineas:
            self.pagina.insert_text((42, self.y), linea[:122], fontsize=7, fontname="cour")
            self.y += 9
        self.y += 8

    def imagen(self, ruta: Path) -> None:
        pix = fitz.Pixmap(str(ruta))
        ancho = 500
        alto = min(350, ancho * pix.height / max(1, pix.width))
        self._asegurar(alto + 20)
        self.pagina.insert_image(
            fitz.Rect(42, self.y, 42 + ancho, self.y + alto),
            filename=str(ruta),
        )
        self.y += alto + 14

    def guardar(self) -> None:
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(self.ruta, deflate=True, garbage=4)
        self.doc.close()


def crear_pdf(
    auditoria: pd.DataFrame,
    resumen_binario: pd.DataFrame,
    resumen_multiclase: pd.DataFrame,
    figuras: list[Path],
    carpeta: Path,
) -> Path:
    ruta = carpeta / "informe_punto3_validacion.pdf"
    pdf = PDFSimple(ruta)
    pdf.encabezado("Protocolo experimental")
    pdf.texto(
        "La base Yaseen se evalúa mediante cinco folds estratificados. La tarea "
        "principal distingue normal frente a anómalo y la clase positiva es anómalo."
    )
    pdf.tabla(auditoria, "Auditoría de la base de datos", 10)
    pdf.encabezado("Representaciones y clasificadores")
    pdf.texto(
        "Se comparan STFT, MFCC, Mel-Spectrogram, Log-Mel Spectrogram, DeepONMF_W "
        "y DeepONMF_H3 mediante SVM, KNN y UjaNet."
    )
    pdf.tabla(
        resumen_binario.sort_values("Score_mean", ascending=False),
        "Resumen binario",
        18,
    )
    for figura in figuras:
        pdf.imagen(figura)
    pdf.tabla(
        resumen_multiclase.sort_values("Score_mean", ascending=False),
        "Resumen multiclase",
        18,
    )
    pdf.guardar()
    return ruta


def generar_informe_final(
    config: ConfiguracionExperimento,
    auditoria: pd.DataFrame,
    resumen_binario: pd.DataFrame,
    resumen_multiclase: pd.DataFrame,
    carpeta: Path,
) -> tuple[Path, Path, list[Path]]:
    figuras = crear_figuras_resumen(resumen_binario, carpeta / "figuras")
    markdown = redactar_informe_markdown(
        config,
        auditoria,
        resumen_binario,
        resumen_multiclase,
        carpeta,
    )
    pdf = crear_pdf(
        auditoria,
        resumen_binario,
        resumen_multiclase,
        figuras,
        carpeta,
    )
    return markdown, pdf, figuras
