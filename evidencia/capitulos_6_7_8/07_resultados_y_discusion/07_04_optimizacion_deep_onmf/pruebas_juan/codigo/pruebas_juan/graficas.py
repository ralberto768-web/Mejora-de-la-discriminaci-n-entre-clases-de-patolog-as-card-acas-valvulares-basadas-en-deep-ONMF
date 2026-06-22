from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .configuracion_pruebas import CLASIFICADORES


DISTRIBUCIONES = ("9-8-7", "15-10-5", "10-6-4", "8-5-3")
COLOR_W = "#356A9A"
COLOR_H3 = "#D37532"


def _valores(
    datos: pd.DataFrame,
    representacion: str,
) -> list[float]:
    indice = datos.set_index(["distribucion", "representacion"])
    return [
        float(indice.loc[(distribucion, representacion), "Score_mean"])
        for distribucion in DISTRIBUCIONES
    ]


def _limites(valores: list[float]) -> tuple[float, float]:
    minimo = min(valores)
    inferior = max(0.0, np.floor((minimo - 0.025) * 20.0) / 20.0)
    return inferior, 1.01


def _etiquetar(eje, barras) -> None:
    eje.bar_label(barras, fmt="%.4f", fontsize=8, padding=3)


def generar_graficas(
    comparacion: pd.DataFrame,
    carpeta: Path,
    tipo_evaluacion: str,
) -> list[Path]:
    """Genera una figura de tres paneles para cada clasificador.

    Los tres paneles muestran exclusivamente el Score medio de la evaluación
    indicada: distribuciones W entre sí, H3 entre sí y W frente a H3.
    distribuciones W entre sí, distribuciones H3 entre sí y W frente a H3.
    """

    carpeta.mkdir(parents=True, exist_ok=True)
    rutas: list[Path] = []
    x = np.arange(len(DISTRIBUCIONES))

    for clasificador in CLASIFICADORES:
        datos = comparacion[comparacion["clasificador"] == clasificador]
        valores_w = _valores(datos, "DeepONMF_W")
        valores_h3 = _valores(datos, "DeepONMF_H3")
        inferior, superior = _limites(valores_w + valores_h3)

        figura, ejes = plt.subplots(1, 3, figsize=(15.2, 4.7), sharey=True)
        figura.patch.set_facecolor("white")

        barras_w = ejes[0].bar(x, valores_w, width=0.62, color=COLOR_W)
        ejes[0].set_title("Comparación de las matrices W")
        _etiquetar(ejes[0], barras_w)

        barras_h3 = ejes[1].bar(x, valores_h3, width=0.62, color=COLOR_H3)
        ejes[1].set_title("Comparación de las matrices H3")
        _etiquetar(ejes[1], barras_h3)

        ancho = 0.36
        barras_pareadas_w = ejes[2].bar(
            x - ancho / 2,
            valores_w,
            width=ancho,
            color=COLOR_W,
            label="W",
        )
        barras_pareadas_h3 = ejes[2].bar(
            x + ancho / 2,
            valores_h3,
            width=ancho,
            color=COLOR_H3,
            label="H3",
        )
        ejes[2].set_title("Comparación W frente a H3")
        ejes[2].legend(loc="lower right", frameon=True)
        _etiquetar(ejes[2], barras_pareadas_w)
        _etiquetar(ejes[2], barras_pareadas_h3)

        for eje in ejes:
            eje.set_xticks(x, DISTRIBUCIONES)
            eje.set_ylim(inferior, superior)
            eje.grid(axis="y", alpha=0.24)
            eje.set_axisbelow(True)
            eje.spines["top"].set_visible(False)
            eje.spines["right"].set_visible(False)
            eje.tick_params(axis="x", labelsize=9)
            eje.tick_params(axis="y", labelsize=9)

        ejes[0].set_ylabel(f"Score {tipo_evaluacion} medio (5-Fold)")
        figura.suptitle(
            f"{clasificador}: comparación Deep-ONMF {tipo_evaluacion}",
            fontsize=15,
            fontweight="bold",
        )
        figura.tight_layout(rect=(0.01, 0.01, 0.99, 0.94), w_pad=1.6)
        ruta = carpeta / f"comparacion_w_h3_{tipo_evaluacion}_{clasificador}.png"
        figura.savefig(ruta, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figura)
        rutas.append(ruta)

    return rutas
