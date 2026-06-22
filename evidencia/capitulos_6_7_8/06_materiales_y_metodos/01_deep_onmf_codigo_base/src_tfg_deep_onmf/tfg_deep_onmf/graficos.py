from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


COLORES = {
    "N": "#1b9e77",
    "AS": "#d95f02",
    "MR": "#7570b3",
    "MS": "#e7298a",
    "MVP": "#66a61e",
}


def figura_5_sbv(w_por_clase: dict[str, np.ndarray], clases: tuple[str, ...], frecuencia_hz: int, ruta: Path) -> None:
    frecuencias = np.linspace(0, frecuencia_hz / 2, next(iter(w_por_clase.values())).shape[0])
    fig, ejes = plt.subplots(len(clases), 1, figsize=(12, 14), sharex=True)
    for eje, clase in zip(ejes, clases):
        w = w_por_clase[clase]
        for indice in range(min(5, w.shape[1])):
            eje.plot(frecuencias, w[:, indice], linewidth=1.4, label=f"SBV {indice + 1}")
        eje.set_title(f"Clase {clase}")
        eje.set_ylabel("Amplitud normalizada")
        eje.grid(True, alpha=0.25)
        eje.legend(ncol=5, fontsize=8)
    ejes[-1].set_xlabel("Frecuencia (Hz)")
    fig.suptitle("Figura 5 - Comparación de cinco SBV extraídos con deep ONMF", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def tabla_2_imagen(tabla: pd.DataFrame, ruta: Path) -> None:
    fig, eje = plt.subplots(figsize=(16, 4.8))
    eje.axis("off")
    tabla_mostrar = tabla.copy()
    tabla_mostrar["p-valor"] = tabla_mostrar["p-valor"].map(lambda x: f"{x:.3e}")
    objeto_tabla = eje.table(
        cellText=tabla_mostrar.values,
        colLabels=tabla_mostrar.columns,
        cellLoc="center",
        loc="center",
    )
    objeto_tabla.auto_set_font_size(False)
    objeto_tabla.set_fontsize(8)
    objeto_tabla.scale(1, 1.8)
    eje.set_title("Tabla 2 - Estadística de SBV 1 a SBV 7 por clase", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def figura_7_distancias(distancias: dict[str, pd.DataFrame], ruta: Path) -> None:
    fig, ejes = plt.subplots(3, 2, figsize=(16, 12))
    for fila, indice in enumerate(range(1, 4)):
        entre = distancias[f"SBV_{indice}_entre_clases"]
        dentro = distancias[f"SBV_{indice}_dentro_clase"]

        eje_entre = ejes[fila, 0]
        eje_dentro = ejes[fila, 1]

        eje_entre.bar(entre["comparación"], entre["distancia"], color="#4c78a8")
        eje_entre.set_title(f"SBV {indice}: distancia entre clases")
        eje_entre.set_ylabel("Distancia euclídea")
        eje_entre.tick_params(axis="x", rotation=45)
        eje_entre.grid(True, axis="y", alpha=0.25)

        colores = [COLORES.get(clase, "#666666") for clase in dentro["clase"]]
        eje_dentro.bar(dentro["clase"], dentro["distancia"], color=colores)
        eje_dentro.set_title(f"SBV {indice}: distancia dentro de cada clase")
        eje_dentro.set_ylabel("Distancia euclídea media")
        eje_dentro.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Figura 7 - Distancias euclídeas entre clases y dentro de clase", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def figura_11d_tsne(caracteristicas: pd.DataFrame, clases: tuple[str, ...], ruta_png: Path, ruta_csv: Path) -> None:
    columnas = [f"SBV_{i}" for i in range(1, 8)]
    x = caracteristicas[columnas].to_numpy(dtype=float)
    x = StandardScaler().fit_transform(x)
    perplejidad = min(30, max(5, (len(x) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplejidad,
        init="pca",
        learning_rate="auto",
        random_state=42,
        max_iter=1000,
    )
    coordenadas = tsne.fit_transform(x)
    salida = caracteristicas[["clase", "archivo"]].copy()
    salida["tSNE_1"] = coordenadas[:, 0]
    salida["tSNE_2"] = coordenadas[:, 1]
    salida.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

    fig, eje = plt.subplots(figsize=(9, 7))
    for clase in clases:
        mascara = salida["clase"] == clase
        eje.scatter(
            salida.loc[mascara, "tSNE_1"],
            salida.loc[mascara, "tSNE_2"],
            s=22,
            alpha=0.8,
            color=COLORES.get(clase, "#666666"),
            label=clase,
            edgecolors="none",
        )
    eje.set_title("Figura 11D - t-SNE con características deep ONMF")
    eje.set_xlabel("t-SNE 1")
    eje.set_ylabel("t-SNE 2")
    eje.grid(True, alpha=0.2)
    eje.legend(title="Clase")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=220)
    plt.close(fig)
