from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .configuraciones import CLASES, CLASIFICADORES, convertir_etiqueta


def _agrupar_predicciones(carpeta: Path) -> pd.DataFrame:
    archivos = []
    for fold in range(1, 6):
        corta = carpeta / f"f{fold}_pred.csv"
        larga = carpeta / f"fold_{fold}_predicciones.csv"
        archivos.append(corta if corta.exists() else larga)
    faltantes = [ruta for ruta in archivos if not ruta.exists()]
    if faltantes:
        raise FileNotFoundError(
            "Faltan predicciones: " + ", ".join(str(ruta) for ruta in faltantes)
        )
    tabla = pd.concat(
        [pd.read_csv(ruta, encoding="utf-8-sig") for ruta in archivos],
        ignore_index=True,
    )
    if len(tabla) != 1000:
        raise AssertionError(
            f"{carpeta}: se esperaban 1000 predicciones, hay {len(tabla)}"
        )
    clave = tabla["clase"].astype(str) + "/" + tabla["archivo"].astype(str)
    if clave.nunique() != 1000:
        raise AssertionError(f"{carpeta}: hay audios repetidos o ausentes")
    conteos = tabla["clase"].value_counts().to_dict()
    if conteos != {clase: 200 for clase in CLASES}:
        raise AssertionError(f"{carpeta}: soportes incorrectos {conteos}")
    return tabla


def _matriz(tabla: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conteos = pd.crosstab(
        pd.Categorical(tabla["clase"], categories=CLASES),
        pd.Categorical(tabla["pred_multiclase"], categories=CLASES),
        dropna=False,
    )
    conteos.index = list(CLASES)
    conteos.columns = list(CLASES)
    porcentajes = conteos.div(conteos.sum(axis=1), axis=0) * 100.0
    return conteos, porcentajes


def _guardar_png(
    conteos: pd.DataFrame,
    porcentajes: pd.DataFrame,
    titulo: str,
    ruta: Path,
) -> None:
    figura, eje = plt.subplots(figsize=(7.0, 5.6))
    imagen = eje.imshow(porcentajes.to_numpy(), cmap="Blues", vmin=0, vmax=100)
    eje.set_xticks(range(len(CLASES)), CLASES)
    eje.set_yticks(range(len(CLASES)), CLASES)
    eje.set_xlabel("Clase predicha")
    eje.set_ylabel("Clase real")
    eje.set_title(titulo, fontsize=11, fontweight="bold")
    for fila in range(len(CLASES)):
        for columna in range(len(CLASES)):
            porcentaje = float(porcentajes.iloc[fila, columna])
            color = "white" if porcentaje >= 55 else "black"
            eje.text(
                columna,
                fila,
                f"{int(conteos.iloc[fila, columna])}\n{porcentaje:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    barra = figura.colorbar(imagen, ax=eje, fraction=0.046, pad=0.04)
    barra.set_label("% dentro de la clase real")
    figura.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=170, bbox_inches="tight")
    plt.close(figura)


def generar_matrices_confusion(
    seleccion: pd.DataFrame,
    resolver_predicciones: Callable[[str, str, str], Path],
    carpeta_salida: Path,
    clasificadores: tuple[str, ...] = CLASIFICADORES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filas: list[dict[str, object]] = []
    manifiesto: list[dict[str, object]] = []
    for _, pareja in seleccion.iterrows():
        distribuciones = (
            ("decreciente", str(pareja["distribucion"])),
            ("creciente", str(pareja["distribucion_invertida"])),
        )
        for sentido, distribucion in distribuciones:
            numero_capas = len(convertir_etiqueta(distribucion))
            representacion = f"DeepONMF_H{numero_capas}"
            for clasificador in clasificadores:
                origen = resolver_predicciones(
                    distribucion,
                    clasificador,
                    representacion,
                )
                predicciones = _agrupar_predicciones(origen)
                conteos, porcentajes = _matriz(predicciones)
                carpeta = (
                    carpeta_salida
                    / f"pareja_{int(pareja['posicion_seleccion']):02d}"
                    / clasificador
                    / sentido
                )
                carpeta.mkdir(parents=True, exist_ok=True)
                ruta_conteos = carpeta / "matriz_confusion_conteos.csv"
                ruta_porcentajes = carpeta / "matriz_confusion_porcentajes.csv"
                ruta_predicciones = carpeta / "predicciones_1000.csv"
                ruta_png = carpeta / "matriz_confusion.png"
                conteos.to_csv(ruta_conteos, encoding="utf-8-sig")
                porcentajes.to_csv(ruta_porcentajes, encoding="utf-8-sig")
                predicciones.to_csv(
                    ruta_predicciones,
                    index=False,
                    encoding="utf-8-sig",
                )
                _guardar_png(
                    conteos,
                    porcentajes,
                    f"{distribucion} - H{numero_capas} - {clasificador}",
                    ruta_png,
                )
                diagonal = int(np.trace(conteos.to_numpy()))
                errores = conteos.to_numpy(copy=True)
                np.fill_diagonal(errores, 0)
                fila_error, columna_error = np.unravel_index(
                    int(np.argmax(errores)),
                    errores.shape,
                )
                mayor_error = int(errores[fila_error, columna_error])
                filas.append(
                    {
                        "pareja": int(pareja["posicion_seleccion"]),
                        "sentido": sentido,
                        "distribucion": distribucion,
                        "numero_capas": numero_capas,
                        "clasificador": clasificador,
                        "representacion": representacion,
                        "predicciones": len(predicciones),
                        "aciertos": diagonal,
                        "accuracy_directa": diagonal / len(predicciones),
                        "mayor_error_real": CLASES[fila_error],
                        "mayor_error_predicha": CLASES[columna_error],
                        "mayor_error_cantidad": mayor_error,
                        "ruta_png": str(ruta_png.resolve()),
                    }
                )
                manifiesto.append(
                    {
                        "pareja": int(pareja["posicion_seleccion"]),
                        "sentido": sentido,
                        "distribucion": distribucion,
                        "clasificador": clasificador,
                        "origen_predicciones": str(origen.resolve()),
                        "csv_conteos": str(ruta_conteos.resolve()),
                        "csv_porcentajes": str(ruta_porcentajes.resolve()),
                        "png": str(ruta_png.resolve()),
                    }
                )
    resumen = pd.DataFrame(filas)
    tabla_manifiesto = pd.DataFrame(manifiesto)
    esperadas = len(seleccion) * 2 * len(clasificadores)
    if len(resumen) != esperadas:
        raise AssertionError(
            f"Se esperaban {esperadas} matrices, hay {len(resumen)}"
        )
    if not resumen["predicciones"].eq(1000).all():
        raise AssertionError("Alguna matriz no contiene 1000 predicciones")
    resumen.to_csv(
        carpeta_salida / "resumen_matrices_confusion.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tabla_manifiesto.to_csv(
        carpeta_salida / "manifiesto_matrices_confusion.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return resumen, tabla_manifiesto
