from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import kruskal

from .audio import DatosClase


def caracteristicas_por_audio(
    datos_por_clase: dict[str, DatosClase],
    h_por_clase: dict[str, np.ndarray],
) -> pd.DataFrame:
    filas: list[dict[str, object]] = []
    for clase, datos in datos_por_clase.items():
        h = h_por_clase[clase]
        for registro in datos.audios_usados:
            inicio, fin = datos.rangos_columnas[str(registro.ruta)]
            vector = h[:, inicio:fin].mean(axis=1)
            fila: dict[str, object] = {
                "clase": clase,
                "archivo": registro.ruta.name,
                "ruta": str(registro.ruta),
                "columnas_espectrograma": fin - inicio,
                "duracion_s": registro.duracion_s,
            }
            for indice, valor in enumerate(vector, start=1):
                fila[f"SBV_{indice}"] = float(valor)
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_2_desde_w(w_por_clase: dict[str, np.ndarray], clases: tuple[str, ...]) -> pd.DataFrame:
    filas: list[dict[str, object]] = []
    numero_sbv = next(iter(w_por_clase.values())).shape[1]
    for indice in range(numero_sbv):
        fila: dict[str, object] = {"Número de característica": f"SBV {indice + 1}"}
        grupos = []
        for clase in clases:
            valores = w_por_clase[clase][:, indice]
            grupos.append(valores)
            fila[clase] = f"{np.mean(valores):.6f} ± {np.std(valores, ddof=1):.6f}"
        fila["p-valor"] = float(kruskal(*grupos).pvalue)
        filas.append(fila)
    return pd.DataFrame(filas)


def distancias_figura_7(caracteristicas: pd.DataFrame, clases: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    resultado: dict[str, pd.DataFrame] = {}
    for indice in range(1, 4):
        columna = f"SBV_{indice}"
        filas_entre = []
        filas_dentro = []

        medias = {
            clase: caracteristicas.loc[caracteristicas["clase"] == clase, columna].to_numpy(dtype=float)
            for clase in clases
        }

        for clase_a, clase_b in combinations(clases, 2):
            distancia = abs(float(np.mean(medias[clase_a])) - float(np.mean(medias[clase_b])))
            filas_entre.append({"comparación": f"{clase_a}-{clase_b}", "distancia": distancia})

        for clase in clases:
            valores = medias[clase].reshape(-1, 1)
            distancia = float(np.mean(pdist(valores, metric="euclidean"))) if len(valores) > 1 else 0.0
            filas_dentro.append({"clase": clase, "distancia": distancia})

        resultado[f"SBV_{indice}_entre_clases"] = pd.DataFrame(filas_entre)
        resultado[f"SBV_{indice}_dentro_clase"] = pd.DataFrame(filas_dentro)
    return resultado


def resumen_auditoria(registros: list, datos_por_clase: dict[str, DatosClase], clases: tuple[str, ...]) -> pd.DataFrame:
    filas = []
    for clase in clases:
        registros_clase = [r for r in registros if r.clase == clase]
        usados = datos_por_clase[clase].audios_usados
        descartados = datos_por_clase[clase].audios_descartados
        duraciones = np.array([r.duracion_s for r in registros_clase], dtype=float)
        filas.append(
            {
                "clase": clase,
                "audios_totales": len(registros_clase),
                "audios_usados": len(usados),
                "audios_descartados_por_duracion": len(descartados),
                "audios_cortos_menores_2s": int(np.sum(duraciones < 2.0)),
                "duracion_min_s": float(np.min(duraciones)),
                "duracion_media_s": float(np.mean(duraciones)),
                "duracion_max_s": float(np.max(duraciones)),
                "columnas_matriz_x": int(datos_por_clase[clase].matriz.shape[1]),
            }
        )
    return pd.DataFrame(filas)


def guardar_distancias_txt(distancias: dict[str, pd.DataFrame], ruta: Path) -> None:
    partes = []
    for nombre, tabla in distancias.items():
        partes.append(nombre)
        partes.append(tabla.to_string(index=False))
        partes.append("")
    ruta.write_text("\n".join(partes), encoding="utf-8-sig")
