from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .configuracion_pruebas import CLASIFICADORES, DISTRIBUCIONES, REPRESENTACIONES_DEEP
from .evaluacion import COLUMNAS_RESUMEN


REPRESENTACIONES_CLASICAS = ("STFT", "MFCC", "MelSpectrogram", "LogMelSpectrogram")
ORDEN_REPRESENTACIONES = {
    nombre: indice
    for indice, nombre in enumerate((*REPRESENTACIONES_CLASICAS, *REPRESENTACIONES_DEEP))
}
ORDEN_CLASIFICADORES = {nombre: indice for indice, nombre in enumerate(CLASIFICADORES)}
ORDEN_DISTRIBUCIONES = {
    "No aplica": 0,
    "9-8-7": 1,
    "15-10-5": 2,
    "10-6-4": 3,
    "8-5-3": 4,
}


def _ordenar(tabla: pd.DataFrame) -> pd.DataFrame:
    resultado = tabla.copy()
    resultado["_orden_clasificador"] = resultado["clasificador"].map(ORDEN_CLASIFICADORES)
    resultado["_orden_representacion"] = resultado["representacion"].map(ORDEN_REPRESENTACIONES)
    resultado["_orden_distribucion"] = resultado["distribucion"].map(ORDEN_DISTRIBUCIONES)
    return (
        resultado.sort_values(
            ["_orden_clasificador", "_orden_representacion", "_orden_distribucion"]
        )
        .drop(columns=["_orden_clasificador", "_orden_representacion", "_orden_distribucion"])
        .reset_index(drop=True)
    )


def _anadir_distribucion_original(tabla: pd.DataFrame) -> pd.DataFrame:
    resultado = tabla.copy()
    resultado.insert(
        0,
        "distribucion",
        resultado["representacion"].map(
            lambda nombre: "9-8-7" if nombre in REPRESENTACIONES_DEEP else "No aplica"
        ),
    )
    return resultado


def _cargar_nuevas(
    raiz_resultados: Path,
    nombre_archivo: str,
) -> pd.DataFrame:
    tablas = []
    for clave in DISTRIBUCIONES:
        ruta = raiz_resultados / f"distribucion_{clave}" / "metricas" / nombre_archivo
        if not ruta.exists():
            raise FileNotFoundError(f"Falta el resultado necesario: {ruta}")
        tablas.append(pd.read_csv(ruta, encoding="utf-8-sig"))
    return pd.concat(tablas, ignore_index=True)


def consolidar_tablas(
    raiz_resultados: Path,
    resultados_originales: Path,
) -> dict[str, pd.DataFrame]:
    carpeta = raiz_resultados / "tablas_csv"
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas_originales = resultados_originales / "metricas"

    original_bin = pd.read_csv(
        rutas_originales / "resumen_metricas_binarias.csv",
        encoding="utf-8-sig",
    )
    original_multi = pd.read_csv(
        rutas_originales / "resumen_metricas_multiclase.csv",
        encoding="utf-8-sig",
    )
    original_fold_bin = pd.read_csv(
        rutas_originales / "metricas_binarias_por_fold.csv",
        encoding="utf-8-sig",
    )
    original_fold_multi = pd.read_csv(
        rutas_originales / "metricas_multiclase_por_fold.csv",
        encoding="utf-8-sig",
    )

    nuevas_bin = _cargar_nuevas(raiz_resultados, "resumen_metricas_binarias.csv")
    nuevas_multi = _cargar_nuevas(raiz_resultados, "resumen_metricas_multiclase.csv")
    nuevas_fold_bin = _cargar_nuevas(raiz_resultados, "metricas_binarias_por_fold.csv")
    nuevas_fold_multi = _cargar_nuevas(raiz_resultados, "metricas_multiclase_por_fold.csv")

    original_bin_dist = _anadir_distribucion_original(original_bin)
    original_multi_dist = _anadir_distribucion_original(original_multi)
    global_bin = _ordenar(pd.concat([original_bin_dist, nuevas_bin], ignore_index=True))
    global_multi = _ordenar(pd.concat([original_multi_dist, nuevas_multi], ignore_index=True))

    deep_original_bin = _anadir_distribucion_original(
        original_fold_bin[original_fold_bin["representacion"].isin(REPRESENTACIONES_DEEP)]
    )
    deep_original_multi = _anadir_distribucion_original(
        original_fold_multi[original_fold_multi["representacion"].isin(REPRESENTACIONES_DEEP)]
    )
    folds_deep_bin = pd.concat([deep_original_bin, nuevas_fold_bin], ignore_index=True)
    folds_deep_multi = pd.concat([deep_original_multi, nuevas_fold_multi], ignore_index=True)
    folds_deep_bin = _ordenar(folds_deep_bin)
    folds_deep_multi = _ordenar(folds_deep_multi)

    comparacion_bin = global_bin[global_bin["representacion"].isin(REPRESENTACIONES_DEEP)].copy()
    comparacion_multi = global_multi[global_multi["representacion"].isin(REPRESENTACIONES_DEEP)].copy()
    diferencias_bin = calcular_diferencias(comparacion_bin)
    diferencias_multi = calcular_diferencias(comparacion_multi)

    tablas: dict[str, pd.DataFrame] = {
        "tabla_original_binaria_seis_representaciones": _ordenar(original_bin_dist),
        "tabla_original_multiclase_seis_representaciones": _ordenar(original_multi_dist),
        "tabla_global_binaria_completa_36_filas": global_bin,
        "tabla_global_multiclase_completa_36_filas": global_multi,
        "comparacion_distribuciones_binaria": comparacion_bin,
        "comparacion_distribuciones_multiclase": comparacion_multi,
        "tp_tn_fp_fn_por_fold_deeponmf": folds_deep_bin,
        "metricas_multiclase_por_fold_deeponmf": folds_deep_multi,
        "diferencias_respecto_9_8_7_binaria": diferencias_bin,
        "diferencias_respecto_9_8_7_multiclase": diferencias_multi,
    }
    for clasificador in CLASIFICADORES:
        tablas[f"tabla_{clasificador}_binaria"] = global_bin[
            global_bin["clasificador"] == clasificador
        ].reset_index(drop=True)
        tablas[f"tabla_{clasificador}_multiclase"] = global_multi[
            global_multi["clasificador"] == clasificador
        ].reset_index(drop=True)

    for nombre, tabla in tablas.items():
        tabla.to_csv(carpeta / f"{nombre}.csv", index=False, encoding="utf-8-sig")

    manifiesto = crear_manifiesto_csv(carpeta)
    tablas["manifiesto_csv"] = manifiesto
    return tablas


def calcular_diferencias(comparacion: pd.DataFrame) -> pd.DataFrame:
    referencia = comparacion[comparacion["distribucion"] == "9-8-7"].copy()
    referencia = referencia.set_index(["clasificador", "representacion"])
    filas = []
    for _, fila in comparacion[comparacion["distribucion"] != "9-8-7"].iterrows():
        clave = (fila["clasificador"], fila["representacion"])
        fila_ref = referencia.loc[clave]
        nueva: dict[str, object] = {
            "distribucion": fila["distribucion"],
            "clasificador": fila["clasificador"],
            "representacion": fila["representacion"],
        }
        for metrica in COLUMNAS_RESUMEN:
            columna = f"{metrica}_mean"
            nueva[f"{metrica}_nuevo"] = float(fila[columna])
            nueva[f"{metrica}_referencia_9_8_7"] = float(fila_ref[columna])
            nueva[f"delta_{metrica}"] = float(fila[columna] - fila_ref[columna])
        filas.append(nueva)
    return _ordenar(pd.DataFrame(filas))


def crear_manifiesto_csv(carpeta: Path) -> pd.DataFrame:
    filas = []
    for ruta in sorted(carpeta.glob("*.csv")):
        if ruta.name == "manifiesto_csv.csv" or ruta.name.startswith("auditoria_"):
            continue
        contenido = ruta.read_bytes()
        tabla = pd.read_csv(ruta, encoding="utf-8-sig")
        filas.append(
            {
                "archivo": ruta.name,
                "filas": len(tabla),
                "columnas": len(tabla.columns),
                "sha256": hashlib.sha256(contenido).hexdigest(),
            }
        )
    manifiesto = pd.DataFrame(filas)
    manifiesto.to_csv(carpeta / "manifiesto_csv.csv", index=False, encoding="utf-8-sig")
    return manifiesto
