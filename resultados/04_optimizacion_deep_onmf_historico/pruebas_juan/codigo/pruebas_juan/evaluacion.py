from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from codigo.clasificadores import (
    _entrenar_ujanet,
    _pipeline_knn,
    _pipeline_svm,
    etiquetas_numericas,
)
from codigo.configuracion import CLASES, ConfiguracionExperimento
from codigo.metricas import matriz_confusion_binaria, metricas_binarias, metricas_multiclase
from codigo.representaciones import vectorizar_para_clasicos

from .configuracion_pruebas import CLASIFICADORES, REPRESENTACIONES_DEEP, etiqueta_distribucion


COLUMNAS_RESUMEN = ("Accuracy", "Sensitivity", "Specificity", "Precision", "Score")


def adaptar_para_ujanet(x: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Rellena con ceros únicamente cuando una dimensión espacial es menor que 4.

    UjaNet contiene dos MaxPool2D consecutivos. Una matriz con rango final 3
    necesita una fila o columna nula adicional para que ambas operaciones sean
    matemáticamente posibles. El relleno no duplica ni inventa características.
    """

    filas, columnas = x.shape[1], x.shape[2]
    filas_objetivo = max(4, filas)
    columnas_objetivo = max(4, columnas)
    if (filas, columnas) == (filas_objetivo, columnas_objetivo):
        return x.astype(np.float32, copy=False), {
            "padding_aplicado": False,
            "forma_original": f"{filas}x{columnas}",
            "forma_ujanet": f"{filas}x{columnas}",
            "filas_cero_anadidas": 0,
            "columnas_cero_anadidas": 0,
        }
    adaptada = np.pad(
        x,
        ((0, 0), (0, filas_objetivo - filas), (0, columnas_objetivo - columnas)),
        mode="constant",
    )
    return adaptada.astype(np.float32), {
        "padding_aplicado": True,
        "forma_original": f"{filas}x{columnas}",
        "forma_ujanet": f"{filas_objetivo}x{columnas_objetivo}",
        "filas_cero_anadidas": filas_objetivo - filas,
        "columnas_cero_anadidas": columnas_objetivo - columnas,
    }


def _leer_parcial(ruta: Path) -> list[dict[str, object]]:
    if not ruta.exists():
        return []
    return pd.read_csv(ruta, encoding="utf-8-sig").to_dict(orient="records")


def _guardar_parcial(filas: list[dict[str, object]], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8-sig")


def _clave(fila: dict[str, object]) -> tuple[str, str, int]:
    return str(fila["clasificador"]), str(fila["representacion"]), int(fila["fold"])


def _reemplazar_fila(
    filas: list[dict[str, object]],
    nueva: dict[str, object],
) -> None:
    clave_nueva = _clave(nueva)
    filas[:] = [fila for fila in filas if _clave(fila) != clave_nueva]
    filas.append(nueva)


def _guardar_resultados_fold(
    carpeta_repr_bin: Path,
    carpeta_repr_multi: Path,
    fold: int,
    metadatos: pd.DataFrame,
    idx_test: np.ndarray,
    y_bin_test: np.ndarray,
    y_multi_test: np.ndarray,
    pred_bin: np.ndarray,
    pred_multi: np.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    carpeta_repr_bin.mkdir(parents=True, exist_ok=True)
    carpeta_repr_multi.mkdir(parents=True, exist_ok=True)
    predicciones = metadatos.iloc[idx_test][["clase", "etiqueta_binaria", "archivo", "ruta"]].copy()
    predicciones["fold"] = fold
    predicciones["pred_binaria"] = np.where(pred_bin == 1, "anomalo", "normal")
    predicciones["pred_multiclase"] = [CLASES[int(i)] for i in pred_multi]
    predicciones.to_csv(
        carpeta_repr_bin / f"fold_{fold}_predicciones.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predicciones.to_csv(
        carpeta_repr_multi / f"fold_{fold}_predicciones.csv",
        index=False,
        encoding="utf-8-sig",
    )
    matriz_confusion_binaria(y_bin_test, pred_bin).to_csv(
        carpeta_repr_bin / f"fold_{fold}_matriz_confusion_binaria.csv",
        encoding="utf-8-sig",
    )
    matriz_multi, tabla_multi = metricas_multiclase(y_multi_test, pred_multi)
    matriz_multi.to_csv(
        carpeta_repr_multi / f"fold_{fold}_matriz_confusion_multiclase.csv",
        encoding="utf-8-sig",
    )
    tabla_multi.to_csv(
        carpeta_repr_multi / f"fold_{fold}_metricas_multiclase.csv",
        index=False,
        encoding="utf-8-sig",
    )
    macro = tabla_multi.loc[tabla_multi["clase"] == "PROMEDIO_MACRO"].iloc[0].to_dict()
    return metricas_binarias(y_bin_test, pred_bin), macro


def evaluar_distribucion(
    matrices: dict[str, np.ndarray],
    metadatos: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: ConfiguracionExperimento,
    carpeta: Path,
    reutilizar: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evalúa W y H3 con SVM, KNN y UjaNet, binaria y multiclase."""

    carpeta_metricas = carpeta / "metricas"
    ruta_bin = carpeta_metricas / "metricas_binarias_por_fold.csv"
    ruta_multi = carpeta_metricas / "metricas_multiclase_por_fold.csv"
    filas_bin = _leer_parcial(ruta_bin) if reutilizar else []
    filas_multi = _leer_parcial(ruta_multi) if reutilizar else []
    completadas_bin = {_clave(fila) for fila in filas_bin}
    completadas_multi = {_clave(fila) for fila in filas_multi}
    y_bin, y_multi = etiquetas_numericas(metadatos)
    distribucion = etiqueta_distribucion(config.rangos_deep_onmf)
    auditoria_padding: list[dict[str, object]] = []

    for representacion in REPRESENTACIONES_DEEP:
        x_matriz = matrices[representacion].astype(np.float32)
        x_vector = vectorizar_para_clasicos(x_matriz)
        x_ujanet, datos_padding = adaptar_para_ujanet(x_matriz)
        auditoria_padding.append(
            {
                "distribucion": distribucion,
                "representacion": representacion,
                **datos_padding,
            }
        )

        for clasificador in CLASIFICADORES:
            for numero_fold, (idx_train, idx_test) in enumerate(folds, start=1):
                clave = (clasificador, representacion, numero_fold)
                if clave in completadas_bin and clave in completadas_multi:
                    print(f"[{distribucion}] Se reutiliza {clasificador} {representacion} fold {numero_fold}")
                    continue

                if clasificador == "SVM":
                    modelo_bin = _pipeline_svm(config, len(idx_train), x_vector.shape[1])
                    modelo_multi = _pipeline_svm(config, len(idx_train), x_vector.shape[1])
                    modelo_bin.fit(x_vector[idx_train], y_bin[idx_train])
                    modelo_multi.fit(x_vector[idx_train], y_multi[idx_train])
                    pred_bin = modelo_bin.predict(x_vector[idx_test])
                    pred_multi = modelo_multi.predict(x_vector[idx_test])
                    raiz_clasificador = carpeta / "clasificadores" / clasificador / representacion
                    carpeta_bin = raiz_clasificador / "binaria"
                    carpeta_multi = raiz_clasificador / "multiclase"
                elif clasificador == "KNN":
                    modelo_bin = _pipeline_knn(config, len(idx_train), x_vector.shape[1])
                    modelo_multi = _pipeline_knn(config, len(idx_train), x_vector.shape[1])
                    modelo_bin.fit(x_vector[idx_train], y_bin[idx_train])
                    modelo_multi.fit(x_vector[idx_train], y_multi[idx_train])
                    pred_bin = modelo_bin.predict(x_vector[idx_test])
                    pred_multi = modelo_multi.predict(x_vector[idx_test])
                    raiz_clasificador = carpeta / "clasificadores" / clasificador / representacion
                    carpeta_bin = raiz_clasificador / "binaria"
                    carpeta_multi = raiz_clasificador / "multiclase"
                else:
                    carpeta_bin = carpeta / "clasificadores" / "UjaNet" / "binaria" / representacion
                    carpeta_multi = carpeta / "clasificadores" / "UjaNet" / "multiclase" / representacion
                    pred_bin = _entrenar_ujanet(
                        x_ujanet,
                        y_bin,
                        idx_train,
                        idx_test,
                        config,
                        binario=True,
                        ruta_modelo=carpeta_bin / f"fold_{numero_fold}_modelo.pt",
                    )
                    pred_multi = _entrenar_ujanet(
                        x_ujanet,
                        y_multi,
                        idx_train,
                        idx_test,
                        config,
                        binario=False,
                        ruta_modelo=carpeta_multi / f"fold_{numero_fold}_modelo.pt",
                    )

                metricas_bin_fold, metricas_multi_fold = _guardar_resultados_fold(
                    carpeta_bin,
                    carpeta_multi,
                    numero_fold,
                    metadatos,
                    idx_test,
                    y_bin[idx_test],
                    y_multi[idx_test],
                    np.asarray(pred_bin, dtype=int),
                    np.asarray(pred_multi, dtype=int),
                )
                fila_bin = {
                    "distribucion": distribucion,
                    "clasificador": clasificador,
                    "representacion": representacion,
                    "fold": numero_fold,
                    **metricas_bin_fold,
                }
                fila_multi = {
                    "distribucion": distribucion,
                    "clasificador": clasificador,
                    "representacion": representacion,
                    "fold": numero_fold,
                    **metricas_multi_fold,
                }
                _reemplazar_fila(filas_bin, fila_bin)
                _reemplazar_fila(filas_multi, fila_multi)
                _guardar_parcial(filas_bin, ruta_bin)
                _guardar_parcial(filas_multi, ruta_multi)
                completadas_bin.add(clave)
                completadas_multi.add(clave)
                print(f"[{distribucion}] {clasificador} {representacion} fold {numero_fold} completado")

    tabla_bin = pd.DataFrame(filas_bin).sort_values(
        ["clasificador", "representacion", "fold"]
    ).reset_index(drop=True)
    tabla_multi = pd.DataFrame(filas_multi).sort_values(
        ["clasificador", "representacion", "fold"]
    ).reset_index(drop=True)
    tabla_bin.to_csv(ruta_bin, index=False, encoding="utf-8-sig")
    tabla_multi.to_csv(ruta_multi, index=False, encoding="utf-8-sig")
    resumen_bin = resumir(tabla_bin)
    resumen_multi = resumir(tabla_multi)
    resumen_bin.to_csv(carpeta_metricas / "resumen_metricas_binarias.csv", index=False, encoding="utf-8-sig")
    resumen_multi.to_csv(carpeta_metricas / "resumen_metricas_multiclase.csv", index=False, encoding="utf-8-sig")
    auditoria_padding_df = pd.DataFrame(auditoria_padding)
    auditoria_padding_df.to_csv(
        carpeta / "auditoria_adaptacion_ujanet.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return tabla_bin, tabla_multi, auditoria_padding_df


def resumir(tabla: pd.DataFrame) -> pd.DataFrame:
    agrupadores = ["distribucion", "clasificador", "representacion"]
    resumen = tabla.groupby(agrupadores, as_index=False).agg(
        {columna: ["mean", "std"] for columna in COLUMNAS_RESUMEN}
    )
    resumen.columns = [
        "_".join(str(valor) for valor in columna if valor).strip("_")
        for columna in resumen.columns.to_flat_index()
    ]
    return resumen.sort_values(["clasificador", "representacion"]).reset_index(drop=True)

