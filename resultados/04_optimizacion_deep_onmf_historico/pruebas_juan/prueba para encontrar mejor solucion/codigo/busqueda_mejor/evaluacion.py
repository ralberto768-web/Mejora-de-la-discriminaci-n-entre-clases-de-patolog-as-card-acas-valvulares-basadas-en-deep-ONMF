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
from codigo.metricas import (
    matriz_confusion_binaria,
    metricas_binarias,
    metricas_multiclase,
)
from codigo.representaciones import vectorizar_para_clasicos

from .configuracion_busqueda import CLASIFICADORES, METRICAS, etiqueta, nombre_h


def adaptar_para_ujanet(x: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    filas, columnas = x.shape[1], x.shape[2]
    filas_objetivo = max(4, filas)
    columnas_objetivo = max(4, columnas)
    adaptada = np.pad(
        x,
        (
            (0, 0),
            (0, filas_objetivo - filas),
            (0, columnas_objetivo - columnas),
        ),
        mode="constant",
    )
    return adaptada.astype(np.float32, copy=False), {
        "padding_aplicado": (filas, columnas)
        != (filas_objetivo, columnas_objetivo),
        "forma_original": f"{filas}x{columnas}",
        "forma_ujanet": f"{filas_objetivo}x{columnas_objetivo}",
        "filas_cero_anadidas": filas_objetivo - filas,
        "columnas_cero_anadidas": columnas_objetivo - columnas,
    }


def _leer(ruta: Path) -> list[dict[str, object]]:
    if not ruta.exists():
        return []
    return pd.read_csv(ruta, encoding="utf-8-sig").to_dict(orient="records")


def _guardar(filas: list[dict[str, object]], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8-sig")


def _clave(fila: dict[str, object]) -> tuple[str, str, int]:
    return (
        str(fila["clasificador"]),
        str(fila["representacion"]),
        int(fila["fold"]),
    )


def _reemplazar(
    filas: list[dict[str, object]],
    nueva: dict[str, object],
) -> None:
    clave = _clave(nueva)
    filas[:] = [fila for fila in filas if _clave(fila) != clave]
    filas.append(nueva)


def _guardar_resultados_fold(
    carpeta_bin: Path,
    carpeta_multi: Path,
    fold: int,
    metadatos: pd.DataFrame,
    idx_test: np.ndarray,
    y_bin_test: np.ndarray,
    y_multi_test: np.ndarray,
    pred_bin: np.ndarray,
    pred_multi: np.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    carpeta_bin.mkdir(parents=True, exist_ok=True)
    carpeta_multi.mkdir(parents=True, exist_ok=True)
    predicciones = metadatos.iloc[idx_test][
        ["clase", "etiqueta_binaria", "archivo", "ruta"]
    ].copy()
    predicciones["fold"] = fold
    predicciones["pred_binaria"] = np.where(
        pred_bin == 1,
        "anomalo",
        "normal",
    )
    predicciones["pred_multiclase"] = [
        CLASES[int(indice)] for indice in pred_multi
    ]
    predicciones.to_csv(
        carpeta_bin / f"fold_{fold}_predicciones.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predicciones.to_csv(
        carpeta_multi / f"fold_{fold}_predicciones.csv",
        index=False,
        encoding="utf-8-sig",
    )
    matriz_confusion_binaria(y_bin_test, pred_bin).to_csv(
        carpeta_bin / f"fold_{fold}_matriz_confusion_binaria.csv",
        encoding="utf-8-sig",
    )
    matriz_multi, tabla_multi = metricas_multiclase(y_multi_test, pred_multi)
    matriz_multi.to_csv(
        carpeta_multi / f"fold_{fold}_matriz_confusion_multiclase.csv",
        encoding="utf-8-sig",
    )
    tabla_multi.to_csv(
        carpeta_multi / f"fold_{fold}_metricas_multiclase.csv",
        index=False,
        encoding="utf-8-sig",
    )
    macro = tabla_multi.loc[
        tabla_multi["clase"] == "PROMEDIO_MACRO"
    ].iloc[0]
    return metricas_binarias(y_bin_test, pred_bin), macro.to_dict()


def evaluar_configuracion(
    matrices: dict[str, np.ndarray],
    metadatos: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: ConfiguracionExperimento,
    carpeta: Path,
    reutilizar: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalua W y H final con los tres clasificadores y ambos problemas."""

    carpeta_metricas = carpeta / "metricas"
    ruta_bin = carpeta_metricas / "metricas_binarias_por_fold.csv"
    ruta_multi = carpeta_metricas / "metricas_multiclase_por_fold.csv"
    filas_bin = _leer(ruta_bin) if reutilizar else []
    filas_multi = _leer(ruta_multi) if reutilizar else []
    completadas_bin = {_clave(fila) for fila in filas_bin}
    completadas_multi = {_clave(fila) for fila in filas_multi}
    y_bin, y_multi = etiquetas_numericas(metadatos)
    distribucion = etiqueta(config.rangos_deep_onmf)
    representaciones = ("DeepONMF_W", nombre_h(len(config.rangos_deep_onmf)))
    filas_padding: list[dict[str, object]] = []

    for representacion in representaciones:
        x_matriz = matrices[representacion].astype(np.float32)
        x_vector = vectorizar_para_clasicos(x_matriz)
        x_ujanet, auditoria_padding = adaptar_para_ujanet(x_matriz)
        filas_padding.append(
            {
                "distribucion": distribucion,
                "representacion": representacion,
                **auditoria_padding,
            }
        )

        for clasificador in CLASIFICADORES:
            for numero_fold, (idx_train, idx_test) in enumerate(folds, start=1):
                clave = (clasificador, representacion, numero_fold)
                if clave in completadas_bin and clave in completadas_multi:
                    print(
                        f"[{distribucion}] Reutilizado: {clasificador} "
                        f"{representacion} fold {numero_fold}"
                    )
                    continue

                raiz = carpeta / "clasificadores" / clasificador
                carpeta_bin = raiz / "binaria" / representacion
                carpeta_multi = raiz / "multiclase" / representacion
                if clasificador in ("SVM", "KNN"):
                    creador = (
                        _pipeline_svm
                        if clasificador == "SVM"
                        else _pipeline_knn
                    )
                    modelo_bin = creador(
                        config,
                        len(idx_train),
                        x_vector.shape[1],
                    )
                    modelo_multi = creador(
                        config,
                        len(idx_train),
                        x_vector.shape[1],
                    )
                    modelo_bin.fit(x_vector[idx_train], y_bin[idx_train])
                    modelo_multi.fit(x_vector[idx_train], y_multi[idx_train])
                    pred_bin = modelo_bin.predict(x_vector[idx_test])
                    pred_multi = modelo_multi.predict(x_vector[idx_test])
                else:
                    ruta_bin_modelo = (
                        carpeta_bin / f"fold_{numero_fold}_modelo.pt"
                    )
                    ruta_multi_modelo = (
                        carpeta_multi / f"fold_{numero_fold}_modelo.pt"
                    )
                    pred_bin = _entrenar_ujanet(
                        x_ujanet,
                        y_bin,
                        idx_train,
                        idx_test,
                        config,
                        binario=True,
                        ruta_modelo=ruta_bin_modelo,
                    )
                    pred_multi = _entrenar_ujanet(
                        x_ujanet,
                        y_multi,
                        idx_train,
                        idx_test,
                        config,
                        binario=False,
                        ruta_modelo=ruta_multi_modelo,
                    )
                    # Los pesos no son necesarios para auditar las metricas.
                    ruta_bin_modelo.unlink(missing_ok=True)
                    ruta_multi_modelo.unlink(missing_ok=True)

                metricas_bin, metricas_multi = _guardar_resultados_fold(
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
                    "numero_capas": len(config.rangos_deep_onmf),
                    "clasificador": clasificador,
                    "representacion": representacion,
                    "fold": numero_fold,
                    **metricas_bin,
                }
                fila_multi = {
                    "distribucion": distribucion,
                    "numero_capas": len(config.rangos_deep_onmf),
                    "clasificador": clasificador,
                    "representacion": representacion,
                    "fold": numero_fold,
                    **metricas_multi,
                }
                _reemplazar(filas_bin, fila_bin)
                _reemplazar(filas_multi, fila_multi)
                _guardar(filas_bin, ruta_bin)
                _guardar(filas_multi, ruta_multi)
                completadas_bin.add(clave)
                completadas_multi.add(clave)
                print(
                    f"[{distribucion}] {clasificador} {representacion} "
                    f"fold {numero_fold} completado"
                )

    tabla_bin = pd.DataFrame(filas_bin).sort_values(
        ["clasificador", "representacion", "fold"]
    )
    tabla_multi = pd.DataFrame(filas_multi).sort_values(
        ["clasificador", "representacion", "fold"]
    )
    tabla_bin.to_csv(ruta_bin, index=False, encoding="utf-8-sig")
    tabla_multi.to_csv(ruta_multi, index=False, encoding="utf-8-sig")
    resumir(tabla_bin).to_csv(
        carpeta_metricas / "resumen_metricas_binarias.csv",
        index=False,
        encoding="utf-8-sig",
    )
    resumir(tabla_multi).to_csv(
        carpeta_metricas / "resumen_metricas_multiclase.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(filas_padding).to_csv(
        carpeta / "auditoria_adaptacion_ujanet.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return tabla_bin.reset_index(drop=True), tabla_multi.reset_index(drop=True)


def resumir(tabla: pd.DataFrame) -> pd.DataFrame:
    agrupadores = [
        "distribucion",
        "numero_capas",
        "clasificador",
        "representacion",
    ]
    resumen = tabla.groupby(agrupadores, as_index=False).agg(
        {metrica: ["mean", "std"] for metrica in METRICAS}
    )
    resumen.columns = [
        "_".join(str(valor) for valor in columna if valor).strip("_")
        for columna in resumen.columns.to_flat_index()
    ]
    return resumen.sort_values(
        ["clasificador", "representacion"]
    ).reset_index(drop=True)


def evaluacion_completa(carpeta: Path, folds: int) -> bool:
    ruta_bin = carpeta / "metricas" / "metricas_binarias_por_fold.csv"
    ruta_multi = carpeta / "metricas" / "metricas_multiclase_por_fold.csv"
    if not ruta_bin.exists() or not ruta_multi.exists():
        return False
    binaria = pd.read_csv(ruta_bin, encoding="utf-8-sig")
    multi = pd.read_csv(ruta_multi, encoding="utf-8-sig")
    esperadas = 2 * len(CLASIFICADORES) * folds
    return len(binaria) == esperadas and len(multi) == esperadas

