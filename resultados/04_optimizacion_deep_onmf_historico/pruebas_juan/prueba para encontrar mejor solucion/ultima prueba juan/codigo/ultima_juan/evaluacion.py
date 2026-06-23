from __future__ import annotations

from dataclasses import replace
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
from codigo.metricas import metricas_multiclase
from codigo.representaciones import vectorizar_para_clasicos

from busqueda_mejor.evaluacion import adaptar_para_ujanet, resumir
from busqueda_mejor.extraccion import (
    comprobar_formas,
    extraer_representaciones,
)

from .configuraciones import CLASIFICADORES, clave, etiqueta


def configurar(
    base: ConfiguracionExperimento,
    rangos: tuple[int, ...],
    rapido: bool,
) -> ConfiguracionExperimento:
    cambios: dict[str, object] = {"rangos_deep_onmf": rangos}
    if rapido:
        cambios.update(
            {
                "iteraciones_onmf": 4,
                "pca_componentes_max": 16,
                "ujanet_epocas": 2,
                "ujanet_paciencia": 1,
                "ujanet_lote": 4,
            }
        )
    return replace(base, **cambios)


def _leer(ruta: Path) -> list[dict[str, object]]:
    if not ruta.exists():
        return []
    return pd.read_csv(ruta, encoding="utf-8-sig").to_dict(orient="records")


def _guardar(filas: list[dict[str, object]], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8-sig")


def _clave_fila(fila: dict[str, object]) -> tuple[str, str, int]:
    return (
        str(fila["clasificador"]),
        str(fila["representacion"]),
        int(fila["fold"]),
    )


def evaluar_multiclase(
    matrices: dict[str, np.ndarray],
    metadatos: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: ConfiguracionExperimento,
    carpeta: Path,
) -> pd.DataFrame:
    ruta = carpeta / "metricas" / "metricas_multiclase_por_fold.csv"
    filas = _leer(ruta)
    completadas = {_clave_fila(fila) for fila in filas}
    _, y_multi = etiquetas_numericas(metadatos)
    distribucion = etiqueta(tuple(config.rangos_deep_onmf))
    nombre_h = f"DeepONMF_H{len(config.rangos_deep_onmf)}"

    for representacion in ("DeepONMF_W", nombre_h):
        x_matriz = matrices[representacion].astype(np.float32)
        x_vector = vectorizar_para_clasicos(x_matriz)
        x_ujanet, _ = adaptar_para_ujanet(x_matriz)
        for clasificador in CLASIFICADORES:
            for numero_fold, (idx_train, idx_test) in enumerate(folds, start=1):
                identificador = (clasificador, representacion, numero_fold)
                if identificador in completadas:
                    continue
                carpeta_salida = (
                    carpeta
                    / "pred"
                    / clasificador
                    / ("W" if representacion == "DeepONMF_W" else nombre_h[9:])
                )
                carpeta_salida.mkdir(parents=True, exist_ok=True)
                if clasificador in ("SVM", "KNN"):
                    creador = (
                        _pipeline_svm
                        if clasificador == "SVM"
                        else _pipeline_knn
                    )
                    modelo = creador(config, len(idx_train), x_vector.shape[1])
                    modelo.fit(x_vector[idx_train], y_multi[idx_train])
                    pred = modelo.predict(x_vector[idx_test])
                else:
                    ruta_modelo = (
                        carpeta_salida / f"f{numero_fold}.pt"
                    )
                    pred = _entrenar_ujanet(
                        x_ujanet,
                        y_multi,
                        idx_train,
                        idx_test,
                        config,
                        binario=False,
                        ruta_modelo=ruta_modelo,
                    )
                    ruta_modelo.unlink(missing_ok=True)

                pred = np.asarray(pred, dtype=int)
                predicciones = metadatos.iloc[idx_test][
                    ["clase", "etiqueta_binaria", "archivo", "ruta"]
                ].copy()
                predicciones["fold"] = numero_fold
                predicciones["pred_multiclase"] = [
                    CLASES[int(indice)] for indice in pred
                ]
                predicciones.to_csv(
                    carpeta_salida
                    / f"f{numero_fold}_pred.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                matriz, tabla_metricas = metricas_multiclase(
                    y_multi[idx_test],
                    pred,
                )
                matriz.to_csv(
                    carpeta_salida
                    / f"f{numero_fold}_cm.csv",
                    encoding="utf-8-sig",
                )
                tabla_metricas.to_csv(
                    carpeta_salida
                    / f"f{numero_fold}_met.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                macro = tabla_metricas.loc[
                    tabla_metricas["clase"].eq("PROMEDIO_MACRO")
                ].iloc[0]
                nueva = {
                    "distribucion": distribucion,
                    "numero_capas": len(config.rangos_deep_onmf),
                    "clasificador": clasificador,
                    "representacion": representacion,
                    "fold": numero_fold,
                    **macro.to_dict(),
                }
                filas = [
                    fila
                    for fila in filas
                    if _clave_fila(fila) != identificador
                ]
                filas.append(nueva)
                completadas.add(identificador)
                _guardar(filas, ruta)
                print(
                    f"[{distribucion}] {clasificador} {representacion} "
                    f"fold {numero_fold}"
                )

    tabla = pd.DataFrame(filas).sort_values(
        ["clasificador", "representacion", "fold"]
    )
    esperadas = 2 * len(CLASIFICADORES) * len(folds)
    if len(tabla) != esperadas:
        raise AssertionError(
            f"{distribucion}: se esperaban {esperadas} filas multiclase, "
            f"hay {len(tabla)}"
        )
    tabla.to_csv(ruta, index=False, encoding="utf-8-sig")
    resumir(tabla).to_csv(
        carpeta / "metricas" / "resumen_metricas_multiclase.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return tabla.reset_index(drop=True)


def ejecutar_configuracion(
    rangos: tuple[int, ...],
    registros,
    metadatos_maestros: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config_base: ConfiguracionExperimento,
    carpeta_raiz: Path,
    rapido: bool,
) -> Path:
    carpeta = carpeta_raiz / clave(rangos)
    marcador = carpeta / "CONFIGURACION_COMPLETADA.txt"
    if marcador.exists():
        return carpeta
    config = configurar(config_base, rangos, rapido)
    extraccion = extraer_representaciones(
        registros,
        config,
        carpeta / "representaciones",
        reutilizar=True,
    )
    claves = (
        extraccion.metadatos["clase"].astype(str)
        + "/"
        + extraccion.metadatos["archivo"].astype(str)
    ).tolist()
    claves_maestras = (
        metadatos_maestros["clase"].astype(str)
        + "/"
        + metadatos_maestros["archivo"].astype(str)
    ).tolist()
    if claves != claves_maestras:
        raise AssertionError("El orden de los audios no coincide")
    comprobar_formas(
        extraccion.matrices,
        numero_audios=len(registros),
        numero_capas=len(rangos),
        rango_final=rangos[-1],
        bins_frecuencia=config.bins_frecuencia,
        tramas_tiempo=config.tramas_stft_por_segmento,
    )
    evaluar_multiclase(
        extraccion.matrices,
        extraccion.metadatos,
        folds,
        config,
        carpeta,
    )
    marcador.write_text(
        f"{etiqueta(rangos)}; audios={len(registros)}; folds={len(folds)}\n",
        encoding="utf-8",
    )
    # Las matrices pueden reconstruirse; las predicciones y auditorias se guardan.
    (
        carpeta / "representaciones" / "representaciones_deep_onmf.npz"
    ).unlink(missing_ok=True)
    return carpeta
