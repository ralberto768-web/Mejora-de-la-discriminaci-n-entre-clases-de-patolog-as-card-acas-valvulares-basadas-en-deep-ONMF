from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from codigo.clasificadores import _entrenar_ujanet, crear_folds
from codigo.configuracion import ConfiguracionExperimento
from codigo.metricas import metricas_multiclase

from .configuracion import CLASES, METRICAS


PROTOCOLO = "ultima_juan_ujanet_multiclase_v3_nndsvd_auditado"


def folds_desde_metadatos(
    metadatos: pd.DataFrame,
    config: ConfiguracionExperimento,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = metadatos["clase"].map({clase: indice for indice, clase in enumerate(CLASES)}).to_numpy(dtype=int)
    return crear_folds(y, config)


def adaptar_para_ujanet(x: np.ndarray) -> np.ndarray:
    filas, columnas = x.shape[1], x.shape[2]
    filas_objetivo = max(4, filas)
    columnas_objetivo = max(4, columnas)
    if (filas, columnas) == (filas_objetivo, columnas_objetivo):
        return x.astype(np.float32, copy=False)
    return np.pad(
        x,
        (
            (0, 0),
            (0, filas_objetivo - filas),
            (0, columnas_objetivo - columnas),
        ),
        mode="constant",
    ).astype(np.float32)


def _protocolo_valido(ruta: Path, representacion: str, forma: tuple[int, ...], folds: int) -> bool:
    if not ruta.exists():
        return False
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        datos.get("protocolo") == PROTOCOLO
        and datos.get("representacion") == representacion
        and tuple(datos.get("forma", [])) == tuple(int(v) for v in forma)
        and int(datos.get("folds", -1)) == folds
    )


def _guardar_matriz_agregada(
    predicciones: pd.DataFrame,
    carpeta: Path,
    titulo: str,
) -> dict[str, object]:
    conteos = pd.crosstab(
        pd.Categorical(predicciones["clase"], categories=CLASES),
        pd.Categorical(predicciones["pred_multiclase"], categories=CLASES),
        dropna=False,
    )
    conteos.index = list(CLASES)
    conteos.columns = list(CLASES)
    porcentajes = conteos.div(conteos.sum(axis=1), axis=0) * 100.0
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_conteos = carpeta / "matriz_confusion_conteos.csv"
    ruta_porcentajes = carpeta / "matriz_confusion_porcentajes.csv"
    ruta_png = carpeta / "matriz_confusion.png"
    ruta_pred = carpeta / "predicciones_agregadas.csv"
    conteos.to_csv(ruta_conteos, encoding="utf-8-sig")
    porcentajes.to_csv(ruta_porcentajes, encoding="utf-8-sig")
    predicciones.to_csv(ruta_pred, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(porcentajes.to_numpy(), cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(CLASES)), CLASES, fontsize=8)
    ax.set_yticks(range(len(CLASES)), CLASES, fontsize=8)
    ax.set_xlabel("Predicha", fontsize=8)
    ax.set_ylabel("Real", fontsize=8)
    ax.set_title(titulo, fontsize=9)
    for fila in range(len(CLASES)):
        for columna in range(len(CLASES)):
            porcentaje = float(porcentajes.iloc[fila, columna])
            ax.text(
                columna,
                fila,
                f"{int(conteos.iloc[fila, columna])}\n{porcentaje:.1f}%",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if porcentaje >= 55 else "black",
            )
    fig.tight_layout(pad=0.5)
    fig.savefig(ruta_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    aciertos = int(np.trace(conteos.to_numpy()))
    return {
        "predicciones": len(predicciones),
        "aciertos": aciertos,
        "exactitud_directa": aciertos / max(len(predicciones), 1),
        "ruta_matriz_png": str(ruta_png.resolve()),
        "ruta_matriz_conteos": str(ruta_conteos.resolve()),
        "ruta_predicciones": str(ruta_pred.resolve()),
    }


def evaluar_ujanet_multiclase(
    base: str,
    representacion: str,
    x: np.ndarray,
    metadatos: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: ConfiguracionExperimento,
    carpeta: Path,
    distribucion: str = "",
    numero_capas: int | None = None,
    inicializacion_onmf: str = "",
    entrada_ujanet: str | None = None,
) -> dict[str, object]:
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_resumen = carpeta / "resumen_metricas.csv"
    ruta_protocolo = carpeta / "protocolo_entrenamiento.json"
    cache_valida = (
        ruta_resumen.exists()
        and _protocolo_valido(ruta_protocolo, representacion, tuple(x.shape), len(folds))
    )
    if cache_valida:
        return pd.read_csv(ruta_resumen, encoding="utf-8-sig").iloc[0].to_dict()

    x_ujanet = adaptar_para_ujanet(np.asarray(x, dtype=np.float32))
    y = metadatos["clase"].map({clase: indice for indice, clase in enumerate(CLASES)}).to_numpy(dtype=int)
    filas: list[dict[str, object]] = []
    predicciones_todas: list[pd.DataFrame] = []
    config_eval = replace(config, semilla=config.semilla)
    ruta_protocolo.write_text(
        json.dumps(
            {
                "protocolo": PROTOCOLO,
                "base": base,
                "representacion": representacion,
                "distribucion": distribucion,
                "numero_capas": numero_capas,
                "inicializacion_onmf": inicializacion_onmf,
                "entrada_ujanet": entrada_ujanet
                or ("matriz_H_final_deep_onmf" if inicializacion_onmf else "representacion_clasica"),
                "forma": list(x.shape),
                "folds": len(folds),
                "semilla": config.semilla,
                "epocas": config.ujanet_epocas,
                "paciencia": config.ujanet_paciencia,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for fold, (idx_train, idx_test) in enumerate(folds, start=1):
        ruta_pred = carpeta / f"fold_{fold}_predicciones.csv"
        if cache_valida and ruta_pred.exists():
            predicciones = pd.read_csv(ruta_pred, encoding="utf-8-sig")
        else:
            pred = _entrenar_ujanet(
                x_ujanet,
                y,
                idx_train,
                idx_test,
                config_eval,
                binario=False,
                ruta_modelo=carpeta / f"fold_{fold}_modelo.pt",
            )
            (carpeta / f"fold_{fold}_modelo.pt").unlink(missing_ok=True)
            predicciones = metadatos.iloc[idx_test][["clase", "etiqueta_binaria", "archivo", "ruta"]].copy()
            predicciones["fold"] = fold
            predicciones["pred_multiclase"] = [CLASES[int(indice)] for indice in np.asarray(pred, dtype=int)]
            predicciones.to_csv(ruta_pred, index=False, encoding="utf-8-sig")

        matriz, metricas = metricas_multiclase(
            predicciones["clase"].map({clase: indice for indice, clase in enumerate(CLASES)}).to_numpy(dtype=int),
            predicciones["pred_multiclase"].map({clase: indice for indice, clase in enumerate(CLASES)}).to_numpy(dtype=int),
        )
        matriz.to_csv(carpeta / f"fold_{fold}_matriz_confusion_multiclase.csv", encoding="utf-8-sig")
        metricas.to_csv(carpeta / f"fold_{fold}_metricas_multiclase.csv", index=False, encoding="utf-8-sig")
        fila_macro = metricas.loc[metricas["clase"].eq("PROMEDIO_MACRO")].iloc[0].to_dict()
        filas.append({"base": base, "representacion": representacion, "fold": fold, **fila_macro})
        predicciones_todas.append(predicciones)
        pd.DataFrame(filas).to_csv(carpeta / "metricas_por_fold.csv", index=False, encoding="utf-8-sig")
        print(f"[UjaNet] {base} / {representacion} / fold {fold}/{len(folds)}")

    tabla = pd.DataFrame(filas)
    resumen: dict[str, object] = {
        "base": base,
        "clasificador": "UjaNet",
        "protocolo_ujanet": PROTOCOLO,
        "representacion": representacion,
        "distribucion": distribucion,
        "numero_capas": numero_capas if numero_capas is not None else "",
    }
    for metrica in METRICAS:
        resumen[f"{metrica}_mean"] = float(tabla[metrica].mean())
        resumen[f"{metrica}_std"] = float(tabla[metrica].std()) if len(tabla) > 1 else 0.0
    resumen.update(
        _guardar_matriz_agregada(
            pd.concat(predicciones_todas, ignore_index=True),
            carpeta / "matriz_confusion_agregada",
            representacion,
        )
    )
    pd.DataFrame([resumen]).to_csv(ruta_resumen, index=False, encoding="utf-8-sig")
    return resumen
