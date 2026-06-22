from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .configuracion import CLASES


def metricas_binarias(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_real = np.asarray(y_real, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_real == 1) & (y_pred == 1)))
    tn = int(np.sum((y_real == 0) & (y_pred == 0)))
    fp = int(np.sum((y_real == 0) & (y_pred == 1)))
    fn = int(np.sum((y_real == 1) & (y_pred == 0)))
    total = max(tp + tn + fp + fn, 1)
    sensibilidad = tp / max(tp + fn, 1)
    especificidad = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": (tp + tn) / total,
        "Sensitivity": sensibilidad,
        "Specificity": especificidad,
        "Precision": precision,
        "Score": (sensibilidad + especificidad) / 2.0,
    }


def matriz_confusion_binaria(y_real: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    matriz = confusion_matrix(y_real, y_pred, labels=[0, 1])
    return pd.DataFrame(matriz, index=["real_normal", "real_anomalo"], columns=["pred_normal", "pred_anomalo"])


def metricas_multiclase(y_real: np.ndarray, y_pred: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    matriz = confusion_matrix(y_real, y_pred, labels=list(range(len(CLASES))))
    matriz_df = pd.DataFrame(matriz, index=[f"real_{c}" for c in CLASES], columns=[f"pred_{c}" for c in CLASES])
    filas: list[dict[str, float | int | str]] = []
    total = int(np.sum(matriz))
    for indice, clase in enumerate(CLASES):
        tp = int(matriz[indice, indice])
        fn = int(np.sum(matriz[indice, :]) - tp)
        fp = int(np.sum(matriz[:, indice]) - tp)
        tn = int(total - tp - fp - fn)
        base = metricas_binarias(
            np.concatenate([np.ones(tp + fn), np.zeros(tn + fp)]),
            np.concatenate([np.ones(tp), np.zeros(fn), np.zeros(tn), np.ones(fp)]),
        )
        filas.append({"clase": clase, **base, "soporte": int(np.sum(matriz[indice, :]))})
    resumen = pd.DataFrame(filas)
    medias = {"clase": "PROMEDIO_MACRO"}
    for columna in ["Accuracy", "Sensitivity", "Specificity", "Precision", "Score"]:
        medias[columna] = float(resumen[columna].mean())
    for columna in ["TP", "TN", "FP", "FN", "soporte"]:
        medias[columna] = int(resumen[columna].sum())
    resumen = pd.concat([resumen, pd.DataFrame([medias])], ignore_index=True)
    return matriz_df, resumen


def comprobar_formulas_metricas() -> None:
    y_real = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    metricas = metricas_binarias(y_real, y_pred)
    assert metricas["TP"] == 2
    assert metricas["TN"] == 2
    assert metricas["FP"] == 1
    assert metricas["FN"] == 1
    assert abs(metricas["Accuracy"] - (4 / 6)) < 1e-12
    assert abs(metricas["Score"] - ((2 / 3 + 2 / 3) / 2)) < 1e-12

