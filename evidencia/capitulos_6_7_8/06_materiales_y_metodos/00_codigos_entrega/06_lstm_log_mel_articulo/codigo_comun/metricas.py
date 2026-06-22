from __future__ import annotations

import numpy as np


def matriz_confusion(y_real: np.ndarray, y_predicho: np.ndarray, numero_clases: int) -> np.ndarray:
    matriz = np.zeros((numero_clases, numero_clases), dtype=np.int64)
    for real, predicho in zip(y_real, y_predicho):
        matriz[int(real), int(predicho)] += 1
    return matriz


def resumen_metricas(y_real: np.ndarray, y_predicho: np.ndarray, clases: list[str]) -> str:
    matriz = matriz_confusion(y_real, y_predicho, len(clases))
    exactitud = float(np.mean(y_real == y_predicho)) if len(y_real) else 0.0
    lineas = [f"Exactitud global: {exactitud:.4f}", "", "Matriz de confusion (filas=real, columnas=predicho):"]
    lineas.append(" " * 12 + " ".join(f"{clase:>7}" for clase in clases))
    for indice, clase in enumerate(clases):
        lineas.append(f"{clase:>10}: " + " ".join(f"{valor:7d}" for valor in matriz[indice]))
    lineas.append("")
    lineas.append("Metricas por clase:")
    for indice, clase in enumerate(clases):
        vp = matriz[indice, indice]
        precision = vp / max(1, matriz[:, indice].sum())
        sensibilidad = vp / max(1, matriz[indice, :].sum())
        f1 = 2 * precision * sensibilidad / max(1e-12, precision + sensibilidad)
        lineas.append(
            f"- {clase}: precision={precision:.4f}, sensibilidad={sensibilidad:.4f}, f1={f1:.4f}"
        )
    return "\n".join(lineas)

