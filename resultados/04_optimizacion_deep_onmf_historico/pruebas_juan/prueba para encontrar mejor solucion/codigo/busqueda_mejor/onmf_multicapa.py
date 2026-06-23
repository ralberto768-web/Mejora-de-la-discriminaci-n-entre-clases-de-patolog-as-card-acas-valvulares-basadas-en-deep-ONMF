from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from codigo.onmf import CapaONMF, EPS, factorizar_onmf


@dataclass(frozen=True)
class ResultadoDeepONMFMulticapa:
    w_final: np.ndarray
    h_final: np.ndarray
    capas: list[CapaONMF]
    error_relativo_final: float


def deep_onmf_multicapa(
    matriz: np.ndarray,
    rangos: tuple[int, ...],
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
) -> ResultadoDeepONMFMulticapa:
    """Aplica Deep-ONMF con cualquier numero de capas entre dos y cinco."""

    if not 2 <= len(rangos) <= 5:
        raise ValueError("Deep-ONMF debe tener entre dos y cinco capas")
    if any(rango <= 0 for rango in rangos):
        raise ValueError("Todos los rangos deben ser positivos")

    entrada = np.maximum(matriz.astype(np.float64, copy=False), EPS)
    matrices_w: list[np.ndarray] = []
    capas: list[CapaONMF] = []
    for indice, rango in enumerate(rangos, start=1):
        inicio = time.perf_counter()
        w, h, error, ortogonalidad = factorizar_onmf(
            entrada,
            rango=rango,
            iteraciones=iteraciones,
            penalizacion_ortogonal=penalizacion_ortogonal,
            semilla=semilla + indice * 1000,
        )
        capas.append(
            CapaONMF(
                indice=indice,
                rango=rango,
                forma_entrada=entrada.shape,
                forma_w=w.shape,
                forma_h=h.shape,
                error_relativo=error,
                ortogonalidad_media=ortogonalidad,
                segundos=time.perf_counter() - inicio,
            )
        )
        matrices_w.append(w)
        entrada = h

    w_final = matrices_w[0]
    for matriz_w in matrices_w[1:]:
        w_final = w_final @ matriz_w

    normas = np.maximum(np.linalg.norm(w_final, axis=0), EPS)
    w_final = w_final / normas[None, :]
    h_final = entrada * normas[:, None]
    original = np.maximum(matriz.astype(np.float64, copy=False), EPS)
    error_final = float(
        np.linalg.norm(original - w_final @ h_final, ord="fro")
        / max(np.linalg.norm(original, ord="fro"), EPS)
    )
    return ResultadoDeepONMFMulticapa(
        w_final=w_final,
        h_final=h_final,
        capas=capas,
        error_relativo_final=error_final,
    )

