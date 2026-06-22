from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class CapaONMF:
    indice: int
    rango: int
    forma_entrada: tuple[int, int]
    forma_w: tuple[int, int]
    forma_h: tuple[int, int]
    error_relativo: float
    ortogonalidad_media: float
    segundos: float


@dataclass(frozen=True)
class ResultadoDeepONMF:
    w_final: np.ndarray
    h3: np.ndarray
    capas: list[CapaONMF]
    error_relativo_final: float


def _normalizar_columnas_w(w: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    escala = np.maximum(np.linalg.norm(w, axis=0), EPS)
    return w / escala[None, :], h * escala[:, None]


def _ortogonalidad_media(h: np.ndarray) -> float:
    normas = np.maximum(np.linalg.norm(h, axis=1, keepdims=True), EPS)
    h_norm = h / normas
    gramo = h_norm @ h_norm.T
    mascara = ~np.eye(gramo.shape[0], dtype=bool)
    return float(np.mean(np.abs(gramo[mascara])))


def factorizar_onmf(
    matriz: np.ndarray,
    rango: int,
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Factorizacion ONMF no negativa con penalizacion de ortogonalidad en H."""

    x = np.maximum(matriz.astype(np.float64, copy=False), EPS)
    rng = np.random.default_rng(semilla)
    w = rng.random((x.shape[0], rango)) + EPS
    h = rng.random((rango, x.shape[1])) + EPS
    w, h = _normalizar_columnas_w(w, h)

    for _ in range(iteraciones):
        w *= (x @ h.T) / (w @ (h @ h.T) + EPS)
        w = np.maximum(w, EPS)
        h *= (w.T @ x + penalizacion_ortogonal * h) / (
            (w.T @ w) @ h + penalizacion_ortogonal * ((h @ h.T) @ h) + EPS
        )
        h = np.maximum(h, EPS)
        w, h = _normalizar_columnas_w(w, h)

    reconstruida = w @ h
    error = float(np.linalg.norm(x - reconstruida, ord="fro") / max(np.linalg.norm(x, ord="fro"), EPS))
    return w, h, error, _ortogonalidad_media(h)


def deep_onmf(
    matriz: np.ndarray,
    rangos: tuple[int, int, int],
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
) -> ResultadoDeepONMF:
    """Aplica tres capas Deep-ONMF y devuelve W final y H3.

    W final representa la estructura espectral aprendida. H3 representa la
    activacion temporal final de la tercera capa, que es la representacion
    temporal que se compara en el TFG.
    """

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

    w_final = matrices_w[0] @ matrices_w[1] @ matrices_w[2]
    normas = np.maximum(np.linalg.norm(w_final, axis=0), EPS)
    w_final = w_final / normas[None, :]
    h3 = entrada * normas[:, None]
    x = np.maximum(matriz.astype(np.float64, copy=False), EPS)
    error_final = float(np.linalg.norm(x - w_final @ h3, ord="fro") / max(np.linalg.norm(x, ord="fro"), EPS))
    return ResultadoDeepONMF(w_final=w_final, h3=h3, capas=capas, error_relativo_final=error_final)
