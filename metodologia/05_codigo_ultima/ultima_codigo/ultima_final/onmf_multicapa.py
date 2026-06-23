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
class ResultadoDeepONMFMulticapa:
    w_final: np.ndarray
    h_final: np.ndarray
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


def _inicializar_nndsvd(matriz: np.ndarray, rango: int) -> tuple[np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(matriz, full_matrices=False)
    componentes = min(rango, len(s))
    w = np.zeros((matriz.shape[0], rango), dtype=np.float64)
    h = np.zeros((rango, matriz.shape[1]), dtype=np.float64)

    if componentes == 0:
        return np.full_like(w, EPS), np.full_like(h, EPS)

    w[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
    h[0, :] = np.sqrt(s[0]) * np.abs(vt[0, :])

    for j in range(1, componentes):
        x = u[:, j]
        y = vt[j, :]
        x_pos, x_neg = np.maximum(x, 0), np.maximum(-x, 0)
        y_pos, y_neg = np.maximum(y, 0), np.maximum(-y, 0)
        norma_pos = np.linalg.norm(x_pos) * np.linalg.norm(y_pos)
        norma_neg = np.linalg.norm(x_neg) * np.linalg.norm(y_neg)
        if norma_pos >= norma_neg:
            u_j = x_pos / max(np.linalg.norm(x_pos), EPS)
            v_j = y_pos / max(np.linalg.norm(y_pos), EPS)
            sigma = norma_pos
        else:
            u_j = x_neg / max(np.linalg.norm(x_neg), EPS)
            v_j = y_neg / max(np.linalg.norm(y_neg), EPS)
            sigma = norma_neg
        factor = np.sqrt(s[j] * sigma)
        w[:, j] = factor * u_j
        h[j, :] = factor * v_j

    media = float(np.mean(matriz[matriz > EPS])) if np.any(matriz > EPS) else EPS
    w[w <= EPS] = media * 1e-3
    h[h <= EPS] = media * 1e-3
    return np.maximum(w, EPS), np.maximum(h, EPS)


def factorizar_onmf(
    matriz: np.ndarray,
    rango: int,
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Factorizacion ONMF inicializada con NNDSVD para W y H."""

    del semilla
    x = np.maximum(matriz.astype(np.float64, copy=False), EPS)
    w, h = _inicializar_nndsvd(x, int(rango))
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


def deep_onmf_multicapa(
    matriz: np.ndarray,
    rangos: tuple[int, ...],
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
) -> ResultadoDeepONMFMulticapa:
    if not 1 <= len(rangos) <= 4:
        raise ValueError("Deep-ONMF debe tener entre una y cuatro capas")
    if any(int(rango) <= 0 for rango in rangos):
        raise ValueError("Todos los rangos deben ser positivos")

    entrada = np.maximum(matriz.astype(np.float64, copy=False), EPS)
    original = entrada.copy()
    matrices_w: list[np.ndarray] = []
    capas: list[CapaONMF] = []

    for indice, rango in enumerate(rangos, start=1):
        inicio = time.perf_counter()
        w, h, error, ortogonalidad = factorizar_onmf(
            entrada,
            rango=int(rango),
            iteraciones=iteraciones,
            penalizacion_ortogonal=penalizacion_ortogonal,
            semilla=semilla + indice * 1000,
        )
        capas.append(
            CapaONMF(
                indice=indice,
                rango=int(rango),
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
