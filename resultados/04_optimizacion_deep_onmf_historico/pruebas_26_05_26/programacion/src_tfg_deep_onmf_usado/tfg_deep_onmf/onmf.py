from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass
class CapaONMF:
    indice: int
    rango: int
    forma_entrada: tuple[int, int]
    forma_w: tuple[int, int]
    forma_h: tuple[int, int]
    error_relativo: float
    ortogonalidad_media: float
    segundos: float


@dataclass
class ResultadoONMF:
    w_final: np.ndarray
    h_final: np.ndarray
    capas: list[CapaONMF]
    error_relativo_final: float


def _normalizar_columnas_w(w: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    escala = np.linalg.norm(w, axis=0)
    escala = np.maximum(escala, 1e-12)
    return w / escala[None, :], h * escala[:, None]


def _ortogonalidad_media(h: np.ndarray) -> float:
    normas = np.linalg.norm(h, axis=1, keepdims=True)
    h_norm = h / np.maximum(normas, 1e-12)
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
    """Factorización no negativa con penalización de ortogonalidad sobre H."""

    eps = 1e-12
    x = np.maximum(matriz, eps).astype(np.float64, copy=False)
    filas, columnas = x.shape
    rng = np.random.default_rng(semilla)
    w = rng.random((filas, rango)) + eps
    h = rng.random((rango, columnas)) + eps
    w, h = _normalizar_columnas_w(w, h)

    for _ in range(iteraciones):
        numerador_w = x @ h.T
        denominador_w = w @ (h @ h.T) + eps
        w *= numerador_w / denominador_w
        w = np.maximum(w, eps)

        numerador_h = w.T @ x + penalizacion_ortogonal * h
        denominador_h = (w.T @ w) @ h + penalizacion_ortogonal * ((h @ h.T) @ h) + eps
        h *= numerador_h / denominador_h
        h = np.maximum(h, eps)

        w, h = _normalizar_columnas_w(w, h)

    reconstruida = w @ h
    error = float(np.linalg.norm(x - reconstruida, ord="fro") / np.maximum(np.linalg.norm(x, ord="fro"), eps))
    ortogonalidad = _ortogonalidad_media(h)
    return w, h, error, ortogonalidad


def deep_onmf(
    matriz: np.ndarray,
    rangos: tuple[int, int, int],
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
) -> ResultadoONMF:
    entrada = matriz
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
        segundos = time.perf_counter() - inicio
        matrices_w.append(w)
        capas.append(
            CapaONMF(
                indice=indice,
                rango=rango,
                forma_entrada=entrada.shape,
                forma_w=w.shape,
                forma_h=h.shape,
                error_relativo=error,
                ortogonalidad_media=ortogonalidad,
                segundos=segundos,
            )
        )
        entrada = h

    w_final = matrices_w[0] @ matrices_w[1] @ matrices_w[2]
    normas = np.maximum(np.linalg.norm(w_final, axis=0), 1e-12)
    w_final = w_final / normas[None, :]
    h_final = entrada * normas[:, None]
    reconstruida_final = w_final @ h_final
    x = np.maximum(matriz, 1e-12)
    error_final = float(
        np.linalg.norm(x - reconstruida_final, ord="fro") / np.maximum(np.linalg.norm(x, ord="fro"), 1e-12)
    )
    return ResultadoONMF(
        w_final=w_final,
        h_final=h_final,
        capas=capas,
        error_relativo_final=error_final,
    )


def proyectar_sobre_w(
    matriz: np.ndarray,
    w: np.ndarray,
    iteraciones: int = 80,
) -> tuple[np.ndarray, float]:
    """Calcula H con W fijo y devuelve el error relativo de reconstrucción."""

    eps = 1e-12
    x = np.maximum(matriz, eps).astype(np.float64, copy=False)
    w = np.maximum(w, eps).astype(np.float64, copy=False)
    h = np.maximum(w.T @ x, eps)
    wt_x = w.T @ x
    wt_w = w.T @ w

    for _ in range(iteraciones):
        h *= wt_x / (wt_w @ h + eps)
        h = np.maximum(h, eps)

    reconstruida = w @ h
    error = float(np.linalg.norm(x - reconstruida, ord="fro") / np.maximum(np.linalg.norm(x, ord="fro"), eps))
    return h, error
