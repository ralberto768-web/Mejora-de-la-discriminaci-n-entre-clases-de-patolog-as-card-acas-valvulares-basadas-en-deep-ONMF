from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def etiqueta(rangos: Iterable[int]) -> str:
    return "-".join(str(int(valor)) for valor in rangos)


def clave(rangos: Iterable[int]) -> str:
    return "_".join(str(int(valor)) for valor in rangos)


def convertir_etiqueta(valor: str) -> tuple[int, ...]:
    return tuple(int(parte) for parte in str(valor).replace("_", "-").split("-"))


def nombre_h(rangos: tuple[int, ...]) -> str:
    return f"DeepONMF_H{len(rangos)}"


def generar_arquitecturas_excel_decrecientes() -> list[tuple[int, ...]]:
    arquitecturas: list[tuple[int, ...]] = []
    for inicio in range(32, 7, -2):
        for fin in range(2, 11, 2):
            if inicio <= fin:
                continue
            for numero_capas in (2, 3, 4):
                capas = tuple(
                    int(valor)
                    for valor in np.round(
                        np.geomspace(inicio, fin, numero_capas)
                    ).astype(int)
                )
                arquitecturas.append(capas)
    if len(arquitecturas) != 186 or len(set(arquitecturas)) != 186:
        raise AssertionError("La lista de arquitecturas del Excel debe contener 186 configuraciones unicas")
    return arquitecturas


def generar_plan_completo() -> list[tuple[int, ...]]:
    una_capa = [(base,) for base in range(8, 33, 2)]
    return una_capa + generar_arquitecturas_excel_decrecientes()


def limitar_por_capas(
    arquitecturas: list[tuple[int, ...]],
    max_por_capa: int,
) -> list[tuple[int, ...]]:
    if max_por_capa <= 0:
        return arquitecturas
    grupos: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for arquitectura in arquitecturas:
        grupos[len(arquitectura)].append(arquitectura)
    seleccionadas: list[tuple[int, ...]] = []
    for profundidad in sorted(grupos):
        valores = grupos[profundidad]
        if len(valores) <= max_por_capa:
            seleccionadas.extend(valores)
            continue
        indices = np.linspace(0, len(valores) - 1, max_por_capa).round().astype(int)
        seleccionadas.extend([valores[int(indice)] for indice in indices])
    return seleccionadas
