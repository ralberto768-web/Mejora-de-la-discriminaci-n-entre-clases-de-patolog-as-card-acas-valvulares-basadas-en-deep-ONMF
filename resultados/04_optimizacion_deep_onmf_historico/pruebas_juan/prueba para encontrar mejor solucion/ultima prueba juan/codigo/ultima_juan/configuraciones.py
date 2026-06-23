from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd


CLASIFICADORES = ("SVM", "KNN", "UjaNet")
METRICAS = ("Accuracy", "Sensitivity", "Specificity", "Precision", "Score")
CLASES = ("N", "AS", "MR", "MS", "MVP")
REFERENCIA = (9, 8, 7)
OBLIGATORIA_CUATRO = (15, 10, 5, 2)


def etiqueta(rangos: tuple[int, ...]) -> str:
    return "-".join(str(valor) for valor in rangos)


def clave(rangos: tuple[int, ...]) -> str:
    return "_".join(str(valor) for valor in rangos)


def convertir_etiqueta(valor: str) -> tuple[int, ...]:
    return tuple(int(numero) for numero in str(valor).split("-"))


def es_decreciente(rangos: tuple[int, ...]) -> bool:
    return all(a > b for a, b in zip(rangos, rangos[1:]))


def es_creciente(rangos: tuple[int, ...]) -> bool:
    return all(a < b for a, b in zip(rangos, rangos[1:]))


def invertir(rangos: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(rangos))


def _ranking_h(tabla_resumen: pd.DataFrame) -> pd.DataFrame:
    h = tabla_resumen[
        tabla_resumen["representacion"].astype(str).str.startswith(
            "DeepONMF_H"
        )
    ].copy()
    filas: list[dict[str, object]] = []
    for (distribucion, numero_capas), grupo in h.groupby(
        ["distribucion", "numero_capas"]
    ):
        por_clasificador = {
            str(fila["clasificador"]): fila
            for _, fila in grupo.iterrows()
        }
        if set(por_clasificador) != set(CLASIFICADORES):
            continue
        fila_salida: dict[str, object] = {
            "distribucion": str(distribucion),
            "numero_capas": int(numero_capas),
        }
        for clasificador in CLASIFICADORES:
            fila = por_clasificador[clasificador]
            fila_salida[f"Accuracy_{clasificador}"] = float(
                fila["Accuracy_mean"]
            )
            fila_salida[f"Score_{clasificador}"] = float(fila["Score_mean"])
            fila_salida[f"Accuracy_std_{clasificador}"] = float(
                fila["Accuracy_std"]
            )
        fila_salida["Accuracy_media"] = float(
            np.mean(
                [
                    fila_salida[f"Accuracy_{clasificador}"]
                    for clasificador in CLASIFICADORES
                ]
            )
        )
        fila_salida["Score_medio"] = float(
            np.mean(
                [
                    fila_salida[f"Score_{clasificador}"]
                    for clasificador in CLASIFICADORES
                ]
            )
        )
        fila_salida["estabilidad_std"] = float(
            np.mean(
                [
                    fila_salida[f"Accuracy_std_{clasificador}"]
                    for clasificador in CLASIFICADORES
                ]
            )
        )
        filas.append(fila_salida)
    return pd.DataFrame(filas).sort_values(
        ["Accuracy_media", "Score_medio", "estabilidad_std", "distribucion"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _vecinos(
    rangos: tuple[int, ...],
    minimo: int = 2,
    maximo: int = 20,
) -> set[tuple[int, ...]]:
    vecinos: set[tuple[int, ...]] = set()
    for deltas in product((-2, -1, 0, 1, 2), repeat=len(rangos)):
        if sum(delta != 0 for delta in deltas) > 2:
            continue
        candidato = tuple(
            valor + delta for valor, delta in zip(rangos, deltas)
        )
        if (
            candidato != rangos
            and all(minimo <= valor <= maximo for valor in candidato)
            and es_decreciente(candidato)
        ):
            vecinos.add(candidato)
    return vecinos


def _rellenar_diversidad(
    seleccionadas: set[tuple[int, ...]],
    profundidad: int,
    objetivo: int,
) -> set[tuple[int, ...]]:
    universo = [
        tuple(reversed(valores))
        for valores in combinations(range(2, 21), profundidad)
    ]
    candidatos = [valor for valor in universo if valor not in seleccionadas]
    seleccion = list(sorted(seleccionadas))
    while len(seleccion) < objetivo:
        matriz_seleccion = np.asarray(seleccion, dtype=float)
        mejor: tuple[int, ...] | None = None
        mejor_distancia = -1.0
        for candidato in candidatos:
            vector = np.asarray(candidato, dtype=float)
            distancia = float(
                np.min(np.linalg.norm(matriz_seleccion - vector, axis=1))
            )
            if distancia > mejor_distancia or (
                abs(distancia - mejor_distancia) < 1e-12
                and (mejor is None or candidato < mejor)
            ):
                mejor = candidato
                mejor_distancia = distancia
        if mejor is None:
            raise RuntimeError("No quedan arquitecturas para completar el muestreo")
        seleccion.append(mejor)
        candidatos.remove(mejor)
    return set(seleccion)


def seleccionar_arquitecturas_profundas(
    resumen_anterior: pd.DataFrame,
    objetivo_por_profundidad: int = 50,
) -> dict[int, list[tuple[int, ...]]]:
    ranking = _ranking_h(resumen_anterior)
    resultado: dict[int, list[tuple[int, ...]]] = {}
    for profundidad in (4, 5):
        existentes = {
            convertir_etiqueta(valor)
            for valor in resumen_anterior.loc[
                resumen_anterior["numero_capas"].eq(profundidad),
                "distribucion",
            ].astype(str)
        }
        seleccionadas = set(existentes)
        if profundidad == 4:
            seleccionadas.add(OBLIGATORIA_CUATRO)
        mejores = ranking[ranking["numero_capas"].eq(profundidad)].head(10)
        vecinos: set[tuple[int, ...]] = set()
        for valor in mejores["distribucion"].astype(str):
            vecinos.update(_vecinos(convertir_etiqueta(valor)))
        for candidato in sorted(
            vecinos,
            key=lambda valor: (
                min(
                    sum(abs(a - b) for a, b in zip(valor, base))
                    for base in existentes
                ),
                valor,
            ),
        ):
            if len(seleccionadas) >= min(
                objetivo_por_profundidad,
                len(existentes)
                + max(1, (objetivo_por_profundidad - len(existentes)) * 2 // 3),
            ):
                break
            seleccionadas.add(candidato)
        seleccionadas = _rellenar_diversidad(
            seleccionadas,
            profundidad,
            objetivo_por_profundidad,
        )
        resultado[profundidad] = sorted(seleccionadas, reverse=True)
    return resultado


def guardar_plan_arquitecturas(
    arquitecturas: dict[int, list[tuple[int, ...]]],
    anteriores: set[tuple[int, ...]],
    ruta: Path,
) -> pd.DataFrame:
    filas = []
    for profundidad, valores in arquitecturas.items():
        for rangos in valores:
            filas.append(
                {
                    "distribucion": etiqueta(rangos),
                    "numero_capas": profundidad,
                    "estado_inicial": (
                        "reutilizada" if rangos in anteriores else "pendiente"
                    ),
                    "obligatoria_juan": rangos == OBLIGATORIA_CUATRO,
                }
            )
    tabla = pd.DataFrame(filas).sort_values(
        ["numero_capas", "distribucion"]
    )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False, encoding="utf-8-sig")
    return tabla


def construir_ranking(
    resumen: pd.DataFrame,
    origenes: dict[str, str],
) -> pd.DataFrame:
    ranking = resumen[
        resumen["representacion"].astype(str).str.startswith("DeepONMF_H")
    ].copy()
    ranking = ranking.rename(
        columns={
            "Accuracy_mean": "Accuracy",
            "Score_mean": "Score",
            "Sensitivity_mean": "Sensitivity",
            "Specificity_mean": "Specificity",
            "Precision_mean": "Precision",
        }
    )
    ranking["origen"] = ranking["distribucion"].astype(str).map(
        origenes
    ).fillna("anterior")
    ranking = ranking.sort_values(
        [
            "Accuracy",
            "Score",
            "Sensitivity",
            "Specificity",
            "Precision",
            "clasificador",
            "distribucion",
        ],
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "posicion_resultado", range(1, len(ranking) + 1))
    mejores_por_arquitectura = ranking.drop_duplicates(
        "distribucion",
        keep="first",
    ).copy()
    posiciones = {
        distribucion: posicion
        for posicion, distribucion in enumerate(
            mejores_por_arquitectura["distribucion"].astype(str),
            start=1,
        )
    }
    ranking["posicion_arquitectura"] = ranking["distribucion"].astype(
        str
    ).map(posiciones)
    ranking["mejor_resultado_arquitectura"] = ~ranking.duplicated(
        "distribucion"
    )
    return ranking


def seleccionar_diez(ranking: pd.DataFrame) -> pd.DataFrame:
    referencia = etiqueta(REFERENCIA)
    mejores = ranking[ranking["mejor_resultado_arquitectura"]].copy()
    primeras = mejores.head(10).copy()
    if referencia not in set(primeras["distribucion"].astype(str)):
        primeras = pd.concat(
            [
                mejores.head(9),
                mejores[
                    mejores["distribucion"].astype(str).eq(referencia)
                ],
            ],
            ignore_index=True,
        )
    if len(primeras) != 10 or referencia not in set(
        primeras["distribucion"].astype(str)
    ):
        raise AssertionError("No se pudieron seleccionar diez con 9-8-7")
    primeras = primeras.copy()
    primeras.insert(0, "posicion_seleccion", range(1, 11))
    primeras["forzada_como_referencia"] = (
        primeras["distribucion"].astype(str).eq(referencia)
        & primeras["posicion_arquitectura"].gt(10)
    )
    primeras["distribucion_invertida"] = [
        etiqueta(invertir(convertir_etiqueta(valor)))
        for valor in primeras["distribucion"].astype(str)
    ]
    return primeras
