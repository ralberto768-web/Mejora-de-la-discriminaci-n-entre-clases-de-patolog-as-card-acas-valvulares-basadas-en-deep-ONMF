from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
RAIZ_ULTIMA = RAIZ.parent
RAIZ_BUSQUEDA = RAIZ_ULTIMA.parent
RAIZ_PRUEBAS = RAIZ_BUSQUEDA.parent
RAIZ_IMPLEMENTACION = RAIZ_PRUEBAS.parent
for ruta in (
    RAIZ / "codigo",
    RAIZ_ULTIMA / "codigo",
    RAIZ_BUSQUEDA / "codigo",
    RAIZ_IMPLEMENTACION,
    RAIZ_PRUEBAS / "codigo",
):
    sys.path.insert(0, str(ruta))

from busqueda_mejor.onmf_multicapa import deep_onmf_multicapa
from excel_juan.arquitecturas import construir_plan
from excel_juan.resultados import seleccionar_resumenes


def test_excel_y_codigo_coinciden() -> None:
    plan = construir_plan(RAIZ / "configuraciones_arquitecturas.xlsx")
    assert len(plan) == 372
    assert plan["pareja"].nunique() == 186
    assert plan["distribucion"].nunique() == 372
    assert plan["dimensiones_repetidas"].sum() == 6


def test_onmf_admite_dimensiones_repetidas() -> None:
    matriz = np.abs(np.random.default_rng(42).normal(size=(126, 212)))
    resultado = deep_onmf_multicapa(
        matriz,
        rangos=(10, 9, 9, 8),
        iteraciones=2,
        penalizacion_ortogonal=0.05,
        semilla=42,
    )
    assert resultado.w_final.shape == (126, 8)
    assert resultado.h_final.shape == (8, 212)


def test_selecciones_tienen_veinte_parejas() -> None:
    filas = []
    for pareja in range(1, 187):
        for sentido, desplazamiento in (
            ("decreciente", 0.0),
            ("creciente", 0.0001),
        ):
            filas.append(
                {
                    "pareja": pareja,
                    "sentido": sentido,
                    "distribucion": (
                        f"{pareja + 10}-2"
                        if sentido == "decreciente"
                        else f"2-{pareja + 10}"
                    ),
                    "numero_capas": 2,
                    "clasificador": "UjaNet",
                    "representacion": "DeepONMF_H2",
                    "Accuracy_mean": 1 - pareja / 1000 + desplazamiento,
                    "Score_mean": 0.9 - pareja / 2000,
                    "Sensitivity_mean": 0.9,
                    "Specificity_mean": 0.9,
                    "Precision_mean": 0.9,
                }
            )
    with tempfile.TemporaryDirectory() as temporal:
        carpeta = Path(temporal)
        (carpeta / "tablas_csv").mkdir()
        selecciones = seleccionar_resumenes(
            pd.DataFrame(filas),
            carpeta,
        )
        for tabla in selecciones.values():
            assert len(tabla) == 40
            assert tabla["pareja"].nunique() == 20
            assert list(tabla.iloc[::2]["sentido"].astype(str)) == [
                "decreciente"
            ] * 20
            assert list(tabla.iloc[1::2]["sentido"].astype(str)) == [
                "creciente"
            ] * 20
