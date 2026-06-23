from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
IMPLEMENTACION = RAIZ.parents[1]
PRUEBAS_JUAN = RAIZ.parent
for ruta in (RAIZ / "codigo", IMPLEMENTACION, PRUEBAS_JUAN / "codigo"):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from codigo.onmf import deep_onmf
from busqueda_mejor.busqueda import generar_vecinos, tabla_compacta_busqueda
from busqueda_mejor.configuracion_busqueda import (
    CLASIFICADORES,
    ConfiguracionBusqueda,
    arquitectura_valida,
)
from busqueda_mejor.onmf_multicapa import deep_onmf_multicapa


class PruebasBusquedaMejor(unittest.TestCase):
    def test_tres_capas_reproduce_el_motor_original(self) -> None:
        rng = np.random.default_rng(123)
        matriz = rng.random((18, 24))
        original = deep_onmf(
            matriz,
            rangos=(9, 8, 7),
            iteraciones=3,
            penalizacion_ortogonal=0.05,
            semilla=42,
        )
        multicapa = deep_onmf_multicapa(
            matriz,
            rangos=(9, 8, 7),
            iteraciones=3,
            penalizacion_ortogonal=0.05,
            semilla=42,
        )
        np.testing.assert_allclose(multicapa.w_final, original.w_final)
        np.testing.assert_allclose(multicapa.h_final, original.h3)
        self.assertEqual(
            multicapa.error_relativo_final,
            original.error_relativo_final,
        )

    def test_formas_para_dos_y_cinco_capas(self) -> None:
        matriz = np.ones((126, 212), dtype=float)
        for rangos in ((10, 5), (12, 10, 8, 6, 4)):
            resultado = deep_onmf_multicapa(
                matriz,
                rangos=rangos,
                iteraciones=1,
                penalizacion_ortogonal=0.05,
                semilla=42,
            )
            self.assertEqual(resultado.w_final.shape, (126, rangos[-1]))
            self.assertEqual(resultado.h_final.shape, (rangos[-1], 212))
            self.assertEqual(len(resultado.capas), len(rangos))

    def test_vecinos_respetan_limites_y_descenso(self) -> None:
        config = ConfiguracionBusqueda()
        vecinos = generar_vecinos((10, 9, 7, 5), config, paso=1)
        self.assertTrue(vecinos)
        self.assertTrue(
            all(arquitectura_valida(vecino, config) for vecino in vecinos)
        )
        self.assertTrue(any(len(vecino) == 5 for vecino in vecinos))
        self.assertTrue(any(len(vecino) == 3 for vecino in vecinos))

    def test_ranking_manda_accuracy_y_score_desempata(self) -> None:
        filas = []
        valores = {
            "9-8-7": (0.95, 0.94),
            "10-9-7-5": (0.96, 0.91),
            "12-8-4": (0.96, 0.93),
        }
        for distribucion, (accuracy, score) in valores.items():
            for clasificador in CLASIFICADORES:
                filas.append(
                    {
                        "distribucion": distribucion,
                        "numero_capas": len(distribucion.split("-")),
                        "clasificador": clasificador,
                        "representacion": (
                            "DeepONMF_H"
                            + str(len(distribucion.split("-")))
                        ),
                        "Accuracy_mean": accuracy,
                        "Accuracy_std": 0.01,
                        "Score_mean": score,
                    }
                )
        tabla = tabla_compacta_busqueda(pd.DataFrame(filas), 2)
        self.assertEqual(tabla.iloc[0]["distribucion"], "12-8-4")
        self.assertEqual(tabla.iloc[1]["distribucion"], "10-9-7-5")
        self.assertEqual(int(tabla["finalista"].sum()), 2)


if __name__ == "__main__":
    unittest.main()
