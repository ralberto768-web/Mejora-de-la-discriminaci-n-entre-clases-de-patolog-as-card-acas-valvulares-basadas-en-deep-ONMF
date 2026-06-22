from __future__ import annotations

import unittest

import numpy as np

from codigo.clasificadores import crear_folds
from codigo.configuracion import ConfiguracionExperimento, REPRESENTACIONES
from codigo.metricas import metricas_binarias
from codigo.onmf import deep_onmf


class PruebasPunto3(unittest.TestCase):
    def test_representaciones_solicitadas(self) -> None:
        self.assertEqual(
            REPRESENTACIONES,
            (
                "STFT",
                "MFCC",
                "MelSpectrogram",
                "LogMelSpectrogram",
                "DeepONMF_W",
                "DeepONMF_H3",
            ),
        )

    def test_metricas_binarias_conocidas(self) -> None:
        resultado = metricas_binarias(
            np.array([1, 1, 1, 0, 0, 0]),
            np.array([1, 0, 1, 0, 1, 0]),
        )
        self.assertEqual(resultado["TP"], 2)
        self.assertEqual(resultado["TN"], 2)
        self.assertEqual(resultado["FP"], 1)
        self.assertEqual(resultado["FN"], 1)
        self.assertAlmostEqual(resultado["Accuracy"], 4 / 6)
        self.assertAlmostEqual(resultado["Score"], 2 / 3)

    def test_formas_w_h3(self) -> None:
        matriz = np.random.default_rng(42).random((126, 212))
        resultado = deep_onmf(
            matriz,
            rangos=(9, 8, 7),
            iteraciones=2,
            penalizacion_ortogonal=0.05,
            semilla=42,
        )
        self.assertEqual(resultado.w_final.shape, (126, 7))
        self.assertEqual(resultado.h3.shape, (7, 212))

    def test_folds_completos(self) -> None:
        etiquetas = np.repeat(np.arange(5), 200)
        folds = crear_folds(etiquetas, ConfiguracionExperimento())
        self.assertEqual(len(folds), 5)
        for entrenamiento, test in folds:
            self.assertEqual(len(entrenamiento), 800)
            self.assertEqual(len(test), 200)
            conteos = np.bincount(etiquetas[test], minlength=5)
            np.testing.assert_array_equal(conteos, np.full(5, 40))


if __name__ == "__main__":
    unittest.main()
