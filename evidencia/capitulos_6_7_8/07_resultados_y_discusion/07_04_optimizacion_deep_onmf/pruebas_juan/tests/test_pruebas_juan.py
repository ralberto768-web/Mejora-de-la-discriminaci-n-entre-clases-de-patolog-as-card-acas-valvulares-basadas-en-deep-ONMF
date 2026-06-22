from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "codigo"))
sys.path.insert(1, str(RAIZ.parent))

from codigo.configuracion import ConfiguracionExperimento
from codigo.onmf import deep_onmf
from pruebas_juan.configuracion_pruebas import configurar_rangos
from pruebas_juan.evaluacion import adaptar_para_ujanet, resumir
from pruebas_juan.extraccion_deep import comprobar_formas


def test_configuracion_solo_cambia_rangos() -> None:
    base = ConfiguracionExperimento()
    nueva = configurar_rangos(base, (15, 10, 5), rapido=False)
    datos_base = asdict(base)
    datos_nuevos = asdict(nueva)
    diferencias = {
        clave
        for clave in datos_base
        if datos_base[clave] != datos_nuevos[clave]
    }
    assert diferencias == {"rangos_deep_onmf"}


def test_formas_deep_onmf_para_los_tres_rangos() -> None:
    matriz = np.abs(np.random.default_rng(42).normal(size=(126, 212))) + 1e-6
    for rangos in ((15, 10, 5), (10, 6, 4), (8, 5, 3)):
        resultado = deep_onmf(
            matriz,
            rangos=rangos,
            iteraciones=2,
            penalizacion_ortogonal=0.05,
            semilla=42,
        )
        matrices = {
            "DeepONMF_W": resultado.w_final[None, :, :],
            "DeepONMF_H3": resultado.h3[None, :, :],
        }
        comprobar_formas(matrices, 1, rangos[-1], 126, 212)


def test_padding_ujanet_solo_cuando_es_necesario() -> None:
    x_tres = np.ones((2, 126, 3), dtype=np.float32)
    adaptada, auditoria = adaptar_para_ujanet(x_tres)
    assert adaptada.shape == (2, 126, 4)
    assert auditoria["padding_aplicado"] is True
    assert np.all(adaptada[:, :, -1] == 0)

    x_cuatro = np.ones((2, 4, 212), dtype=np.float32)
    igual, auditoria_igual = adaptar_para_ujanet(x_cuatro)
    assert igual.shape == x_cuatro.shape
    assert auditoria_igual["padding_aplicado"] is False


def test_resumen_genera_media_y_desviacion() -> None:
    filas = []
    for fold, accuracy in enumerate((0.8, 1.0), start=1):
        filas.append(
            {
                "distribucion": "15-10-5",
                "clasificador": "SVM",
                "representacion": "DeepONMF_W",
                "fold": fold,
                "Accuracy": accuracy,
                "Sensitivity": accuracy,
                "Specificity": accuracy,
                "Precision": accuracy,
                "Score": accuracy,
            }
        )
    resumen = resumir(pd.DataFrame(filas))
    assert len(resumen) == 1
    assert abs(resumen.loc[0, "Accuracy_mean"] - 0.9) < 1e-12
    assert resumen.loc[0, "Accuracy_std"] > 0

