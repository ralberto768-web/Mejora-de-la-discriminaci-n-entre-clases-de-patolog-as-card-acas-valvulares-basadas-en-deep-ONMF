from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
BUSQUEDA = RAIZ.parent
IMPLEMENTACION = RAIZ.parents[2]
PRUEBAS = RAIZ.parents[1]
for ruta in (
    RAIZ / "codigo",
    BUSQUEDA / "codigo",
    IMPLEMENTACION,
    PRUEBAS / "codigo",
):
    sys.path.insert(0, str(ruta))

from busqueda_mejor.onmf_multicapa import deep_onmf_multicapa
from ultima_juan.configuraciones import (
    es_creciente,
    es_decreciente,
    invertir,
    seleccionar_diez,
)
from ultima_juan.informes import generar_pdf_comparacion, generar_pdf_completo


def test_inversion_de_arquitecturas() -> None:
    original = (15, 10, 5, 2)
    invertida = invertir(original)
    assert es_decreciente(original)
    assert es_creciente(invertida)
    assert invertida == (2, 5, 10, 15)


def test_onmf_admite_capas_crecientes() -> None:
    matriz = np.abs(np.random.default_rng(42).normal(size=(126, 212)))
    for rangos in ((4, 16), (7, 8, 9), (2, 5, 10, 15)):
        resultado = deep_onmf_multicapa(
            matriz,
            rangos=rangos,
            iteraciones=2,
            penalizacion_ortogonal=0.05,
            semilla=42,
        )
        assert resultado.w_final.shape == (126, rangos[-1])
        assert resultado.h_final.shape == (rangos[-1], 212)


def test_seleccion_incluye_referencia() -> None:
    filas = []
    for posicion in range(1, 13):
        distribucion = "9-8-7" if posicion == 12 else f"{20-posicion}-3"
        filas.append(
            {
                "posicion_resultado": posicion,
                "posicion_arquitectura": posicion,
                "distribucion": distribucion,
                "numero_capas": 3 if distribucion == "9-8-7" else 2,
                "clasificador": "UjaNet",
                "representacion": (
                    "DeepONMF_H3"
                    if distribucion == "9-8-7"
                    else "DeepONMF_H2"
                ),
                "Accuracy": 1.0 - posicion / 100,
                "Score": 0.9,
                "Sensitivity": 0.9,
                "Specificity": 0.9,
                "Precision": 0.9,
                "origen": "prueba",
                "mejor_resultado_arquitectura": True,
            }
        )
    seleccion = seleccionar_diez(pd.DataFrame(filas))
    assert len(seleccion) == 10
    assert "9-8-7" in set(seleccion["distribucion"])
    assert len(set(seleccion["distribucion_invertida"])) == 10


def test_generacion_de_los_dos_pdf() -> None:
    with tempfile.TemporaryDirectory() as temporal:
        carpeta = Path(temporal)
        png = carpeta / "matriz.png"
        figura, eje = plt.subplots(figsize=(2, 2))
        eje.imshow(np.eye(5))
        figura.savefig(png)
        plt.close(figura)

        seleccion = pd.DataFrame(
            [
                {
                    "posicion_seleccion": indice,
                    "posicion_resultado": indice,
                    "posicion_arquitectura": indice,
                    "distribucion": f"{20-indice}-3",
                    "distribucion_invertida": f"3-{20-indice}",
                    "numero_capas": 2,
                    "clasificador": "UjaNet",
                    "representacion": "DeepONMF_H2",
                    "Accuracy": 0.98,
                    "Score": 0.97,
                    "Sensitivity": 0.96,
                    "Specificity": 0.985,
                    "Precision": 0.965,
                    "origen": "prueba",
                    "forzada_como_referencia": False,
                    "mejor_resultado_arquitectura": True,
                }
                for indice in range(1, 11)
            ]
        )
        ranking = seleccion.drop(
            columns=["posicion_seleccion", "distribucion_invertida"]
        ).copy()
        ranking["seleccionada"] = True
        detalle_completo = pd.DataFrame(
            [
                {
                    "clasificador": "SVM",
                    "representacion": "DeepONMF_H2",
                    "distribucion": "16-4",
                    "numero_capas": 2,
                    "Score_mean": 0.95,
                    "Accuracy_mean": 0.97,
                    "Sensitivity_mean": 0.94,
                    "Specificity_mean": 0.98,
                    "Precision_mean": 0.95,
                    "origen": "prueba",
                }
            ]
        )
        profundidad = pd.DataFrame(
            [
                {
                    "numero_capas": 2,
                    "configuraciones": 10,
                    "accuracy_maxima": 0.99,
                    "score_maximo": 0.96,
                }
            ]
        )
        detalle = []
        diferencias = []
        matrices = []
        for pareja in range(1, 11):
            decreciente = f"{20-pareja}-3"
            creciente = f"3-{20-pareja}"
            for clasificador in ("SVM", "KNN", "UjaNet"):
                for tipo in ("W", "H"):
                    for sentido, distribucion in (
                        ("decreciente", decreciente),
                        ("creciente", creciente),
                    ):
                        detalle.append(
                            {
                                "pareja": pareja,
                                "sentido": sentido,
                                "clasificador": clasificador,
                                "tipo_matriz": tipo,
                                "representacion": (
                                    "DeepONMF_W"
                                    if tipo == "W"
                                    else "DeepONMF_H2"
                                ),
                                "distribucion": distribucion,
                                "Score_mean": 0.95,
                                "Accuracy_mean": 0.97,
                                "Sensitivity_mean": 0.94,
                                "Specificity_mean": 0.98,
                                "Precision_mean": 0.95,
                            }
                        )
                    diferencias.append(
                        {
                            "pareja": pareja,
                            "clasificador": clasificador,
                            "tipo_matriz": tipo,
                            "decreciente": decreciente,
                            "creciente": creciente,
                            "delta_Score": 0.0,
                            "delta_Accuracy": 0.0,
                            "delta_Sensitivity": 0.0,
                            "delta_Specificity": 0.0,
                            "delta_Precision": 0.0,
                        }
                    )
                for sentido, distribucion in (
                    ("decreciente", decreciente),
                    ("creciente", creciente),
                ):
                    matrices.append(
                        {
                            "pareja": pareja,
                            "sentido": sentido,
                            "distribucion": distribucion,
                            "clasificador": clasificador,
                            "ruta_png": str(png),
                            "mayor_error_real": "AS",
                            "mayor_error_predicha": "N",
                            "mayor_error_cantidad": 2,
                        }
                    )
        ruta_uno = generar_pdf_completo(
            carpeta / "uno.pdf",
            ranking,
            detalle_completo,
            profundidad,
            seleccion,
        )
        ruta_dos = generar_pdf_comparacion(
            carpeta / "dos.pdf",
            seleccion,
            pd.DataFrame(detalle),
            pd.DataFrame(diferencias),
            pd.DataFrame(matrices),
            ["Conclusion de prueba."],
        )
        for ruta in (ruta_uno, ruta_dos):
            documento = fitz.open(ruta)
            assert all(
                pagina.rect.width > pagina.rect.height
                for pagina in documento
            )
            if ruta == ruta_dos:
                texto = "\n".join(
                    pagina.get_text() for pagina in documento
                )
                assert len(documento) == 11
                assert sum(
                    len(pagina.get_images(full=True))
                    for pagina in documento
                ) == 20
                assert "SVM" not in texto
                assert "KNN" not in texto
                assert "DeepONMF_W" not in texto
            documento.close()
