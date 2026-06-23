from __future__ import annotations

import argparse
from pathlib import Path
import sys


RAIZ = Path(__file__).resolve().parent
RAIZ_BUSQUEDA = RAIZ.parent
RAIZ_PRUEBAS = RAIZ.parents[1]
RAIZ_IMPLEMENTACION = RAIZ.parents[2]
for ruta in (
    RAIZ / "codigo",
    RAIZ_BUSQUEDA / "codigo",
    RAIZ_IMPLEMENTACION,
    RAIZ_PRUEBAS / "codigo",
):
    sys.path.insert(0, str(ruta))

import pandas as pd  # noqa: E402

from codigo.clasificadores import crear_folds  # noqa: E402
from codigo.configuracion import cargar_configuracion  # noqa: E402
from codigo.datos import descubrir_audios  # noqa: E402
from ultima_juan.configuraciones import convertir_etiqueta  # noqa: E402
from ultima_juan.configuraciones import (  # noqa: E402
    construir_ranking,
    seleccionar_diez,
)
from ultima_juan.evaluacion import ejecutar_configuracion  # noqa: E402
from ultima_juan.flujo import CONFIGURACION_BASE, RESULTADOS_PUNTO3  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mitad",
        choices=("primera", "segunda", "todas"),
        default="todas",
    )
    args = parser.parse_args()
    carpeta_resultados = RAIZ / "resultados"
    resumen = pd.read_csv(
        carpeta_resultados
        / "tablas_csv"
        / "resumen_multiclase_todas_decrecientes.csv",
        encoding="utf-8-sig",
    )
    origenes = dict(
        zip(
            resumen["distribucion"].astype(str),
            resumen["origen"].astype(str),
        )
    )
    ranking = construir_ranking(resumen, origenes)
    seleccion = seleccionar_diez(ranking)
    ranking.to_csv(
        carpeta_resultados / "tablas_csv" / "ranking_completo.csv",
        index=False,
        encoding="utf-8-sig",
    )
    seleccion.to_csv(
        carpeta_resultados / "tablas_csv" / "diez_principales.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if args.mitad == "primera":
        seleccion = seleccion.iloc[:5]
    elif args.mitad == "segunda":
        seleccion = seleccion.iloc[5:]
    config = cargar_configuracion(CONFIGURACION_BASE)
    registros = descubrir_audios(RAIZ_IMPLEMENTACION.parent / "Bases de Datos")
    maestros = pd.read_csv(
        RESULTADOS_PUNTO3
        / "representaciones"
        / "DeepONMF_W"
        / "metadatos.csv",
        encoding="utf-8-sig",
    )
    y_multi = maestros["clase"].map(
        {"N": 0, "AS": 1, "MR": 2, "MS": 3, "MVP": 4}
    ).to_numpy(dtype=int)
    folds = crear_folds(y_multi, config)
    for valor in seleccion["distribucion_invertida"].astype(str):
        rangos = convertir_etiqueta(valor)
        ejecutar_configuracion(
            rangos,
            registros,
            maestros,
            folds,
            config,
            carpeta_resultados / "inc",
            rapido=False,
        )


if __name__ == "__main__":
    main()
