from __future__ import annotations

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
from ultima_juan.evaluacion import ejecutar_configuracion  # noqa: E402
from ultima_juan.flujo import (  # noqa: E402
    CONFIGURACION_BASE,
    RESULTADOS_PUNTO3,
)


def main() -> None:
    carpeta_resultados = RAIZ / "resultados"
    plan = pd.read_csv(
        carpeta_resultados / "plan_arquitecturas_profundas.csv",
        encoding="utf-8-sig",
    )
    pendientes = plan[plan["estado_inicial"].eq("pendiente")].iloc[::-1]
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
    for fila in pendientes.itertuples(index=False):
        rangos = convertir_etiqueta(fila.distribucion)
        carpeta = carpeta_resultados / "dec" / "_".join(map(str, rangos))
        if (carpeta / "CONFIGURACION_COMPLETADA.txt").exists():
            continue
        ejecutar_configuracion(
            rangos,
            registros,
            maestros,
            folds,
            config,
            carpeta_resultados / "dec",
            rapido=False,
        )


if __name__ == "__main__":
    main()
