from __future__ import annotations

import argparse
from pathlib import Path
import sys


RAIZ = Path(__file__).resolve().parent
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

from excel_juan.flujo import ejecutar  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prueba completa de las arquitecturas del Excel de Juan."
    )
    parser.add_argument(
        "--datos",
        type=Path,
        default=RAIZ_IMPLEMENTACION.parent / "Bases de Datos",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=RAIZ / "resultados",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--solo-informe", action="store_true")
    args = parser.parse_args()
    pdfs = ejecutar(
        args.salida.resolve(),
        args.datos.resolve(),
        max(1, args.workers),
        args.solo_informe,
    )
    for ruta in pdfs:
        print(f"[fin] {ruta}")


if __name__ == "__main__":
    main()

