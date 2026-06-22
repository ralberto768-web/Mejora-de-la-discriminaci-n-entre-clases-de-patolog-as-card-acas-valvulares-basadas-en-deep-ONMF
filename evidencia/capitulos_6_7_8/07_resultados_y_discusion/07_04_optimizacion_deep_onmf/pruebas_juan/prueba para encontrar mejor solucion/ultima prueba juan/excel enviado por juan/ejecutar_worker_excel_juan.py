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

from excel_juan.flujo import ejecutar_worker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indice", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--datos", type=Path, required=True)
    args = parser.parse_args()
    ejecutar_worker(
        args.indice,
        args.workers,
        args.salida.resolve(),
        args.datos.resolve(),
    )


if __name__ == "__main__":
    main()

