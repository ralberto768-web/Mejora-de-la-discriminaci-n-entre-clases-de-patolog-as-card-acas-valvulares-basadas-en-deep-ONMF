from __future__ import annotations

from pathlib import Path
import sys


RAIZ = Path(__file__).resolve().parent
IMPLEMENTACION = RAIZ.parents[1]
PRUEBAS_JUAN = RAIZ.parent
for ruta in (RAIZ / "codigo", IMPLEMENTACION, PRUEBAS_JUAN / "codigo"):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from busqueda_mejor.ejecucion import main


if __name__ == "__main__":
    main()

