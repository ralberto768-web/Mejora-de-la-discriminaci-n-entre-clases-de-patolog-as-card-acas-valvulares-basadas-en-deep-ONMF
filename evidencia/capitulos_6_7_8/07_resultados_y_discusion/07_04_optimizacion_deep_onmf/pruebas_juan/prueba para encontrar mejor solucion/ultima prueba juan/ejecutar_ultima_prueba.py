from __future__ import annotations

from pathlib import Path
import sys


RAIZ = Path(__file__).resolve().parent
RAIZ_BUSQUEDA = RAIZ.parent
RAIZ_PRUEBAS = RAIZ.parents[1]
RAIZ_IMPLEMENTACION = RAIZ.parents[2]

sys.path.insert(0, str(RAIZ / "codigo"))
sys.path.insert(1, str(RAIZ_BUSQUEDA / "codigo"))
sys.path.insert(2, str(RAIZ_IMPLEMENTACION))
sys.path.insert(3, str(RAIZ_PRUEBAS / "codigo"))

from ultima_juan.flujo import main  # noqa: E402


if __name__ == "__main__":
    main()
