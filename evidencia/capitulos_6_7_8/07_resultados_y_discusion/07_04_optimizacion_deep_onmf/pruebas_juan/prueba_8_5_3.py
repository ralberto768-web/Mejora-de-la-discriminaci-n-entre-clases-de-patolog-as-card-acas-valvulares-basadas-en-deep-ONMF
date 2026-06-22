from __future__ import annotations

from pathlib import Path
import sys


RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "codigo"))
sys.path.insert(1, str(RAIZ.parent))

from pruebas_juan.ejecucion import main_distribucion


if __name__ == "__main__":
    main_distribucion("8_5_3")

