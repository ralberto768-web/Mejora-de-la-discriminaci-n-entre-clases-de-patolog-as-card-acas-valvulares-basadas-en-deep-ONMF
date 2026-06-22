from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    destino = Path(__file__).resolve().parent / "verificar_todo.py"
    sys.argv[0] = str(destino)
    runpy.run_path(str(destino), run_name="__main__")
