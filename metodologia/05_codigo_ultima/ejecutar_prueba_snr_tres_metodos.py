from __future__ import annotations

from pathlib import Path
import os
import sys


RAIZ = Path(__file__).resolve().parent
RAIZ_TFG = RAIZ.parents[1]
RUTAS_CODIGO = (
    RAIZ / "codigo",
    RAIZ_TFG / "Implementacion_last",
)

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
for ruta in reversed(RUTAS_CODIGO):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from ultima_final.prueba_snr_tres_metodos import main  # noqa: E402


if __name__ == "__main__":
    main()
