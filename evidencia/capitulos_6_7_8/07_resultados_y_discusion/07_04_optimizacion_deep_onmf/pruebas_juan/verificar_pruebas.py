from __future__ import annotations

from pathlib import Path
import sys


RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "codigo"))
sys.path.insert(1, str(RAIZ.parent))

from pruebas_juan.ejecucion import ejecutar_distribuciones


if __name__ == "__main__":
    ejecutar_distribuciones(
        ["15_10_5", "10_6_4", "8_5_3"],
        RAIZ.parent.parent / "Bases de Datos",
        RAIZ / "resultados_verificacion",
        rapido=True,
        limite_por_clase=2,
        reutilizar=True,
    )

