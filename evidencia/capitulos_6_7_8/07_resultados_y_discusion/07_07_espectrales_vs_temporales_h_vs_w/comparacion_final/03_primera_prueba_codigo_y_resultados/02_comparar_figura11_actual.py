from __future__ import annotations

"""Wrapper de la comparacion base de Figura 11 con informe detallado."""

from pathlib import Path
import runpy


RUTA_COMPARACION = Path(__file__).resolve().parents[1] / "01_codigos_ordenados" / "01_comparar_figura11.py"


if __name__ == "__main__":
    runpy.run_path(str(RUTA_COMPARACION), run_name="__main__")
