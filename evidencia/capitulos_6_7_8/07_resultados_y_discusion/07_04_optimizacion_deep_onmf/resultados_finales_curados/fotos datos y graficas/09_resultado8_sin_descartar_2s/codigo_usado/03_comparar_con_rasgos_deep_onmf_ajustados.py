from __future__ import annotations

"""Compara el CSV de rasgos Deep ONMF ajustados con Figura 11."""

from pathlib import Path
import subprocess
import sys


CARPETA_COMPARACION = Path(__file__).resolve().parents[2]
RAIZ_OBJETIVO = CARPETA_COMPARACION.parent
RUTA_COMPARADOR = CARPETA_COMPARACION / "01_codigos_ordenados" / "01_comparar_figura11.py"


def encontrar_rasgos_ajustados() -> Path:
    ruta = CARPETA_COMPARACION / "04_prueba_ajustada_codigo_y_resultados" / "resultados" / "mejores_rasgos_deep_onmf.csv"
    if not ruta.exists():
        raise FileNotFoundError("No existen mejores rasgos ajustados. Ejecuta antes el barrido Deep ONMF.")
    return ruta


def main() -> int:
    ruta_rasgos = encontrar_rasgos_ajustados()
    comando = [
        sys.executable,
        str(RUTA_COMPARADOR),
        "--ruta-rasgos-deep-onmf",
        str(ruta_rasgos),
        *sys.argv[1:],
    ]
    print(f"Comparando rasgos Deep ONMF ajustados: {ruta_rasgos}")
    return subprocess.run(comando, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
