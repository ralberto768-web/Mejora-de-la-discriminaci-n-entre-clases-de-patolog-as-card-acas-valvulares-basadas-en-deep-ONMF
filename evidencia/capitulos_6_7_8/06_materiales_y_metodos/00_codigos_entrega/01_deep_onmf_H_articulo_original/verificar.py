from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


CARPETA = Path(__file__).resolve().parent


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifica la implementacion H original con una ejecucion reducida.")
    parser.add_argument("--datos", type=Path, default=None)
    parser.add_argument("--limite-por-clase", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parsear_argumentos()
    salida = CARPETA / "resultados" / "verificacion_rapida"
    comando = [
        sys.executable,
        str(CARPETA / "ejecutar.py"),
        "--salida",
        str(salida),
        "--rapido",
        "--limite-por-clase",
        str(args.limite_por_clase),
    ]
    if args.datos is not None:
        comando += ["--datos", str(args.datos.expanduser().resolve())]
    subprocess.run(comando, check=True)
    validacion = pd.read_csv(salida / "validacion_rapida.csv", encoding="utf-8-sig")
    if not bool(validacion["ok"].all()):
        raise SystemExit("La validacion rapida contiene errores.")
    print(f"Verificacion OK: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
