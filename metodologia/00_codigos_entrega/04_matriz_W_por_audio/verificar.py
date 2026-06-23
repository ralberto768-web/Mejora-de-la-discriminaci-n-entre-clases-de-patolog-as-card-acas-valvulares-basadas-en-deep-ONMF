from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


CARPETA = Path(__file__).resolve().parent


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifica la implementacion W por audio con una ejecucion reducida.")
    parser.add_argument("--datos", type=Path, default=None)
    parser.add_argument("--limite-por-clase", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parsear_argumentos()
    salida = CARPETA / "resultados" / "verificacion_rapida"
    limite = max(args.limite_por_clase, 4)
    comando = [
        sys.executable,
        str(CARPETA / "ejecutar.py"),
        "--salida",
        str(salida),
        "--rapido",
        "--limite-por-clase",
        str(limite),
    ]
    if args.datos is not None:
        comando += ["--datos", str(args.datos.expanduser().resolve())]
    subprocess.run(comando, check=True)
    validacion = pd.read_csv(salida / "Fotos datos y graficos" / "validacion_salidas.csv", encoding="utf-8-sig")
    if not bool(validacion["ok"].all()):
        raise SystemExit("La validacion rapida contiene errores.")
    print(f"Verificacion OK: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
