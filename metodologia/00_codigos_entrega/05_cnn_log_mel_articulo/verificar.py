from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CARPETA = Path(__file__).resolve().parent


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifica CNN log-mel con ejecucion reducida.")
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
    esperados = [
        salida / "caracteristicas_log_mel_2_0s.npz",
        salida / "3-metricas_cnn_2_0s_entrenamiento_70_prueba_30.txt",
    ]
    faltantes = [str(ruta) for ruta in esperados if not ruta.exists()]
    if faltantes:
        raise SystemExit("Faltan salidas esperadas: " + "; ".join(faltantes))
    print(f"Verificacion OK: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
