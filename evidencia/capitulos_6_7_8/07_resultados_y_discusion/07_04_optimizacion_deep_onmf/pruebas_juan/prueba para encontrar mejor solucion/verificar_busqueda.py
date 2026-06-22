from __future__ import annotations

from pathlib import Path
import subprocess
import sys


RAIZ = Path(__file__).resolve().parent


def main() -> None:
    comando_pruebas = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(RAIZ / "tests"),
        "-v",
    ]
    subprocess.run(comando_pruebas, check=True, cwd=RAIZ)
    comando_rapido = [
        sys.executable,
        str(RAIZ / "ejecutar_busqueda_completa.py"),
        "--rapido",
        "--limite-por-clase",
        "2",
    ]
    subprocess.run(comando_rapido, check=True, cwd=RAIZ)
    print("Verificacion tecnica completada.")


if __name__ == "__main__":
    main()
