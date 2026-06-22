from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


CARPETA = Path(__file__).resolve().parent


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta preparacion, log-mel y LSTM.")
    parser.add_argument("--datos", type=Path, default=None)
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--duracion", type=float, default=2.0)
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--tamano-lote", type=int, default=64)
    parser.add_argument("--proporcion-entrenamiento", type=float, default=0.70)
    parser.add_argument("--limite-por-clase", type=int, default=0)
    parser.add_argument("--rapido", action="store_true")
    return parser.parse_args()


def resolver_datos(ruta: Path | None) -> Path:
    if ruta is not None:
        datos = ruta.expanduser().resolve()
        if datos.exists():
            return datos
        raise FileNotFoundError(f"No existe la carpeta de datos indicada: {datos}")
    for base in [Path.cwd(), CARPETA, *CARPETA.parents]:
        datos = base / "Bases de Datos"
        if datos.exists():
            return datos.resolve()
    raise FileNotFoundError("No se encontro una carpeta 'Bases de Datos'. Usa --datos RUTA.")


def ejecutar(script: str, argumentos: list[str], entorno: dict[str, str]) -> None:
    comando = [sys.executable, str(CARPETA / script), *argumentos]
    print("Ejecutando:", " ".join(comando))
    subprocess.run(comando, cwd=CARPETA, env=entorno, check=True)


def main() -> int:
    args = parsear_argumentos()
    datos = resolver_datos(args.datos)
    salida = (args.salida or (Path.cwd() / "resultados")).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)
    entorno = os.environ.copy()
    entorno["TFG_DATOS"] = str(datos)
    entorno["TFG_SALIDA"] = str(salida)

    epocas = 1 if args.rapido else args.epocas
    limite = args.limite_por_clase or (2 if args.rapido else 0)
    comunes = ["--duracion", str(args.duracion), "--semilla", str(args.semilla)]
    preparar = [*comunes]
    if limite > 0:
        preparar += ["--limite-por-clase", str(limite)]
    ejecutar("1-preparar_datos.py", preparar, entorno)
    ejecutar("2-extraer_espectrogramas.py", comunes, entorno)
    ejecutar(
        "4-entrenar_lstm.py",
        [
            *comunes,
            "--epocas",
            str(epocas),
            "--tamano-lote",
            str(args.tamano_lote),
            "--proporcion-entrenamiento",
            str(args.proporcion_entrenamiento),
        ],
        entorno,
    )
    print(f"Ejecucion LSTM completada. Resultados guardados en: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
