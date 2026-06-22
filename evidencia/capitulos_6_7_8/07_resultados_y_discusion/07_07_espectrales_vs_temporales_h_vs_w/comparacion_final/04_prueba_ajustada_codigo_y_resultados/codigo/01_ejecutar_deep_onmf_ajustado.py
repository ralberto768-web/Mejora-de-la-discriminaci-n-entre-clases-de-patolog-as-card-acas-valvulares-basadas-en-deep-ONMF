from __future__ import annotations

"""Ejecuta una configuracion ajustable de Deep ONMF sin eliminar audios."""

import argparse
from pathlib import Path
import sys


RAIZ_OBJETIVO = Path(__file__).resolve().parents[3]
SRC_OBJETIVO = RAIZ_OBJETIVO / "src"
if str(SRC_OBJETIVO) not in sys.path:
    sys.path.insert(0, str(SRC_OBJETIVO))

from tfg_deep_onmf.configuracion import Configuracion
from tfg_deep_onmf.pipeline import _ejecutar_articulo_deep_onmf


def parsear_rangos(texto: str) -> tuple[int, int, int]:
    partes = tuple(int(valor.strip()) for valor in texto.split(","))
    if len(partes) != 3 or min(partes) <= 0:
        raise argparse.ArgumentTypeError("Los rangos deben ser tres enteros positivos, por ejemplo 9,8,7.")
    return partes


def etiqueta_numero(valor: float) -> str:
    return f"{valor:g}".replace(".", "p")


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta Deep ONMF ajustable sin eliminar WAV.")
    parser.add_argument("--duracion", type=float, default=2.0, help="Duracion de la trama PCG en segundos.")
    parser.add_argument(
        "--solape",
        type=float,
        default=None,
        help="Solape entre tramas. Por defecto se usa la mitad de la duracion.",
    )
    parser.add_argument("--semilla", type=int, default=42, help="Semilla base de inicializacion ONMF.")
    parser.add_argument("--iteraciones", type=int, default=120, help="Iteraciones ONMF por capa.")
    parser.add_argument("--rangos", type=parsear_rangos, default=(9, 8, 7), help="Rangos capa1,capa2,capa3.")
    parser.add_argument("--penalizacion", type=float, default=0.05, help="Penalizacion ortogonal sobre H.")
    return parser.parse_args()


def main() -> int:
    args = parsear_argumentos()
    solape = args.solape if args.solape is not None else args.duracion / 2.0
    if solape < 0 or solape >= args.duracion:
        raise ValueError("El solape debe ser mayor o igual que 0 y menor que la duracion de trama.")
    configuracion = Configuracion(
        raiz=RAIZ_OBJETIVO,
        duracion_trama_s=args.duracion,
        solape_trama_s=solape,
        semilla=args.semilla,
        iteraciones_onmf=args.iteraciones,
        rangos_onmf=args.rangos,
        penalizacion_ortogonal=args.penalizacion,
        rellenar_audios_cortos=True,
    )
    etiqueta = (
        f"deep_onmf_ajustado_trama{etiqueta_numero(args.duracion)}s"
        f"_r{'-'.join(str(valor) for valor in args.rangos)}"
        f"_iter{args.iteraciones}_semilla{args.semilla}"
    )
    carpeta = _ejecutar_articulo_deep_onmf(
        configuracion,
        etiqueta=etiqueta,
        modo_json=etiqueta,
        mensaje_modo=(
            f"Modo Deep ONMF ajustado sin eliminar: trama={args.duracion:g} s, "
            f"solape={solape:g} s, rangos={args.rangos}, penalizacion={args.penalizacion:g}. "
            f"Semilla={args.semilla}; iteraciones={args.iteraciones}."
        ),
    )
    print(f"Resultado Deep ONMF ajustado generado en: {carpeta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
