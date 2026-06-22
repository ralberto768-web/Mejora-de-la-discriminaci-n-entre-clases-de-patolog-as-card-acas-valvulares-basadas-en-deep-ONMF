from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

CARPETA = Path(__file__).resolve().parent


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verificación rápida del punto 3.")
    parser.add_argument("--datos", type=Path, default=None)
    parser.add_argument("--limite-por-clase", type=int, default=2)
    parser.add_argument("--rapido", action="store_true")
    return parser.parse_args()


def resolver_datos(ruta: Path | None) -> Path:
    if ruta is not None:
        return ruta.expanduser().resolve()
    for base in [Path.cwd(), CARPETA, *CARPETA.parents]:
        candidata = base / "Bases de Datos"
        if candidata.exists():
            return candidata.resolve()
    raise FileNotFoundError("No se encontro la carpeta 'Bases de Datos'.")


def main() -> int:
    args = parsear_argumentos()
    datos = resolver_datos(args.datos)
    salida = CARPETA / "resultados_verificacion"
    comando = [
        sys.executable,
        str(CARPETA / "ejecutar.py"),
        "--datos",
        str(datos),
        "--salida",
        str(salida),
        "--rapido",
        "--limite-por-clase",
        str(args.limite_por_clase),
    ]
    print("[verificacion] Ejecutando flujo reducido...", flush=True)
    subprocess.run(comando, cwd=CARPETA, check=True)
    esperados = [
        salida / "auditoria_base_datos.csv",
        salida / "validacion_protocolo.csv",
        salida / "metricas" / "resumen_metricas_binarias.csv",
        salida / "metricas" / "resumen_metricas_multiclase.csv",
        salida / "informe_punto3_validacion" / "informe_punto3_validacion.md",
        salida / "informe_punto3_validacion" / "informe_punto3_validacion.pdf",
    ]
    faltan = [ruta for ruta in esperados if not ruta.exists()]
    if faltan:
        for ruta in faltan:
            print(f"[error] Falta salida esperada: {ruta}")
        return 1
    print("[ok] Verificacion completada. Salidas principales generadas correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
