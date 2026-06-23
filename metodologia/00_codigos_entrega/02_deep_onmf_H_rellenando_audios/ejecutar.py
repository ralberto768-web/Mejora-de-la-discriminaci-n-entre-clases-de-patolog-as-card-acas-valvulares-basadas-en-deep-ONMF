from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA / "codigo"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(CARPETA / ".cache_matplotlib"))

from tfg_deep_onmf.audio import construir_matriz_clase, descubrir_audios
from tfg_deep_onmf.configuracion import Configuracion
from tfg_deep_onmf.onmf import deep_onmf
from tfg_deep_onmf.pipeline import ejecutar_articulo_sin_descartar


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-ONMF H rellenando audios menores de 2 s.")
    parser.add_argument("--datos", type=Path, default=None, help="Ruta a la carpeta Bases de Datos.")
    parser.add_argument("--salida", type=Path, default=None, help="Ruta donde se guardaran los resultados.")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--iteraciones", type=int, default=120)
    parser.add_argument("--limite-por-clase", type=int, default=0, help="Limite para prueba reducida.")
    parser.add_argument("--rapido", action="store_true", help="Ejecuta una prueba reducida de verificacion.")
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


def crear_configuracion(args: argparse.Namespace, datos: Path, salida: Path) -> Configuracion:
    iteraciones = 5 if args.rapido else args.iteraciones
    config = Configuracion(
        raiz=CARPETA,
        semilla=args.semilla,
        iteraciones_onmf=iteraciones,
        rellenar_audios_cortos=True,
    )
    object.__setattr__(config, "carpeta_base_datos", datos)
    object.__setattr__(config, "carpeta_resultados", salida)
    return config


def limitar_registros(registros: list, clases: tuple[str, ...], limite: int) -> list:
    if limite <= 0:
        return registros
    seleccionados = []
    conteos = {clase: 0 for clase in clases}
    for registro in sorted(registros, key=lambda r: (clases.index(r.clase), r.ruta.name)):
        if conteos[registro.clase] < limite:
            seleccionados.append(registro)
            conteos[registro.clase] += 1
    return seleccionados


def ejecutar_prueba_rapida(config: Configuracion, limite_por_clase: int, salida: Path) -> Path:
    salida.mkdir(parents=True, exist_ok=True)
    registros = descubrir_audios(config.carpeta_base_datos, config.clases)
    registros = limitar_registros(registros, config.clases, limite_por_clase or 2)
    if not registros:
        raise RuntimeError("No se encontraron WAV para la prueba rapida.")

    filas = []
    matrices_h = {}
    matrices_w = {}
    for posicion, clase in enumerate(config.clases, start=1):
        datos = construir_matriz_clase(clase, registros, config)
        resultado = deep_onmf(
            datos.matriz,
            rangos=config.rangos_onmf,
            iteraciones=config.iteraciones_onmf,
            penalizacion_ortogonal=config.penalizacion_ortogonal,
            semilla=config.semilla + posicion * 100,
        )
        matrices_h[f"H_{clase}"] = resultado.h_final
        matrices_w[f"W_{clase}"] = resultado.w_final
        filas.append(
            {
                "clase": clase,
                "audios_usados": len(datos.audios_usados),
                "audios_descartados": len(datos.audios_descartados),
                "forma_X": f"{datos.matriz.shape[0]}x{datos.matriz.shape[1]}",
                "forma_W": f"{resultado.w_final.shape[0]}x{resultado.w_final.shape[1]}",
                "forma_H": f"{resultado.h_final.shape[0]}x{resultado.h_final.shape[1]}",
                "error_relativo_final": resultado.error_relativo_final,
            }
        )

    np.savez(salida / "matrices_H_rapido.npz", **matrices_h)
    np.savez(salida / "matrices_W_rapido.npz", **matrices_w)
    resumen = pd.DataFrame(filas)
    resumen.to_csv(salida / "resumen_rapido.csv", index=False, encoding="utf-8-sig")
    (salida / "parametros_rapido.json").write_text(
        json.dumps(config.como_diccionario(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    validacion = pd.DataFrame(
        [
            {"comprobacion": "clases", "ok": len(resumen) == len(config.clases)},
            {"comprobacion": "sin_nan", "ok": not resumen.isna().any().any()},
            {"comprobacion": "H_7_filas", "ok": all(m.shape[0] == 7 for m in matrices_h.values())},
        ]
    )
    validacion.to_csv(salida / "validacion_rapida.csv", index=False, encoding="utf-8-sig")
    return salida


def main() -> int:
    args = parsear_argumentos()
    datos = resolver_datos(args.datos)
    salida = (args.salida or (Path.cwd() / "resultados")).expanduser().resolve()
    config = crear_configuracion(args, datos, salida)

    if args.rapido or args.limite_por_clase > 0:
        carpeta = ejecutar_prueba_rapida(config, args.limite_por_clase, salida)
    else:
        carpeta = ejecutar_articulo_sin_descartar(config)

    print(f"Ejecucion completada. Resultados guardados en: {carpeta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
