from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import pandas as pd


CARPETA = Path(__file__).resolve().parent
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
if str(CARPETA) not in sys.path:
    sys.path.insert(0, str(CARPETA))

from codigo.clasificadores import (
    crear_folds,
    evaluar_clasificador_clasico,
    evaluar_ujanet,
    guardar_particiones,
    guardar_resumen_ejecucion,
    resumir_metricas,
)
from codigo.configuracion import (
    REPRESENTACIONES,
    aplicar_modo_rapido,
    cargar_configuracion,
    guardar_configuracion,
)
from codigo.datos import descubrir_audios, limitar_por_clase, tabla_auditoria
from codigo.informe import generar_informe_final
from codigo.metricas import comprobar_formulas_metricas
from codigo.representaciones import extraer_representaciones


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Punto 3 del TFG: validación automática de representaciones."
    )
    parser.add_argument(
        "--datos",
        type=Path,
        default=None,
        help="Ruta a la carpeta 'Bases de Datos'.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("resultados_punto3_validacion"),
        help="Carpeta donde se guardará la validación del punto 3.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CARPETA / "configuracion_experimento.json",
    )
    parser.add_argument(
        "--limite-por-clase",
        type=int,
        default=0,
        help="Límite equilibrado por clase para pruebas.",
    )
    parser.add_argument(
        "--rapido",
        action="store_true",
        help="Reduce iteraciones y épocas para comprobar el flujo.",
    )
    parser.add_argument(
        "--omitir-ujanet",
        action="store_true",
        help="Ejecuta únicamente SVM y KNN.",
    )
    return parser.parse_args()


def resolver_datos(ruta: Path | None) -> Path:
    if ruta is not None:
        datos = ruta.expanduser().resolve()
        if datos.exists():
            return datos
        raise FileNotFoundError(f"No existe la carpeta de datos indicada: {datos}")
    for base in [Path.cwd(), CARPETA, *CARPETA.parents]:
        candidata = base / "Bases de Datos"
        if candidata.exists():
            return candidata.resolve()
    raise FileNotFoundError("No se encontró la carpeta 'Bases de Datos'. Usa --datos RUTA.")


def validar_protocolo(
    auditoria: pd.DataFrame,
    folds: list[tuple[object, object]],
    metadatos: pd.DataFrame,
    salida: Path,
    exigir_completo: bool,
) -> pd.DataFrame:
    filas: list[dict[str, object]] = []
    total = int(auditoria["audios"].sum())
    filas.append(
        {
            "comprobacion": "total_audios",
            "esperado": 1000 if exigir_completo else "flexible",
            "observado": total,
            "ok": total == 1000 if exigir_completo else True,
        }
    )
    for _, fila in auditoria.iterrows():
        observado = int(fila["audios"])
        filas.append(
            {
                "comprobacion": f"audios_clase_{fila['clase']}",
                "esperado": 200 if exigir_completo else "flexible",
                "observado": observado,
                "ok": observado == 200 if exigir_completo else True,
            }
        )
    for numero_fold, (idx_train, idx_test) in enumerate(folds, start=1):
        filas.extend(
            [
                {
                    "comprobacion": f"fold_{numero_fold}_tamano_entrenamiento",
                    "esperado": 800 if exigir_completo else "flexible",
                    "observado": len(idx_train),
                    "ok": len(idx_train) == 800 if exigir_completo else True,
                },
                {
                    "comprobacion": f"fold_{numero_fold}_tamano_test",
                    "esperado": 200 if exigir_completo else "flexible",
                    "observado": len(idx_test),
                    "ok": len(idx_test) == 200 if exigir_completo else True,
                },
            ]
        )
        conteos = metadatos.iloc[idx_test]["clase"].value_counts()
        for clase in ("N", "AS", "MR", "MS", "MVP"):
            observado = int(conteos.get(clase, 0))
            filas.append(
                {
                    "comprobacion": f"fold_{numero_fold}_test_clase_{clase}",
                    "esperado": 40 if exigir_completo else "flexible",
                    "observado": observado,
                    "ok": observado == 40 if exigir_completo else True,
                }
            )
    validacion = pd.DataFrame(filas)
    validacion.to_csv(
        salida / "validacion_protocolo.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if exigir_completo and not bool(validacion["ok"].all()):
        raise RuntimeError(
            "La validación del protocolo completo ha fallado. "
            f"Revisa {salida / 'validacion_protocolo.csv'}"
        )
    return validacion


def main() -> int:
    inicio = time.perf_counter()
    args = parsear_argumentos()
    datos = resolver_datos(args.datos)
    salida = args.salida.expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)
    comprobar_formulas_metricas()

    config = cargar_configuracion(args.config)
    if args.rapido:
        config = aplicar_modo_rapido(config)
    guardar_configuracion(config, salida / "configuracion_usada.json")

    registros = descubrir_audios(datos)
    registros = limitar_por_clase(
        registros,
        args.limite_por_clase or (2 if args.rapido else 0),
    )
    if not registros:
        raise RuntimeError("No se encontraron audios WAV.")

    auditoria = tabla_auditoria(registros, config)
    auditoria.to_csv(
        salida / "auditoria_base_datos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[datos] audios seleccionados: {len(registros)}")

    representaciones = extraer_representaciones(registros, config, salida)
    y_multi = representaciones.metadatos["clase"].map(
        {"N": 0, "AS": 1, "MR": 2, "MS": 3, "MVP": 4}
    ).to_numpy()
    folds = crear_folds(y_multi, config)
    guardar_particiones(folds, representaciones.metadatos, salida)
    validar_protocolo(
        auditoria,
        folds,
        representaciones.metadatos,
        salida,
        exigir_completo=not args.rapido and args.limite_por_clase <= 0,
    )

    filas_binarias: list[dict[str, object]] = []
    filas_multiclase: list[dict[str, object]] = []
    for clasificador in ("SVM", "KNN"):
        binarias, multiclase = evaluar_clasificador_clasico(
            clasificador,
            representaciones.matrices,
            representaciones.metadatos,
            folds,
            config,
            salida / "clasificadores" / clasificador,
        )
        filas_binarias.extend(binarias)
        filas_multiclase.extend(multiclase)

    if not args.omitir_ujanet:
        binarias, multiclase = evaluar_ujanet(
            representaciones.matrices,
            representaciones.metadatos,
            folds,
            config,
            salida / "clasificadores",
        )
        filas_binarias.extend(binarias)
        filas_multiclase.extend(multiclase)

    resumen = resumir_metricas(
        filas_binarias,
        filas_multiclase,
        salida / "metricas",
    )
    clasificadores_ejecutados = 2 if args.omitir_ujanet else 3
    filas_esperadas = clasificadores_ejecutados * len(REPRESENTACIONES) * len(folds)
    if len(filas_binarias) != filas_esperadas or len(filas_multiclase) != filas_esperadas:
        raise RuntimeError(
            f"Número incorrecto de filas: binarias={len(filas_binarias)}, "
            f"multiclase={len(filas_multiclase)}, esperadas={filas_esperadas}."
        )

    markdown, pdf, figuras = generar_informe_final(
        config,
        auditoria,
        resumen.resumen_binario,
        resumen.resumen_multiclase,
        salida / "informe_punto3_validacion",
    )
    guardar_resumen_ejecucion(
        salida,
        {
            "audios": len(registros),
            "folds": len(folds),
            "modo_rapido": args.rapido,
            "ujanet_entrenada": not args.omitir_ujanet,
            "segundos_totales": time.perf_counter() - inicio,
            "tipo_entregable": "validacion_punto_3",
            "representaciones": list(REPRESENTACIONES),
            "combinaciones_clasificador_representacion": (
                clasificadores_ejecutados * len(REPRESENTACIONES)
            ),
            "filas_metricas_binarias": len(filas_binarias),
            "filas_metricas_multiclase": len(filas_multiclase),
            "informe_markdown": str(markdown),
            "informe_pdf": str(pdf),
            "figuras": [str(ruta) for ruta in figuras],
        },
    )
    print(f"[ok] Informe Markdown: {markdown}")
    print(f"[ok] Informe PDF: {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
