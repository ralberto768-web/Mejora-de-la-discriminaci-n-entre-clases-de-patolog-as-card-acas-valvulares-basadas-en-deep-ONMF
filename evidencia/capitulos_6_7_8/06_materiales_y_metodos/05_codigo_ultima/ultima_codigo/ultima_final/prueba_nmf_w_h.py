from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import fitz
import numpy as np
import pandas as pd

from codigo.clasificadores import guardar_particiones
from codigo.configuracion import cargar_configuracion
from codigo.datos import descubrir_audios

from .arquitecturas import clave, etiqueta
from .configuracion import BASE_ORIGINAL, BASE_SNR0
from .documentos import ANCHO, ALTO, AZUL, BLANCO, MARGEN, _portada, _tabla, _texto
from .evaluacion import evaluar_ujanet_multiclase, folds_desde_metadatos
from .extraccion import _manifest_valido, extraer_clasicas
from .onmf_multicapa import deep_onmf_multicapa


RAIZ = Path(__file__).resolve().parents[2]
RAIZ_TFG = RAIZ.parents[1]
DATOS_DEFECTO = RAIZ / "datasets_ruidosos"
CONFIG_DEFECTO = RAIZ_TFG / "Implementacion_last" / "configuracion_experimento.json"
SALIDA = RAIZ / "05_prueba_nmf_w_h"
PROTOCOLO_PRUEBA = "prueba_nmf_w_h_v1"


def _cargar_optimas(ruta: Path) -> dict[int, tuple[int, ...]]:
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return {int(capas): tuple(int(valor) for valor in rangos) for capas, rangos in datos.items()}


def _penalizacion(metodo: str, config) -> float:
    if metodo == "ONMF":
        return float(config.penalizacion_ortogonal)
    if metodo == "NMF":
        return 0.0
    raise ValueError(f"Metodo no soportado: {metodo}")


def _extraer_wh(
    base: str,
    registros,
    config,
    rangos: tuple[int, ...],
    metodo: str,
    carpeta_cache_clasicas: Path,
    carpeta_salida: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    distribucion = etiqueta(rangos)
    carpeta = carpeta_salida / "representaciones" / base / metodo.lower() / clave(rangos)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_h = carpeta / f"{metodo}_H{len(rangos)}.npy"
    ruta_w = carpeta / f"{metodo}_W{len(rangos)}.npy"
    ruta_meta = carpeta / "metadatos.csv"
    ruta_manifest = carpeta / "manifest.json"
    manifest = {
        "protocolo": PROTOCOLO_PRUEBA,
        "base": base,
        "metodo": metodo,
        "numero_audios": len(registros),
        "distribucion": distribucion,
        "numero_capas": len(rangos),
        "inicializacion": "nndsvd",
        "iteraciones": config.iteraciones_onmf,
        "penalizacion_ortogonal": _penalizacion(metodo, config),
        "matrices_guardadas": ["H", "W"],
        "semilla": config.semilla,
    }
    if (
        _manifest_valido(ruta_manifest, manifest)
        and ruta_h.exists()
        and ruta_w.exists()
        and ruta_meta.exists()
    ):
        return {
            "H": np.load(ruta_h, mmap_mode="r"),
            "W": np.load(ruta_w, mmap_mode="r"),
        }, pd.read_csv(ruta_meta, encoding="utf-8-sig")

    clasicas, metadatos = extraer_clasicas(base, registros, config, carpeta_cache_clasicas)
    stfts = np.asarray(clasicas["STFT"], dtype=np.float32)
    matrices_h: list[np.ndarray] = []
    matrices_w: list[np.ndarray] = []
    auditoria: list[dict[str, object]] = []
    penalizacion = _penalizacion(metodo, config)

    for indice, stft in enumerate(stfts, start=1):
        inicio = time.perf_counter()
        resultado = deep_onmf_multicapa(
            stft,
            rangos=rangos,
            iteraciones=config.iteraciones_onmf,
            penalizacion_ortogonal=penalizacion,
            semilla=config.semilla + indice * 37,
        )
        matrices_h.append(resultado.h_final.astype(np.float32))
        matrices_w.append(resultado.w_final.astype(np.float32))
        auditoria.append(
            {
                "indice_interno": indice - 1,
                "base": base,
                "metodo": metodo,
                "distribucion": distribucion,
                "numero_capas": len(rangos),
                "penalizacion_ortogonal": penalizacion,
                "error_final": resultado.error_relativo_final,
                "forma_h": f"{resultado.h_final.shape[0]}x{resultado.h_final.shape[1]}",
                "forma_w": f"{resultado.w_final.shape[0]}x{resultado.w_final.shape[1]}",
                "segundos": time.perf_counter() - inicio,
            }
        )
        if indice == 1 or indice % 100 == 0 or indice == len(stfts):
            print(f"[{metodo} {base} {distribucion}] {indice}/{len(stfts)}")

    matriz_h = np.stack(matrices_h).astype(np.float32)
    matriz_w = np.stack(matrices_w).astype(np.float32)
    np.save(ruta_h, matriz_h)
    np.save(ruta_w, matriz_w)
    metadatos.to_csv(ruta_meta, index=False, encoding="utf-8-sig")
    pd.DataFrame(auditoria).to_csv(carpeta / "auditoria_wh.csv", index=False, encoding="utf-8-sig")
    ruta_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"H": matriz_h, "W": matriz_w}, metadatos


def _porcentaje(valor: object) -> str:
    return f"{float(valor) * 100:.2f}%"


def _generar_pdf(tabla: pd.DataFrame, ruta_pdf: Path) -> None:
    documento = fitz.open()
    _portada(
        documento,
        "Prueba NMF Estandar y Matrices W",
        "Comparacion adicional solicitada: NMF sin regularizacion frente a ONMF, usando H y W como entrada de UjaNet.",
    )
    pagina = documento.new_page(width=ANCHO, height=ALTO)
    pagina.draw_rect(fitz.Rect(0, 0, ANCHO, 82), color=AZUL, fill=AZUL)
    _texto(pagina, fitz.Rect(MARGEN, 22, ANCHO - MARGEN, 60), "Resumen", 18, True, BLANCO)
    texto = (
        "La prueba usa las configuraciones optimas ya obtenidas en la optimizacion final. "
        "Para cada dataset se evalua ONMF con la penalizacion ortogonal original y NMF estandar "
        "con penalizacion 0. En ambos casos la inicializacion es NNDSVD y se alimenta UjaNet "
        "con la matriz H final y con la matriz W final."
    )
    _texto(pagina, fitz.Rect(MARGEN, 110, ANCHO - MARGEN, 190), texto, 10)

    mostrar = tabla.copy()
    for columna in ["Accuracy_mean", "Sensitivity_mean", "Specificity_mean", "Precision_mean", "Score_mean"]:
        mostrar[columna] = mostrar[columna].map(_porcentaje)
    columnas = [
        ("base", "Base", 1.0),
        ("metodo_factorizacion", "Metodo", 0.8),
        ("matriz_entrada", "Matriz", 0.55),
        ("numero_capas", "Capas", 0.55),
        ("distribucion", "Distrib.", 1.0),
        ("Accuracy_mean", "Acc.", 0.75),
        ("Score_mean", "Score", 0.75),
        ("Sensitivity_mean", "Sens.", 0.75),
        ("Precision_mean", "Prec.", 0.75),
    ]
    _tabla(documento, "Resultados completos", mostrar[columnas and [c[0] for c in columnas]], columnas, filas_por_pagina=20)

    resumen = (
        tabla.pivot_table(
            index=["base", "numero_capas"],
            columns=["metodo_factorizacion", "matriz_entrada"],
            values="Accuracy_mean",
            aggfunc="first",
        )
        .reset_index()
    )
    resumen.columns = [
        "_".join(str(parte) for parte in columna if str(parte))
        if isinstance(columna, tuple)
        else str(columna)
        for columna in resumen.columns
    ]
    for columna in resumen.columns:
        if columna not in {"base", "numero_capas"}:
            resumen[columna] = resumen[columna].map(_porcentaje)
    columnas_resumen = [
        ("base", "Base", 1.0),
        ("numero_capas", "Capas", 0.5),
        ("NMF_H", "NMF H", 0.7),
        ("NMF_W", "NMF W", 0.7),
        ("ONMF_H", "ONMF H", 0.7),
        ("ONMF_W", "ONMF W", 0.7),
    ]
    columnas_resumen = [columna for columna in columnas_resumen if columna[0] in resumen.columns]
    _tabla(documento, "Resumen por accuracy", resumen[[c[0] for c in columnas_resumen]], columnas_resumen, filas_por_pagina=18)

    ruta_pdf.parent.mkdir(parents=True, exist_ok=True)
    documento.save(ruta_pdf)
    documento.close()


def ejecutar(datos: Path, config_path: Path) -> None:
    inicio = time.perf_counter()
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "tablas_csv").mkdir(parents=True, exist_ok=True)
    config = cargar_configuracion(config_path)

    escenarios = [
        {
            "base": BASE_ORIGINAL,
            "datos": datos / BASE_ORIGINAL,
            "cache_clasicas": RAIZ / "01_optimizacion_dataset_original",
            "optimas": _cargar_optimas(RAIZ / "01_optimizacion_dataset_original" / "configuraciones_optimas.json"),
        },
        {
            "base": BASE_SNR0,
            "datos": datos / BASE_SNR0,
            "cache_clasicas": RAIZ / "02_optimizacion_dataset_SNR0db",
            "optimas": _cargar_optimas(RAIZ / "02_optimizacion_dataset_SNR0db" / "configuraciones_optimas.json"),
        },
    ]

    ruta_resultados = SALIDA / "tablas_csv" / "resultados_nmf_w_h.csv"
    filas: list[dict[str, object]] = []
    if ruta_resultados.exists():
        existente = pd.read_csv(ruta_resultados, encoding="utf-8-sig")
        if "protocolo_prueba" in existente.columns and existente["protocolo_prueba"].eq(PROTOCOLO_PRUEBA).all():
            filas = existente.to_dict(orient="records")
    hechas = {
        (
            str(fila["base"]),
            str(fila["metodo_factorizacion"]),
            str(fila["matriz_entrada"]),
            str(fila["distribucion"]),
        )
        for fila in filas
    }

    for escenario in escenarios:
        base = escenario["base"]
        registros = descubrir_audios(escenario["datos"])
        if not registros:
            raise RuntimeError(f"No se encontraron audios en {escenario['datos']}")
        _, metadatos = extraer_clasicas(base, registros, config, escenario["cache_clasicas"])
        folds = folds_desde_metadatos(metadatos, config)
        guardar_particiones(folds, metadatos, SALIDA / "particiones" / base)

        for numero_capas, rangos in sorted(escenario["optimas"].items()):
            distribucion = etiqueta(rangos)
            for metodo in ("ONMF", "NMF"):
                matrices, meta_factorizacion = _extraer_wh(
                    base=base,
                    registros=registros,
                    config=config,
                    rangos=rangos,
                    metodo=metodo,
                    carpeta_cache_clasicas=escenario["cache_clasicas"],
                    carpeta_salida=SALIDA,
                )
                for matriz_entrada in ("H", "W"):
                    clave_hecha = (base, metodo, matriz_entrada, distribucion)
                    if clave_hecha in hechas:
                        continue
                    representacion = f"{metodo}_{matriz_entrada}{numero_capas}"
                    print(f"[prueba {base}] {representacion} {distribucion}")
                    resumen = evaluar_ujanet_multiclase(
                        base=base,
                        representacion=representacion,
                        x=np.asarray(matrices[matriz_entrada], dtype=np.float32),
                        metadatos=meta_factorizacion,
                        folds=folds,
                        config=config,
                        carpeta=SALIDA / "evaluacion_ujanet" / base / metodo / f"{matriz_entrada}{numero_capas}_{clave(rangos)}",
                        distribucion=distribucion,
                        numero_capas=numero_capas,
                        inicializacion_onmf="nndsvd",
                        entrada_ujanet=f"matriz_{matriz_entrada}_final_{metodo}",
                    )
                    resumen.update(
                        {
                            "protocolo_prueba": PROTOCOLO_PRUEBA,
                            "metodo_factorizacion": metodo,
                            "matriz_entrada": matriz_entrada,
                            "distribucion": distribucion,
                            "numero_capas": numero_capas,
                            "penalizacion_ortogonal": _penalizacion(metodo, config),
                            "sin_regularizacion": metodo == "NMF",
                            "inicializacion_factorizacion": "nndsvd",
                        }
                    )
                    filas.append(resumen)
                    pd.DataFrame(filas).to_csv(ruta_resultados, index=False, encoding="utf-8-sig")
                    hechas.add(clave_hecha)

    tabla = pd.DataFrame(filas).sort_values(
        ["base", "numero_capas", "metodo_factorizacion", "matriz_entrada"],
        kind="mergesort",
    )
    tabla.to_csv(ruta_resultados, index=False, encoding="utf-8-sig")

    resumen = (
        tabla.pivot_table(
            index=["base", "numero_capas", "distribucion"],
            columns=["metodo_factorizacion", "matriz_entrada"],
            values="Accuracy_mean",
            aggfunc="first",
        )
        .reset_index()
    )
    resumen.to_csv(SALIDA / "tablas_csv" / "resumen_accuracy_nmf_w_h.csv", index=False, encoding="utf-8-sig")
    _generar_pdf(tabla, SALIDA / "prueba_nmf_w_h.pdf")
    (SALIDA / "resumen_ejecucion.json").write_text(
        json.dumps(
            {
                "protocolo": PROTOCOLO_PRUEBA,
                "filas": len(tabla),
                "bases": [escenario["base"] for escenario in escenarios],
                "segundos": time.perf_counter() - inicio,
                "salida": str(SALIDA.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[fin prueba nmf/w/h] {SALIDA}")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prueba NMF sin regularizacion y matrices W/H.")
    parser.add_argument("--datos", type=Path, default=DATOS_DEFECTO)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFECTO)
    return parser


def main() -> None:
    args = construir_parser().parse_args()
    ejecutar(args.datos, args.config)
