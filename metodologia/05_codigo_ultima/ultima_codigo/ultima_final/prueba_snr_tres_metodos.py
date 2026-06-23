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
from .configuracion import SNR_OBJETIVOS, nombre_base_snr
from .documentos import ANCHO, ALTO, AZUL, BLANCO, MARGEN, _portada, _tabla, _texto
from .evaluacion import evaluar_ujanet_multiclase, folds_desde_metadatos
from .extraccion import _manifest_valido, extraer_clasicas
from .onmf_multicapa import deep_onmf_multicapa


RAIZ = Path(__file__).resolve().parents[2]
RAIZ_TFG = RAIZ.parents[1]
DATOS_DEFECTO = RAIZ / "datasets_ruidosos"
CONFIG_DEFECTO = RAIZ_TFG / "Implementacion_last" / "configuracion_experimento.json"
SALIDA = RAIZ / "06_prueba_snr_tres_metodos"
PROTOCOLO_PRUEBA = "prueba_snr_tres_metodos_v1"


METODOS = (
    {
        "metodo": "NMF",
        "descripcion": "NMF estandar sin Deep, 7 bases, sin regularizacion",
        "rangos": (7,),
        "penalizacion": 0.0,
        "sin_regularizacion": True,
    },
    {
        "metodo": "ONMF",
        "descripcion": "ONMF estandar sin Deep, 7 bases",
        "rangos": (7,),
        "penalizacion": None,
        "sin_regularizacion": False,
    },
    {
        "metodo": "DeepONMF",
        "descripcion": "Deep-ONMF con configuracion 9-8-7",
        "rangos": (9, 8, 7),
        "penalizacion": None,
        "sin_regularizacion": False,
    },
)


def _penalizacion(definicion: dict[str, object], config) -> float:
    valor = definicion["penalizacion"]
    return float(config.penalizacion_ortogonal if valor is None else valor)


def _extraer_wh(
    base: str,
    registros,
    config,
    definicion: dict[str, object],
    carpeta_cache_clasicas: Path,
    carpeta_salida: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    metodo = str(definicion["metodo"])
    rangos = tuple(int(valor) for valor in definicion["rangos"])
    distribucion = etiqueta(rangos)
    carpeta = carpeta_salida / "representaciones" / base / metodo.lower() / clave(rangos)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_h = carpeta / f"{metodo}_H.npy"
    ruta_w = carpeta / f"{metodo}_W.npy"
    ruta_meta = carpeta / "metadatos.csv"
    ruta_manifest = carpeta / "manifest.json"
    penalizacion = _penalizacion(definicion, config)
    manifest = {
        "protocolo": PROTOCOLO_PRUEBA,
        "base": base,
        "metodo": metodo,
        "descripcion": str(definicion["descripcion"]),
        "numero_audios": len(registros),
        "distribucion": distribucion,
        "numero_capas": len(rangos),
        "inicializacion": "nndsvd",
        "iteraciones": config.iteraciones_onmf,
        "penalizacion_ortogonal": penalizacion,
        "sin_regularizacion": bool(definicion["sin_regularizacion"]),
        "matrices_guardadas": ["H", "W"],
        "semilla": config.semilla,
    }
    if (
        _manifest_valido(ruta_manifest, manifest)
        and ruta_h.exists()
        and ruta_w.exists()
        and ruta_meta.exists()
    ):
        return {"H": np.load(ruta_h, mmap_mode="r"), "W": np.load(ruta_w, mmap_mode="r")}, pd.read_csv(
            ruta_meta,
            encoding="utf-8-sig",
        )

    clasicas, metadatos = extraer_clasicas(base, registros, config, carpeta_cache_clasicas)
    stfts = np.asarray(clasicas["STFT"], dtype=np.float32)
    matrices_h: list[np.ndarray] = []
    matrices_w: list[np.ndarray] = []
    auditoria: list[dict[str, object]] = []

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
                "sin_regularizacion": bool(definicion["sin_regularizacion"]),
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
    pd.DataFrame(auditoria).to_csv(carpeta / "auditoria_por_senal.csv", index=False, encoding="utf-8-sig")
    ruta_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"H": matriz_h, "W": matriz_w}, metadatos


def _porcentaje(valor: object) -> str:
    return f"{float(valor) * 100:.2f}%"


def _generar_pdf(tabla: pd.DataFrame, ruta_pdf: Path) -> None:
    documento = fitz.open()
    _portada(
        documento,
        "Prueba SNR: NMF, ONMF y Deep-ONMF",
        "Evaluacion por SNR de NMF-7, ONMF-7 y Deep-ONMF 9-8-7. Se guardan H y W por senal y UjaNet recibe la matriz seleccionada.",
    )
    pagina = documento.new_page(width=ANCHO, height=ALTO)
    pagina.draw_rect(fitz.Rect(0, 0, ANCHO, 82), color=AZUL, fill=AZUL)
    _texto(pagina, fitz.Rect(MARGEN, 22, ANCHO - MARGEN, 60), "Protocolo", 18, True, BLANCO)
    _texto(
        pagina,
        fitz.Rect(MARGEN, 105, ANCHO - MARGEN, 210),
        "Cada audio se factoriza de forma independiente. No se entrena un diccionario W comun por fold. "
        "Despues se apilan las matrices H o W por senal y ese tensor es la entrada de UjaNet. "
        "Los folds solo se usan para entrenar/evaluar la CNN sobre las representaciones ya calculadas.",
        10,
    )

    mostrar = tabla.copy()
    for columna in ["Accuracy_mean", "Sensitivity_mean", "Specificity_mean", "Precision_mean", "Score_mean"]:
        mostrar[columna] = mostrar[columna].map(_porcentaje)
    columnas = [
        ("snr_db", "SNR", 0.45),
        ("metodo_factorizacion", "Metodo", 0.9),
        ("matriz_entrada", "Matriz", 0.5),
        ("distribucion", "Bases", 0.8),
        ("Accuracy_mean", "Acc.", 0.65),
        ("Score_mean", "Score", 0.65),
        ("Sensitivity_mean", "Sens.", 0.65),
        ("Precision_mean", "Prec.", 0.65),
    ]
    _tabla(documento, "Resultados completos", mostrar[[c[0] for c in columnas]], columnas, filas_por_pagina=20)

    solo_h = tabla.loc[tabla["matriz_entrada"].eq("H")].copy()
    pivot_h = solo_h.pivot_table(
        index="snr_db",
        columns="metodo_factorizacion",
        values="Accuracy_mean",
        aggfunc="first",
    ).reset_index()
    pivot_h = pivot_h.sort_values("snr_db", ascending=False)
    for columna in pivot_h.columns:
        if columna != "snr_db":
            pivot_h[columna] = pivot_h[columna].map(_porcentaje)
    columnas_h = [
        ("snr_db", "SNR", 0.5),
        ("NMF", "NMF H", 0.8),
        ("ONMF", "ONMF H", 0.8),
        ("DeepONMF", "Deep H", 0.8),
    ]
    columnas_h = [columna for columna in columnas_h if columna[0] in pivot_h.columns]
    _tabla(documento, "Tabla principal: Accuracy usando H", pivot_h[[c[0] for c in columnas_h]], columnas_h, filas_por_pagina=18)

    ruta_pdf.parent.mkdir(parents=True, exist_ok=True)
    documento.save(ruta_pdf)
    documento.close()


def ejecutar(datos: Path, config_path: Path) -> None:
    inicio = time.perf_counter()
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "tablas_csv").mkdir(parents=True, exist_ok=True)
    config = cargar_configuracion(config_path)
    carpeta_cache_clasicas = RAIZ / "03_resultados_optimizacion_original"
    ruta_resultados = SALIDA / "tablas_csv" / "resultados_snr_tres_metodos.csv"

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
        )
        for fila in filas
    }

    for snr_db in SNR_OBJETIVOS:
        base = nombre_base_snr(snr_db)
        carpeta_datos = datos / base
        registros = descubrir_audios(carpeta_datos)
        if not registros:
            raise RuntimeError(f"No se encontraron audios en {carpeta_datos}")
        _, metadatos = extraer_clasicas(base, registros, config, carpeta_cache_clasicas)
        folds = folds_desde_metadatos(metadatos, config)
        guardar_particiones(folds, metadatos, SALIDA / "particiones" / base)

        for definicion in METODOS:
            metodo = str(definicion["metodo"])
            rangos = tuple(int(valor) for valor in definicion["rangos"])
            distribucion = etiqueta(rangos)
            matrices, meta_factorizacion = _extraer_wh(
                base=base,
                registros=registros,
                config=config,
                definicion=definicion,
                carpeta_cache_clasicas=carpeta_cache_clasicas,
                carpeta_salida=SALIDA,
            )
            for matriz_entrada in ("H", "W"):
                clave_hecha = (base, metodo, matriz_entrada)
                if clave_hecha in hechas:
                    continue
                representacion = f"{metodo}_{matriz_entrada}_{distribucion}"
                print(f"[SNR {snr_db}] {representacion}")
                resumen = evaluar_ujanet_multiclase(
                    base=base,
                    representacion=representacion,
                    x=np.asarray(matrices[matriz_entrada], dtype=np.float32),
                    metadatos=meta_factorizacion,
                    folds=folds,
                    config=config,
                    carpeta=SALIDA / "evaluacion_ujanet" / base / metodo / matriz_entrada,
                    distribucion=distribucion,
                    numero_capas=len(rangos),
                    inicializacion_onmf="nndsvd",
                    entrada_ujanet=f"matriz_{matriz_entrada}_por_senal_{metodo}",
                )
                resumen.update(
                    {
                        "protocolo_prueba": PROTOCOLO_PRUEBA,
                        "snr_db": snr_db,
                        "metodo_factorizacion": metodo,
                        "matriz_entrada": matriz_entrada,
                        "distribucion": distribucion,
                        "numero_capas": len(rangos),
                        "penalizacion_ortogonal": _penalizacion(definicion, config),
                        "sin_regularizacion": bool(definicion["sin_regularizacion"]),
                        "inicializacion_factorizacion": "nndsvd",
                        "factorizacion_por_senal": True,
                        "diccionario_comun_por_fold": False,
                    }
                )
                filas.append(resumen)
                pd.DataFrame(filas).to_csv(ruta_resultados, index=False, encoding="utf-8-sig")
                hechas.add(clave_hecha)

    tabla = pd.DataFrame(filas).sort_values(
        ["snr_db", "metodo_factorizacion", "matriz_entrada"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    tabla.to_csv(ruta_resultados, index=False, encoding="utf-8-sig")

    pivot_h = (
        tabla.loc[tabla["matriz_entrada"].eq("H")]
        .pivot_table(index="snr_db", columns="metodo_factorizacion", values="Accuracy_mean", aggfunc="first")
        .sort_index(ascending=False)
    )
    pivot_w = (
        tabla.loc[tabla["matriz_entrada"].eq("W")]
        .pivot_table(index="snr_db", columns="metodo_factorizacion", values="Accuracy_mean", aggfunc="first")
        .sort_index(ascending=False)
    )
    pivot_h.to_csv(SALIDA / "tablas_csv" / "tabla_principal_accuracy_H.csv", encoding="utf-8-sig")
    pivot_w.to_csv(SALIDA / "tablas_csv" / "tabla_auditoria_accuracy_W.csv", encoding="utf-8-sig")
    _generar_pdf(tabla, SALIDA / "prueba_snr_tres_metodos.pdf")
    (SALIDA / "resumen_ejecucion.json").write_text(
        json.dumps(
            {
                "protocolo": PROTOCOLO_PRUEBA,
                "filas": len(tabla),
                "snr": list(SNR_OBJETIVOS),
                "metodos": [
                    {
                        "metodo": str(definicion["metodo"]),
                        "rangos": list(definicion["rangos"]),
                        "penalizacion": _penalizacion(definicion, config),
                        "sin_regularizacion": bool(definicion["sin_regularizacion"]),
                    }
                    for definicion in METODOS
                ],
                "factorizacion_por_senal": True,
                "diccionario_comun_por_fold": False,
                "segundos": time.perf_counter() - inicio,
                "salida": str(SALIDA.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[fin prueba snr tres metodos] {SALIDA}")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prueba por SNR: NMF-7, ONMF-7 y Deep-ONMF 9-8-7.")
    parser.add_argument("--datos", type=Path, default=DATOS_DEFECTO)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFECTO)
    return parser


def main() -> None:
    args = construir_parser().parse_args()
    ejecutar(args.datos, args.config)
