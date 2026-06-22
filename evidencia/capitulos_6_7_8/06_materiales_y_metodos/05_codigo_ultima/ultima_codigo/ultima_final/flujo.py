from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from codigo.configuracion import cargar_configuracion, guardar_configuracion
from codigo.datos import descubrir_audios
from codigo.clasificadores import guardar_particiones

from .arquitecturas import (
    clave,
    etiqueta,
    generar_plan_completo,
    limitar_por_capas,
    nombre_h,
)
from .configuracion import (
    BASE_ORIGINAL,
    BASE_SNR0,
    DOCUMENTOS_FINALES,
    REPRESENTACIONES_CLASICAS,
    SNR_OBJETIVOS,
    carpetas_obligatorias,
    nombre_base_snr,
)
from .documentos import generar_pdf_optimizacion, generar_pdf_resultados
from .evaluacion import PROTOCOLO, evaluar_ujanet_multiclase, folds_desde_metadatos
from .extraccion import extraer_clasicas, extraer_deep_h
from .ruido import preparar_bases


RAIZ = Path(__file__).resolve().parents[2]
RAIZ_TFG = RAIZ.parents[1]
DATOS_DEFECTO = RAIZ_TFG / "Bases de Datos"
CONFIG_DEFECTO = RAIZ_TFG / "Implementacion_last" / "configuracion_experimento.json"
INICIALIZACION_ONMF = "nndsvd"


def crear_estructura(raiz: Path) -> None:
    for carpeta in carpetas_obligatorias(raiz):
        carpeta.mkdir(parents=True, exist_ok=True)


def _configuracion(rapido: bool, ruta_config: Path):
    config = cargar_configuracion(ruta_config)
    if not rapido:
        return config
    return replace(
        config,
        iteraciones_onmf=4,
        pca_componentes_max=16,
        ujanet_epocas=2,
        ujanet_paciencia=1,
        ujanet_lote=4,
    )


def _rutas_trabajo(rapido: bool) -> dict[str, Path]:
    if not rapido:
        return {
            "datasets": RAIZ / "datasets_ruidosos",
            "opt_original": RAIZ / "01_optimizacion_dataset_original",
            "opt_snr0": RAIZ / "02_optimizacion_dataset_SNR0db",
            "res_original": RAIZ / "03_resultados_optimizacion_original",
            "res_snr0": RAIZ / "04_resultados_optimizacion_SNR0db",
            "documentos": RAIZ / "documentos_finales",
            "auditoria": RAIZ / "auditoria",
        }
    base = RAIZ / "auditoria" / "verificacion_rapida"
    return {
        "datasets": base / "datasets_ruidosos",
        "opt_original": base / "01_optimizacion_dataset_original",
        "opt_snr0": base / "02_optimizacion_dataset_SNR0db",
        "res_original": base / "03_resultados_optimizacion_original",
        "res_snr0": base / "04_resultados_optimizacion_SNR0db",
        "documentos": base / "documentos_finales",
        "auditoria": base,
    }


def _distribucion_limpia(valor: object) -> str:
    if pd.isna(valor):
        return ""
    return str(valor).strip()


def _deduplicar_resultados_ruido(filas: list[dict[str, object]]) -> list[dict[str, object]]:
    if not filas:
        return []
    tabla = pd.DataFrame(filas)
    if "distribucion" not in tabla.columns:
        tabla["distribucion"] = ""
    tabla["distribucion"] = tabla["distribucion"].map(_distribucion_limpia)
    tabla["base"] = tabla["base"].astype(str)
    tabla["representacion"] = tabla["representacion"].astype(str)
    tabla = tabla.drop_duplicates(
        subset=["base", "representacion", "distribucion"],
        keep="last",
    )
    return tabla.to_dict(orient="records")


def _guardar_auditoria_config(
    config,
    rutas: dict[str, Path],
    rapido: bool,
    candidatos: list[tuple[int, ...]],
) -> None:
    rutas["auditoria"].mkdir(parents=True, exist_ok=True)
    guardar_configuracion(config, rutas["auditoria"] / "configuracion_usada.json")
    pd.DataFrame(
        [
            {
                "distribucion": etiqueta(rangos),
                "numero_capas": len(rangos),
                "representacion": nombre_h(rangos),
                "modo_rapido": rapido,
            }
            for rangos in candidatos
        ]
    ).to_csv(
        rutas["auditoria"] / "plan_arquitecturas.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _optimizar_dataset(
    nombre_base: str,
    carpeta_datos: Path,
    carpeta_salida: Path,
    config,
    candidatos: list[tuple[int, ...]],
) -> dict[int, tuple[int, ...]]:
    inicio = time.perf_counter()
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    (carpeta_salida / "tablas_csv").mkdir(parents=True, exist_ok=True)
    registros = descubrir_audios(carpeta_datos)
    if not registros:
        raise RuntimeError(f"No se encontraron audios en {carpeta_datos}")
    extraer_clasicas(nombre_base, registros, config, carpeta_salida)
    _, metadatos = extraer_clasicas(nombre_base, registros, config, carpeta_salida)
    folds = folds_desde_metadatos(metadatos, config)
    guardar_particiones(folds, metadatos, carpeta_salida / "particiones")

    ruta_todas = carpeta_salida / "tablas_csv" / "resultados_todas_arquitecturas.csv"
    filas: list[dict[str, object]] = []
    if ruta_todas.exists():
        tabla_existente = pd.read_csv(ruta_todas, encoding="utf-8-sig")
        if (
            "inicializacion_onmf" in tabla_existente.columns
            and tabla_existente["inicializacion_onmf"].eq(INICIALIZACION_ONMF).all()
            and "protocolo_ujanet" in tabla_existente.columns
            and tabla_existente["protocolo_ujanet"].eq(PROTOCOLO).all()
        ):
            filas = tabla_existente.to_dict(orient="records")
    hechas = {str(fila["distribucion"]) for fila in filas}

    for posicion, rangos in enumerate(candidatos, start=1):
        distribucion = etiqueta(rangos)
        representacion = nombre_h(rangos)
        if distribucion in hechas:
            continue
        print(f"[optimizacion {nombre_base}] {posicion}/{len(candidatos)} {distribucion}")
        x_h, meta_h = extraer_deep_h(nombre_base, registros, config, rangos, carpeta_salida)
        resumen = evaluar_ujanet_multiclase(
            base=nombre_base,
            representacion=representacion,
            x=np.asarray(x_h, dtype=np.float32),
            metadatos=meta_h,
            folds=folds,
            config=config,
            carpeta=carpeta_salida / "evaluacion_ujanet" / clave(rangos),
            distribucion=distribucion,
            numero_capas=len(rangos),
            inicializacion_onmf=INICIALIZACION_ONMF,
        )
        resumen["inicializacion_onmf"] = INICIALIZACION_ONMF
        resumen["distribucion"] = distribucion
        resumen["numero_capas"] = len(rangos)
        resumen["representacion"] = representacion
        filas.append(resumen)
        pd.DataFrame(filas).to_csv(ruta_todas, index=False, encoding="utf-8-sig")

    tabla = pd.DataFrame(filas)
    if tabla.empty:
        raise RuntimeError(f"No se pudo generar ningun resultado para {nombre_base}")
    tabla = tabla.sort_values(
        ["numero_capas", "Accuracy_mean", "Score_mean", "Sensitivity_mean", "Specificity_mean", "Precision_mean"],
        ascending=[True, False, False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    mejores = (
        tabla.groupby("numero_capas", group_keys=False)
        .head(4)
        .reset_index(drop=True)
    )
    mejores.to_csv(
        carpeta_salida / "tablas_csv" / "mejores_por_capas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    optimas: dict[int, tuple[int, ...]] = {}
    for capas, grupo in tabla.groupby("numero_capas"):
        mejor = grupo.iloc[0]
        optimas[int(capas)] = tuple(int(valor) for valor in str(mejor["distribucion"]).split("-"))
    (carpeta_salida / "configuraciones_optimas.json").write_text(
        json.dumps(
            {str(capas): list(rangos) for capas, rangos in sorted(optimas.items())},
            indent=2,
        ),
        encoding="utf-8",
    )
    (carpeta_salida / "resumen_ejecucion.json").write_text(
        json.dumps(
            {
                "base": nombre_base,
                "inicializacion_onmf": INICIALIZACION_ONMF,
                "audios": len(registros),
                "folds": len(folds),
                "candidatos": len(candidatos),
                "resultados": len(tabla),
                "segundos": time.perf_counter() - inicio,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return optimas


def _evaluar_resultados_ruido(
    nombre_documento: str,
    optimas: dict[int, tuple[int, ...]],
    carpeta_salida: Path,
    carpeta_datasets: Path,
    config,
) -> pd.DataFrame:
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    (carpeta_salida / "tablas_csv").mkdir(parents=True, exist_ok=True)
    filas: list[dict[str, object]] = []
    ruta_resultados = carpeta_salida / "tablas_csv" / "resultados_por_snr.csv"
    if ruta_resultados.exists():
        tabla_existente = pd.read_csv(ruta_resultados, encoding="utf-8-sig")
        if (
            "inicializacion_onmf" in tabla_existente.columns
            and tabla_existente["inicializacion_onmf"].fillna(INICIALIZACION_ONMF).eq(INICIALIZACION_ONMF).all()
            and "protocolo_ujanet" in tabla_existente.columns
            and tabla_existente["protocolo_ujanet"].eq(PROTOCOLO).all()
        ):
            filas = _deduplicar_resultados_ruido(tabla_existente.to_dict(orient="records"))
            pd.DataFrame(filas).to_csv(ruta_resultados, index=False, encoding="utf-8-sig")
    hechas = {
        (
            str(fila["base"]),
            str(fila["representacion"]),
            _distribucion_limpia(fila.get("distribucion", "")),
        )
        for fila in filas
    }

    for snr_db in SNR_OBJETIVOS:
        base = nombre_base_snr(snr_db)
        registros = descubrir_audios(carpeta_datasets / base)
        clasicas, metadatos = extraer_clasicas(base, registros, config, carpeta_salida)
        folds = folds_desde_metadatos(metadatos, config)
        guardar_particiones(folds, metadatos, carpeta_salida / "particiones" / base)

        for representacion in REPRESENTACIONES_CLASICAS:
            clave_hecha = (base, representacion, "")
            if clave_hecha not in hechas:
                resumen = evaluar_ujanet_multiclase(
                    base=base,
                    representacion=representacion,
                    x=np.asarray(clasicas[representacion], dtype=np.float32),
                    metadatos=metadatos,
                    folds=folds,
                    config=config,
                    carpeta=carpeta_salida / "evaluacion_ujanet" / base / representacion,
                    distribucion="",
                    numero_capas=None,
                    inicializacion_onmf="",
                )
                resumen.update({"snr_db": snr_db, "distribucion": "", "inicializacion_onmf": INICIALIZACION_ONMF})
                filas.append(resumen)
                filas = _deduplicar_resultados_ruido(filas)
                pd.DataFrame(filas).to_csv(ruta_resultados, index=False, encoding="utf-8-sig")
                hechas.add(clave_hecha)

        for capas, rangos in sorted(optimas.items()):
            distribucion = etiqueta(rangos)
            representacion = f"{nombre_h(rangos)}_opt_{nombre_documento}"
            clave_hecha = (base, representacion, distribucion)
            if clave_hecha in hechas:
                continue
            x_h, meta_h = extraer_deep_h(base, registros, config, rangos, carpeta_salida)
            resumen = evaluar_ujanet_multiclase(
                base=base,
                representacion=representacion,
                x=np.asarray(x_h, dtype=np.float32),
                metadatos=meta_h,
                folds=folds,
                config=config,
                carpeta=carpeta_salida / "evaluacion_ujanet" / base / representacion,
                distribucion=distribucion,
                numero_capas=capas,
                inicializacion_onmf=INICIALIZACION_ONMF,
            )
            resumen.update(
                {
                    "snr_db": snr_db,
                    "distribucion": distribucion,
                    "inicializacion_onmf": INICIALIZACION_ONMF,
                }
            )
            filas.append(resumen)
            filas = _deduplicar_resultados_ruido(filas)
            pd.DataFrame(filas).to_csv(ruta_resultados, index=False, encoding="utf-8-sig")
            hechas.add(clave_hecha)

    tabla = pd.DataFrame(_deduplicar_resultados_ruido(filas))
    tabla = tabla.sort_values(["snr_db", "representacion"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    tabla.to_csv(ruta_resultados, index=False, encoding="utf-8-sig")
    return tabla


def ejecutar(
    datos_originales: Path,
    ruta_config: Path,
    rapido: bool,
    limite_por_clase: int,
    max_arq_rapido: int,
    solo_estructura: bool,
) -> None:
    inicio = time.perf_counter()
    crear_estructura(RAIZ)
    if solo_estructura:
        print(f"[ok] Estructura creada en {RAIZ}")
        return

    rutas = _rutas_trabajo(rapido)
    for ruta in rutas.values():
        ruta.mkdir(parents=True, exist_ok=True)
    config = _configuracion(rapido, ruta_config)
    candidatos = generar_plan_completo()
    if rapido:
        candidatos = limitar_por_capas(candidatos, max_arq_rapido)
    _guardar_auditoria_config(config, rutas, rapido, candidatos)

    limite = limite_por_clase if rapido else 0
    auditoria_bases = preparar_bases(
        datos_originales.resolve(),
        rutas["datasets"].resolve(),
        semilla=config.semilla,
        limite_por_clase=limite,
    )
    auditoria_bases.to_csv(
        rutas["auditoria"] / "auditoria_generacion_bases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    optimas_original = _optimizar_dataset(
        BASE_ORIGINAL,
        rutas["datasets"] / BASE_ORIGINAL,
        rutas["opt_original"],
        config,
        candidatos,
    )
    optimas_snr0 = _optimizar_dataset(
        BASE_SNR0,
        rutas["datasets"] / BASE_SNR0,
        rutas["opt_snr0"],
        config,
        candidatos,
    )

    generar_pdf_optimizacion(
        "Optimizacion dataset original",
        rutas["opt_original"],
        rutas["documentos"] / DOCUMENTOS_FINALES["optimizacion_original"],
    )
    generar_pdf_optimizacion(
        "Optimizacion dataset SNR0db",
        rutas["opt_snr0"],
        rutas["documentos"] / DOCUMENTOS_FINALES["optimizacion_snr0"],
    )

    _evaluar_resultados_ruido(
        "original",
        optimas_original,
        rutas["res_original"],
        rutas["datasets"],
        config,
    )
    _evaluar_resultados_ruido(
        "SNR0db",
        optimas_snr0,
        rutas["res_snr0"],
        rutas["datasets"],
        config,
    )
    generar_pdf_resultados(
        "Resultados Optimizacion original",
        rutas["res_original"],
        rutas["documentos"] / DOCUMENTOS_FINALES["resultados_original"],
    )
    generar_pdf_resultados(
        "Resultados Optimizacion SNR0db",
        rutas["res_snr0"],
        rutas["documentos"] / DOCUMENTOS_FINALES["resultados_snr0"],
    )

    resumen = {
        "modo_rapido_no_entregable": rapido,
        "raiz": str(RAIZ.resolve()),
        "inicializacion_onmf": INICIALIZACION_ONMF,
        "datasets": str(rutas["datasets"].resolve()),
        "documentos": {clave: str((rutas["documentos"] / nombre).resolve()) for clave, nombre in DOCUMENTOS_FINALES.items()},
        "optimas_original": {str(k): list(v) for k, v in sorted(optimas_original.items())},
        "optimas_snr0": {str(k): list(v) for k, v in sorted(optimas_snr0.items())},
        "segundos_totales": time.perf_counter() - inicio,
    }
    (rutas["auditoria"] / "resumen_final.json").write_text(
        json.dumps(resumen, indent=2),
        encoding="utf-8",
    )
    print(f"[fin] Documentos: {rutas['documentos']}")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ultimas pruebas Deep-ONMF solicitadas por Juan.")
    parser.add_argument("--datos-originales", type=Path, default=DATOS_DEFECTO)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFECTO)
    parser.add_argument("--rapido", action="store_true")
    parser.add_argument("--limite-por-clase", type=int, default=2)
    parser.add_argument("--max-arquitecturas-rapido", type=int, default=1)
    parser.add_argument("--solo-estructura", action="store_true")
    return parser


def main() -> None:
    args = construir_parser().parse_args()
    ejecutar(
        datos_originales=args.datos_originales,
        ruta_config=args.config,
        rapido=args.rapido,
        limite_por_clase=args.limite_por_clase,
        max_arq_rapido=args.max_arquitecturas_rapido,
        solo_estructura=args.solo_estructura,
    )
