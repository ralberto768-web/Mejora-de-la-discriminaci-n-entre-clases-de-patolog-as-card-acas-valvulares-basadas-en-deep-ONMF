from __future__ import annotations

from dataclasses import replace
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from codigo.clasificadores import _entrenar_ujanet, etiquetas_numericas
from codigo.configuracion import CLASES, ConfiguracionExperimento
from codigo.datos import descubrir_audios
from codigo.metricas import metricas_multiclase
from codigo.representaciones import espectrograma_fijo_audio
from busqueda_mejor.evaluacion import adaptar_para_ujanet
from busqueda_mejor.onmf_multicapa import deep_onmf_multicapa

from .arquitecturas import clave, convertir_etiqueta, etiqueta


def _archivos_predicciones(carpeta: Path) -> list[Path]:
    archivos: list[Path] = []
    for fold in range(1, 6):
        corta = carpeta / f"f{fold}_pred.csv"
        larga = carpeta / f"fold_{fold}_predicciones.csv"
        archivos.append(corta if corta.exists() else larga)
    return archivos


def predicciones_completas(carpeta: Path) -> bool:
    archivos = _archivos_predicciones(carpeta)
    if not all(ruta.exists() for ruta in archivos):
        return False
    try:
        tabla = pd.concat(
            [pd.read_csv(ruta, encoding="utf-8-sig") for ruta in archivos],
            ignore_index=True,
        )
    except Exception:
        return False
    if len(tabla) != 1000:
        return False
    claves = tabla["clase"].astype(str) + "/" + tabla["archivo"].astype(str)
    return claves.nunique() == 1000


def buscar_fuente_anterior(
    rangos: tuple[int, ...],
    raiz_ultima: Path,
) -> Path | None:
    numero_capas = len(rangos)
    nombre_h = f"DeepONMF_H{numero_capas}"
    llave = clave(rangos)
    raiz_busqueda = raiz_ultima.parent
    candidatas = [
        raiz_ultima
        / "resultados"
        / "dec"
        / llave
        / "pred"
        / "UjaNet"
        / f"H{numero_capas}",
        raiz_ultima
        / "resultados"
        / "inc"
        / llave
        / "pred"
        / "UjaNet"
        / f"H{numero_capas}",
        raiz_busqueda
        / "resultados"
        / "configuraciones"
        / llave
        / "clasificadores"
        / "UjaNet"
        / "multiclase"
        / nombre_h,
    ]
    for candidata in candidatas:
        if predicciones_completas(candidata):
            return candidata
    return None


def preparar_cache_stft(
    carpeta_datos: Path,
    metadatos_maestros: pd.DataFrame,
    config: ConfiguracionExperimento,
    carpeta_cache: Path,
) -> tuple[Path, pd.DataFrame]:
    carpeta_cache.mkdir(parents=True, exist_ok=True)
    ruta_matriz = carpeta_cache / "stft_1000_float32.npy"
    ruta_meta = carpeta_cache / "metadatos_stft.csv"
    ruta_manifest = carpeta_cache / "manifiesto_stft.json"
    esperada = (
        len(metadatos_maestros),
        config.bins_frecuencia,
        config.tramas_stft_por_segmento,
    )
    if ruta_matriz.exists() and ruta_meta.exists() and ruta_manifest.exists():
        manifiesto = json.loads(ruta_manifest.read_text(encoding="utf-8"))
        if (
            tuple(manifiesto["forma"]) == esperada
            and manifiesto["frecuencia_objetivo_hz"]
            == config.frecuencia_objetivo_hz
            and manifiesto["ventana_stft_muestras"]
            == config.ventana_stft_muestras
            and manifiesto["salto_stft_muestras"]
            == config.salto_stft_muestras
            and manifiesto["puntos_fft"] == config.puntos_fft
        ):
            return ruta_matriz, pd.read_csv(
                ruta_meta,
                encoding="utf-8-sig",
            )
    registros = descubrir_audios(carpeta_datos)
    claves_registros = [
        registro.clase + "/" + registro.archivo for registro in registros
    ]
    claves_maestras = (
        metadatos_maestros["clase"].astype(str)
        + "/"
        + metadatos_maestros["archivo"].astype(str)
    ).tolist()
    if claves_registros != claves_maestras:
        raise AssertionError("El orden de los 1000 audios no coincide")
    matriz = np.lib.format.open_memmap(
        ruta_matriz,
        mode="w+",
        dtype=np.float32,
        shape=esperada,
    )
    filas = []
    for posicion, registro in enumerate(registros, start=1):
        stft, rellenado, tramas = espectrograma_fijo_audio(registro, config)
        matriz[posicion - 1] = stft.astype(np.float32)
        filas.append(
            {
                "indice_interno": posicion - 1,
                "clase": registro.clase,
                "etiqueta_binaria": registro.etiqueta_binaria,
                "archivo": registro.archivo,
                "ruta": str(registro.ruta),
                "rellenado_menor_2s": rellenado,
                "tramas_pcg": tramas,
            }
        )
        if posicion == 1 or posicion % 100 == 0:
            print(f"[STFT] {posicion}/{len(registros)}")
    matriz.flush()
    del matriz
    metadatos = pd.DataFrame(filas)
    metadatos.to_csv(ruta_meta, index=False, encoding="utf-8-sig")
    ruta_manifest.write_text(
        json.dumps(
            {
                "forma": list(esperada),
                "frecuencia_objetivo_hz": config.frecuencia_objetivo_hz,
                "ventana_stft_muestras": config.ventana_stft_muestras,
                "salto_stft_muestras": config.salto_stft_muestras,
                "puntos_fft": config.puntos_fft,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ruta_matriz, metadatos


def _calcular_metricas_fold(
    predicciones: pd.DataFrame,
) -> dict[str, object]:
    mapa = {clase: indice for indice, clase in enumerate(CLASES)}
    reales = predicciones["clase"].map(mapa).to_numpy(dtype=int)
    predichas = predicciones["pred_multiclase"].map(mapa).to_numpy(dtype=int)
    _, tabla = metricas_multiclase(reales, predichas)
    return tabla.loc[tabla["clase"].eq("PROMEDIO_MACRO")].iloc[0].to_dict()


def _resumir_metricas(tabla: pd.DataFrame) -> pd.DataFrame:
    metricas = (
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "Precision",
        "Score",
    )
    fila: dict[str, object] = {
        "distribucion": tabla.iloc[0]["distribucion"],
        "numero_capas": int(tabla.iloc[0]["numero_capas"]),
        "clasificador": "UjaNet",
        "representacion": tabla.iloc[0]["representacion"],
    }
    for metrica in metricas:
        fila[f"{metrica}_mean"] = float(tabla[metrica].mean())
        fila[f"{metrica}_std"] = float(tabla[metrica].std())
    return pd.DataFrame([fila])


def _reconstruir_metricas(
    carpeta_pred: Path,
    rangos: tuple[int, ...],
    carpeta_configuracion: Path,
) -> pd.DataFrame:
    filas = []
    for fold, ruta in enumerate(_archivos_predicciones(carpeta_pred), start=1):
        predicciones = pd.read_csv(ruta, encoding="utf-8-sig")
        fila = {
            "distribucion": etiqueta(rangos),
            "numero_capas": len(rangos),
            "clasificador": "UjaNet",
            "representacion": f"DeepONMF_H{len(rangos)}",
            "fold": fold,
            **_calcular_metricas_fold(predicciones),
        }
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    carpeta_metricas = carpeta_configuracion / "metricas"
    carpeta_metricas.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(
        carpeta_metricas / "metricas_multiclase_por_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    resumen = _resumir_metricas(tabla)
    resumen.to_csv(
        carpeta_metricas / "resumen_metricas_multiclase.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return resumen


def reutilizar_fuente(
    fuente: Path,
    rangos: tuple[int, ...],
    carpeta_configuracion: Path,
) -> pd.DataFrame:
    destino = (
        carpeta_configuracion
        / "pred"
        / "UjaNet"
        / f"H{len(rangos)}"
    )
    destino.mkdir(parents=True, exist_ok=True)
    for fold, origen in enumerate(_archivos_predicciones(fuente), start=1):
        shutil.copy2(origen, destino / f"f{fold}_pred.csv")
    resumen = _reconstruir_metricas(
        destino,
        rangos,
        carpeta_configuracion,
    )
    (carpeta_configuracion / "CONFIGURACION_COMPLETADA.txt").write_text(
        f"reutilizada_desde={fuente}\n",
        encoding="utf-8",
    )
    return resumen


def _extraer_h(
    rangos: tuple[int, ...],
    ruta_stft: Path,
    config: ConfiguracionExperimento,
    carpeta_configuracion: Path,
) -> Path:
    carpeta_extraccion = carpeta_configuracion / "extraccion"
    carpeta_extraccion.mkdir(parents=True, exist_ok=True)
    ruta_h = carpeta_extraccion / "h_final_float32.npy"
    ruta_auditoria = carpeta_extraccion / "auditoria_deep_onmf.csv"
    ruta_progreso = carpeta_extraccion / "progreso.json"
    stft = np.load(ruta_stft, mmap_mode="r")
    forma = (stft.shape[0], rangos[-1], stft.shape[2])
    progreso = 0
    filas_auditoria: list[dict[str, object]] = []
    if ruta_h.exists() and ruta_progreso.exists():
        datos = json.loads(ruta_progreso.read_text(encoding="utf-8"))
        if (
            tuple(datos.get("rangos", [])) == rangos
            and tuple(datos.get("forma", [])) == forma
        ):
            progreso = int(datos.get("audios_completados", 0))
            if ruta_auditoria.exists():
                filas_auditoria = pd.read_csv(
                    ruta_auditoria,
                    encoding="utf-8-sig",
                ).to_dict(orient="records")
    modo = "r+" if ruta_h.exists() and progreso > 0 else "w+"
    h_final = np.lib.format.open_memmap(
        ruta_h,
        mode=modo,
        dtype=np.float32,
        shape=forma,
    )
    for indice in range(progreso, stft.shape[0]):
        inicio = time.perf_counter()
        resultado = deep_onmf_multicapa(
            np.asarray(stft[indice], dtype=np.float64),
            rangos=rangos,
            iteraciones=config.iteraciones_onmf,
            penalizacion_ortogonal=config.penalizacion_ortogonal,
            semilla=config.semilla + (indice + 1) * 37,
        )
        h_final[indice] = resultado.h_final.astype(np.float32)
        fila: dict[str, object] = {
            "indice_interno": indice,
            "distribucion": etiqueta(rangos),
            "numero_capas": len(rangos),
            "error_final": resultado.error_relativo_final,
            "forma_h_final": (
                f"{resultado.h_final.shape[0]}x"
                f"{resultado.h_final.shape[1]}"
            ),
            "segundos_extraccion": time.perf_counter() - inicio,
        }
        for capa in resultado.capas:
            fila[f"capa_{capa.indice}_rango"] = capa.rango
            fila[f"capa_{capa.indice}_error"] = capa.error_relativo
            fila[f"capa_{capa.indice}_ortogonalidad_h"] = (
                capa.ortogonalidad_media
            )
        filas_auditoria.append(fila)
        completados = indice + 1
        if completados % 10 == 0 or completados == stft.shape[0]:
            h_final.flush()
            pd.DataFrame(filas_auditoria).to_csv(
                ruta_auditoria,
                index=False,
                encoding="utf-8-sig",
            )
            ruta_progreso.write_text(
                json.dumps(
                    {
                        "rangos": list(rangos),
                        "forma": list(forma),
                        "audios_completados": completados,
                    }
                ),
                encoding="utf-8",
            )
        if completados == 1 or completados % 100 == 0:
            print(
                f"[{etiqueta(rangos)}] Deep-ONMF "
                f"{completados}/{stft.shape[0]}"
            )
    h_final.flush()
    del h_final
    return ruta_h


def evaluar_configuracion(
    rangos: tuple[int, ...],
    ruta_stft: Path,
    metadatos: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config_base: ConfiguracionExperimento,
    carpeta_configuracion: Path,
) -> pd.DataFrame:
    marcador = carpeta_configuracion / "CONFIGURACION_COMPLETADA.txt"
    ruta_resumen = (
        carpeta_configuracion
        / "metricas"
        / "resumen_metricas_multiclase.csv"
    )
    carpeta_pred = (
        carpeta_configuracion
        / "pred"
        / "UjaNet"
        / f"H{len(rangos)}"
    )
    if marcador.exists() and ruta_resumen.exists() and predicciones_completas(
        carpeta_pred
    ):
        return pd.read_csv(ruta_resumen, encoding="utf-8-sig")
    config = replace(config_base, rangos_deep_onmf=rangos)
    ruta_h = _extraer_h(
        rangos,
        ruta_stft,
        config,
        carpeta_configuracion,
    )
    h_final = np.load(ruta_h, mmap_mode="r")
    x_ujanet, auditoria_padding = adaptar_para_ujanet(
        np.asarray(h_final, dtype=np.float32)
    )
    pd.DataFrame(
        [
            {
                "distribucion": etiqueta(rangos),
                "representacion": f"DeepONMF_H{len(rangos)}",
                **auditoria_padding,
            }
        ]
    ).to_csv(
        carpeta_configuracion / "auditoria_adaptacion_ujanet.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _, y_multi = etiquetas_numericas(metadatos)
    carpeta_pred.mkdir(parents=True, exist_ok=True)
    carpeta_metricas = carpeta_configuracion / "metricas"
    carpeta_metricas.mkdir(parents=True, exist_ok=True)
    filas = []
    for numero_fold, (idx_train, idx_test) in enumerate(folds, start=1):
        ruta_pred = carpeta_pred / f"f{numero_fold}_pred.csv"
        if ruta_pred.exists():
            predicciones = pd.read_csv(ruta_pred, encoding="utf-8-sig")
            if len(predicciones) == len(idx_test):
                filas.append(
                    {
                        "distribucion": etiqueta(rangos),
                        "numero_capas": len(rangos),
                        "clasificador": "UjaNet",
                        "representacion": f"DeepONMF_H{len(rangos)}",
                        "fold": numero_fold,
                        **_calcular_metricas_fold(predicciones),
                    }
                )
                continue
        ruta_modelo = carpeta_pred / f"f{numero_fold}.pt"
        torch.set_num_threads(2)
        pred = _entrenar_ujanet(
            x_ujanet,
            y_multi,
            idx_train,
            idx_test,
            config,
            binario=False,
            ruta_modelo=ruta_modelo,
        )
        ruta_modelo.unlink(missing_ok=True)
        predicciones = metadatos.iloc[idx_test][
            ["clase", "etiqueta_binaria", "archivo", "ruta"]
        ].copy()
        predicciones["fold"] = numero_fold
        predicciones["pred_multiclase"] = [
            CLASES[int(indice)] for indice in np.asarray(pred, dtype=int)
        ]
        predicciones.to_csv(
            ruta_pred,
            index=False,
            encoding="utf-8-sig",
        )
        filas.append(
            {
                "distribucion": etiqueta(rangos),
                "numero_capas": len(rangos),
                "clasificador": "UjaNet",
                "representacion": f"DeepONMF_H{len(rangos)}",
                "fold": numero_fold,
                **_calcular_metricas_fold(predicciones),
            }
        )
        pd.DataFrame(filas).to_csv(
            carpeta_metricas / "metricas_multiclase_por_fold.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[{etiqueta(rangos)}] UjaNet-H fold {numero_fold}/5")
    tabla = pd.DataFrame(filas).sort_values("fold")
    if len(tabla) != 5:
        raise AssertionError(
            f"{etiqueta(rangos)}: se esperaban cinco folds"
        )
    tabla.to_csv(
        carpeta_metricas / "metricas_multiclase_por_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    resumen = _resumir_metricas(tabla)
    resumen.to_csv(
        ruta_resumen,
        index=False,
        encoding="utf-8-sig",
    )
    marcador.write_text(
        (
            f"{etiqueta(rangos)}; audios={len(metadatos)}; "
            f"folds={len(folds)}; UjaNet-H multiclase\n"
        ),
        encoding="utf-8",
    )
    del x_ujanet
    del h_final
    gc.collect()
    try:
        ruta_h.unlink(missing_ok=True)
    except PermissionError:
        # El resultado es reconstruible y se limpiara en la siguiente ejecucion.
        pass
    return resumen
