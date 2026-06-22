from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from codigo.configuracion import ConfiguracionExperimento
from codigo.datos import RegistroAudio
from codigo.representaciones import espectrograma_fijo_audio

from .configuracion_busqueda import etiqueta, nombre_h
from .onmf_multicapa import deep_onmf_multicapa


@dataclass(frozen=True)
class ResultadoExtraccion:
    matrices: dict[str, np.ndarray]
    metadatos: pd.DataFrame
    auditoria: pd.DataFrame


def _rutas(carpeta: Path) -> tuple[Path, Path, Path, Path]:
    return (
        carpeta / "representaciones_deep_onmf.npz",
        carpeta / "metadatos.csv",
        carpeta / "auditoria_deep_onmf.csv",
        carpeta / "manifiesto_extraccion.json",
    )


def _cache_valida(
    carpeta: Path,
    numero_audios: int,
    config: ConfiguracionExperimento,
) -> bool:
    ruta_npz, ruta_meta, ruta_auditoria, ruta_manifest = _rutas(carpeta)
    if not all(
        ruta.exists()
        for ruta in (ruta_npz, ruta_meta, ruta_auditoria, ruta_manifest)
    ):
        return False
    try:
        datos = json.loads(ruta_manifest.read_text(encoding="utf-8"))
        return (
            int(datos["numero_audios"]) == numero_audios
            and tuple(datos["rangos"]) == tuple(config.rangos_deep_onmf)
            and int(datos["iteraciones"]) == config.iteraciones_onmf
            and float(datos["penalizacion_ortogonal"])
            == config.penalizacion_ortogonal
            and int(datos["semilla"]) == config.semilla
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def cargar_extraccion(carpeta: Path, numero_capas: int) -> ResultadoExtraccion:
    ruta_npz, ruta_meta, ruta_auditoria, _ = _rutas(carpeta)
    representaciones = ("DeepONMF_W", nombre_h(numero_capas))
    with np.load(ruta_npz) as archivo:
        matrices = {
            representacion: archivo[representacion].astype(np.float32)
            for representacion in representaciones
        }
    return ResultadoExtraccion(
        matrices=matrices,
        metadatos=pd.read_csv(ruta_meta, encoding="utf-8-sig"),
        auditoria=pd.read_csv(ruta_auditoria, encoding="utf-8-sig"),
    )


def extraer_representaciones(
    registros: list[RegistroAudio],
    config: ConfiguracionExperimento,
    carpeta: Path,
    reutilizar: bool = True,
) -> ResultadoExtraccion:
    numero_capas = len(config.rangos_deep_onmf)
    representacion_h = nombre_h(numero_capas)
    carpeta.mkdir(parents=True, exist_ok=True)
    if reutilizar and _cache_valida(carpeta, len(registros), config):
        print(
            f"[extraccion] Se reutiliza {etiqueta(config.rangos_deep_onmf)}"
        )
        return cargar_extraccion(carpeta, numero_capas)

    matrices_w: list[np.ndarray] = []
    matrices_h: list[np.ndarray] = []
    filas_meta: list[dict[str, object]] = []
    filas_auditoria: list[dict[str, object]] = []
    distribucion = etiqueta(config.rangos_deep_onmf)

    for posicion, registro in enumerate(registros, start=1):
        inicio = time.perf_counter()
        stft, rellenado, tramas_pcg = espectrograma_fijo_audio(registro, config)
        resultado = deep_onmf_multicapa(
            stft,
            rangos=tuple(config.rangos_deep_onmf),
            iteraciones=config.iteraciones_onmf,
            penalizacion_ortogonal=config.penalizacion_ortogonal,
            semilla=config.semilla + posicion * 37,
        )
        matrices_w.append(resultado.w_final.astype(np.float32))
        matrices_h.append(resultado.h_final.astype(np.float32))
        filas_meta.append(
            {
                "indice_interno": posicion - 1,
                "clase": registro.clase,
                "etiqueta_binaria": registro.etiqueta_binaria,
                "archivo": registro.archivo,
                "ruta": str(registro.ruta),
                "duracion_s": registro.duracion_s,
                "rellenado_menor_2s": rellenado,
                "tramas_pcg": tramas_pcg,
                "segundos_extraccion": time.perf_counter() - inicio,
            }
        )
        fila: dict[str, object] = {
            "indice_interno": posicion - 1,
            "clase": registro.clase,
            "archivo": registro.archivo,
            "distribucion": distribucion,
            "numero_capas": numero_capas,
            "error_final": resultado.error_relativo_final,
            "forma_w_final": (
                f"{resultado.w_final.shape[0]}x{resultado.w_final.shape[1]}"
            ),
            "forma_h_final": (
                f"{resultado.h_final.shape[0]}x{resultado.h_final.shape[1]}"
            ),
        }
        for capa in resultado.capas:
            fila[f"capa_{capa.indice}_rango"] = capa.rango
            fila[f"capa_{capa.indice}_error"] = capa.error_relativo
            fila[f"capa_{capa.indice}_ortogonalidad_h"] = (
                capa.ortogonalidad_media
            )
            fila[f"capa_{capa.indice}_segundos"] = capa.segundos
        filas_auditoria.append(fila)
        if posicion == 1 or posicion % 25 == 0 or posicion == len(registros):
            print(
                f"[extraccion {distribucion}] {posicion}/{len(registros)} audios"
            )

    matrices = {
        "DeepONMF_W": np.stack(matrices_w).astype(np.float32),
        representacion_h: np.stack(matrices_h).astype(np.float32),
    }
    metadatos = pd.DataFrame(filas_meta)
    auditoria = pd.DataFrame(filas_auditoria)
    ruta_npz, ruta_meta, ruta_auditoria, ruta_manifest = _rutas(carpeta)
    np.savez_compressed(ruta_npz, **matrices)
    metadatos.to_csv(ruta_meta, index=False, encoding="utf-8-sig")
    auditoria.to_csv(ruta_auditoria, index=False, encoding="utf-8-sig")
    ruta_manifest.write_text(
        json.dumps(
            {
                "numero_audios": len(registros),
                "rangos": list(config.rangos_deep_onmf),
                "numero_capas": numero_capas,
                "representacion_h": representacion_h,
                "iteraciones": config.iteraciones_onmf,
                "penalizacion_ortogonal": config.penalizacion_ortogonal,
                "semilla": config.semilla,
                "formas": {
                    nombre: list(matriz.shape)
                    for nombre, matriz in matrices.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return ResultadoExtraccion(matrices, metadatos, auditoria)


def comprobar_formas(
    matrices: dict[str, np.ndarray],
    numero_audios: int,
    numero_capas: int,
    rango_final: int,
    bins_frecuencia: int,
    tramas_tiempo: int,
) -> None:
    esperada_w = (numero_audios, bins_frecuencia, rango_final)
    esperada_h = (numero_audios, rango_final, tramas_tiempo)
    representacion_h = nombre_h(numero_capas)
    if matrices["DeepONMF_W"].shape != esperada_w:
        raise AssertionError(
            f"Forma W incorrecta: {matrices['DeepONMF_W'].shape}; "
            f"esperada {esperada_w}"
        )
    if matrices[representacion_h].shape != esperada_h:
        raise AssertionError(
            f"Forma {representacion_h} incorrecta: "
            f"{matrices[representacion_h].shape}; esperada {esperada_h}"
        )
