from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from codigo.configuracion import ConfiguracionExperimento
from codigo.datos import RegistroAudio
from codigo.onmf import deep_onmf
from codigo.representaciones import espectrograma_fijo_audio

from .configuracion_pruebas import REPRESENTACIONES_DEEP, etiqueta_distribucion


@dataclass(frozen=True)
class ResultadoExtraccionDeep:
    matrices: dict[str, np.ndarray]
    metadatos: pd.DataFrame
    auditoria: pd.DataFrame


def _rutas_cache(carpeta: Path) -> tuple[Path, Path, Path]:
    return (
        carpeta / "representaciones_deep_onmf.npz",
        carpeta / "metadatos.csv",
        carpeta / "auditoria_deep_onmf.csv",
    )


def _cache_valida(
    carpeta: Path,
    numero_audios: int,
    config: ConfiguracionExperimento,
) -> bool:
    ruta_npz, ruta_meta, ruta_auditoria = _rutas_cache(carpeta)
    ruta_manifest = carpeta / "manifiesto_extraccion.json"
    if not all(ruta.exists() for ruta in (ruta_npz, ruta_meta, ruta_auditoria, ruta_manifest)):
        return False
    try:
        manifiesto = json.loads(ruta_manifest.read_text(encoding="utf-8"))
        return (
            manifiesto["numero_audios"] == numero_audios
            and tuple(manifiesto["rangos"]) == config.rangos_deep_onmf
            and manifiesto["iteraciones"] == config.iteraciones_onmf
            and manifiesto["penalizacion_ortogonal"] == config.penalizacion_ortogonal
            and manifiesto["semilla"] == config.semilla
        )
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def cargar_extraccion(carpeta: Path) -> ResultadoExtraccionDeep:
    ruta_npz, ruta_meta, ruta_auditoria = _rutas_cache(carpeta)
    with np.load(ruta_npz) as archivo:
        matrices = {nombre: archivo[nombre].astype(np.float32) for nombre in REPRESENTACIONES_DEEP}
    return ResultadoExtraccionDeep(
        matrices=matrices,
        metadatos=pd.read_csv(ruta_meta, encoding="utf-8-sig"),
        auditoria=pd.read_csv(ruta_auditoria, encoding="utf-8-sig"),
    )


def extraer_w_h3(
    registros: list[RegistroAudio],
    config: ConfiguracionExperimento,
    carpeta: Path,
    reutilizar: bool = True,
) -> ResultadoExtraccionDeep:
    """Extrae W y H3 conservando exactamente la semilla por audio original."""

    carpeta.mkdir(parents=True, exist_ok=True)
    if reutilizar and _cache_valida(carpeta, len(registros), config):
        print(f"[extracción] Se reutiliza la caché válida de {etiqueta_distribucion(config.rangos_deep_onmf)}")
        return cargar_extraccion(carpeta)

    matrices_w: list[np.ndarray] = []
    matrices_h3: list[np.ndarray] = []
    filas_meta: list[dict[str, object]] = []
    filas_auditoria: list[dict[str, object]] = []
    etiqueta = etiqueta_distribucion(config.rangos_deep_onmf)

    for posicion, registro in enumerate(registros, start=1):
        inicio = time.perf_counter()
        stft, rellenado, tramas_pcg = espectrograma_fijo_audio(registro, config)
        resultado = deep_onmf(
            stft,
            rangos=config.rangos_deep_onmf,
            iteraciones=config.iteraciones_onmf,
            penalizacion_ortogonal=config.penalizacion_ortogonal,
            semilla=config.semilla + posicion * 37,
        )
        matrices_w.append(resultado.w_final.astype(np.float32))
        matrices_h3.append(resultado.h3.astype(np.float32))
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
            "distribucion": etiqueta,
            "error_final": resultado.error_relativo_final,
            "forma_w_final": f"{resultado.w_final.shape[0]}x{resultado.w_final.shape[1]}",
            "forma_h3": f"{resultado.h3.shape[0]}x{resultado.h3.shape[1]}",
        }
        for capa in resultado.capas:
            fila[f"capa_{capa.indice}_rango"] = capa.rango
            fila[f"capa_{capa.indice}_error"] = capa.error_relativo
            fila[f"capa_{capa.indice}_ortogonalidad_h"] = capa.ortogonalidad_media
            fila[f"capa_{capa.indice}_segundos"] = capa.segundos
        filas_auditoria.append(fila)
        if posicion == 1 or posicion % 25 == 0 or posicion == len(registros):
            print(f"[extracción {etiqueta}] {posicion}/{len(registros)} audios")

    matrices = {
        "DeepONMF_W": np.stack(matrices_w).astype(np.float32),
        "DeepONMF_H3": np.stack(matrices_h3).astype(np.float32),
    }
    metadatos = pd.DataFrame(filas_meta)
    auditoria = pd.DataFrame(filas_auditoria)
    ruta_npz, ruta_meta, ruta_auditoria = _rutas_cache(carpeta)
    np.savez_compressed(ruta_npz, **matrices)
    metadatos.to_csv(ruta_meta, index=False, encoding="utf-8-sig")
    auditoria.to_csv(ruta_auditoria, index=False, encoding="utf-8-sig")
    (carpeta / "manifiesto_extraccion.json").write_text(
        json.dumps(
            {
                "numero_audios": len(registros),
                "rangos": list(config.rangos_deep_onmf),
                "iteraciones": config.iteraciones_onmf,
                "penalizacion_ortogonal": config.penalizacion_ortogonal,
                "semilla": config.semilla,
                "formas": {nombre: list(matriz.shape) for nombre, matriz in matrices.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return ResultadoExtraccionDeep(matrices=matrices, metadatos=metadatos, auditoria=auditoria)


def comprobar_formas(
    matrices: dict[str, np.ndarray],
    numero_audios: int,
    rango_final: int,
    bins_frecuencia: int,
    tramas_tiempo: int,
) -> None:
    esperada_w = (numero_audios, bins_frecuencia, rango_final)
    esperada_h3 = (numero_audios, rango_final, tramas_tiempo)
    if matrices["DeepONMF_W"].shape != esperada_w:
        raise AssertionError(f"Forma W incorrecta: {matrices['DeepONMF_W'].shape}, esperada {esperada_w}")
    if matrices["DeepONMF_H3"].shape != esperada_h3:
        raise AssertionError(f"Forma H3 incorrecta: {matrices['DeepONMF_H3'].shape}, esperada {esperada_h3}")
