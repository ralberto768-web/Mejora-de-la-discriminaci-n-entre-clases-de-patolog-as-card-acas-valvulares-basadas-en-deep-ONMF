from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.fftpack import dct

from codigo.configuracion import ConfiguracionExperimento
from codigo.datos import RegistroAudio
from codigo.representaciones import banco_mel, espectrograma_fijo_audio

from .arquitecturas import clave, etiqueta, nombre_h
from .configuracion import REPRESENTACIONES_CLASICAS
from .onmf_multicapa import deep_onmf_multicapa


def _metadatos(registros: list[RegistroAudio]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "indice_interno": indice,
                "clase": registro.clase,
                "etiqueta_binaria": registro.etiqueta_binaria,
                "archivo": registro.archivo,
                "ruta": str(registro.ruta.resolve()),
                "duracion_s": registro.duracion_s,
            }
            for indice, registro in enumerate(registros)
        ]
    )


def _manifest_valido(ruta: Path, datos: dict[str, object]) -> bool:
    if not ruta.exists():
        return False
    try:
        existente = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(existente.get(clave) == valor for clave, valor in datos.items())


def extraer_clasicas(
    base: str,
    registros: list[RegistroAudio],
    config: ConfiguracionExperimento,
    carpeta_resultados: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    carpeta = carpeta_resultados / "representaciones" / base / "clasicas"
    carpeta.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base": base,
        "numero_audios": len(registros),
        "representaciones": list(REPRESENTACIONES_CLASICAS),
        "semilla": config.semilla,
    }
    ruta_manifest = carpeta / "manifest.json"
    rutas = {nombre: carpeta / f"{nombre}.npy" for nombre in REPRESENTACIONES_CLASICAS}
    ruta_meta = carpeta / "metadatos.csv"
    if (
        _manifest_valido(ruta_manifest, manifest)
        and ruta_meta.exists()
        and all(ruta.exists() for ruta in rutas.values())
    ):
        matrices = {
            nombre: np.load(ruta, mmap_mode="r")
            for nombre, ruta in rutas.items()
        }
        return matrices, pd.read_csv(ruta_meta, encoding="utf-8-sig")

    mel_bank = banco_mel(config)
    matrices: dict[str, list[np.ndarray]] = {nombre: [] for nombre in REPRESENTACIONES_CLASICAS}
    filas_auditoria: list[dict[str, object]] = []
    filas_meta: list[dict[str, object]] = []
    for indice, registro in enumerate(registros, start=1):
        inicio = time.perf_counter()
        stft, rellenado, tramas_pcg = espectrograma_fijo_audio(registro, config)
        mel = np.maximum(mel_bank @ stft, 1e-12)
        logmel = np.log(mel)
        mfcc = dct(logmel, type=2, axis=0, norm="ortho")[: config.coeficientes_mfcc]
        matrices["STFT"].append(stft)
        matrices["MFCC"].append(mfcc)
        matrices["Mel"].append(mel)
        matrices["LogMel"].append(logmel)
        filas_meta.append(
            {
                "indice_interno": indice - 1,
                "base": base,
                "clase": registro.clase,
                "etiqueta_binaria": registro.etiqueta_binaria,
                "archivo": registro.archivo,
                "ruta": str(registro.ruta.resolve()),
                "duracion_s": registro.duracion_s,
                "rellenado_menor_2s": rellenado,
                "tramas_pcg": tramas_pcg,
            }
        )
        filas_auditoria.append(
            {
                "indice_interno": indice - 1,
                "base": base,
                "clase": registro.clase,
                "archivo": registro.archivo,
                "segundos_extraccion_clasica": time.perf_counter() - inicio,
            }
        )
        if indice == 1 or indice % 100 == 0 or indice == len(registros):
            print(f"[clasicas {base}] {indice}/{len(registros)}")

    matrices_np = {
        nombre: np.stack(valores).astype(np.float32)
        for nombre, valores in matrices.items()
    }
    for nombre, matriz in matrices_np.items():
        np.save(rutas[nombre], matriz)
    metadatos = pd.DataFrame(filas_meta)
    metadatos.to_csv(ruta_meta, index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_auditoria).to_csv(
        carpeta / "auditoria_extraccion_clasica.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ruta_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return matrices_np, metadatos


def extraer_deep_h(
    base: str,
    registros: list[RegistroAudio],
    config: ConfiguracionExperimento,
    rangos: tuple[int, ...],
    carpeta_resultados: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    representacion = nombre_h(rangos)
    distribucion = etiqueta(rangos)
    carpeta = carpeta_resultados / "representaciones" / base / "deep_onmf" / clave(rangos)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_matriz = carpeta / f"{representacion}.npy"
    ruta_meta = carpeta / "metadatos.csv"
    ruta_manifest = carpeta / "manifest.json"
    manifest = {
        "base": base,
        "numero_audios": len(registros),
        "distribucion": distribucion,
        "numero_capas": len(rangos),
        "inicializacion_onmf": "nndsvd",
        "iteraciones_onmf": config.iteraciones_onmf,
        "penalizacion_ortogonal": config.penalizacion_ortogonal,
        "semilla": config.semilla,
    }
    if _manifest_valido(ruta_manifest, manifest) and ruta_matriz.exists() and ruta_meta.exists():
        return np.load(ruta_matriz, mmap_mode="r"), pd.read_csv(ruta_meta, encoding="utf-8-sig")

    clasicas, metadatos = extraer_clasicas(base, registros, config, carpeta_resultados)
    stfts = np.asarray(clasicas["STFT"], dtype=np.float32)
    matrices_h: list[np.ndarray] = []
    filas_auditoria: list[dict[str, object]] = []
    for indice, stft in enumerate(stfts, start=1):
        inicio = time.perf_counter()
        resultado = deep_onmf_multicapa(
            stft,
            rangos=rangos,
            iteraciones=config.iteraciones_onmf,
            penalizacion_ortogonal=config.penalizacion_ortogonal,
            semilla=config.semilla + indice * 37,
        )
        matrices_h.append(resultado.h_final.astype(np.float32))
        fila: dict[str, object] = {
            "indice_interno": indice - 1,
            "base": base,
            "clase": str(metadatos.iloc[indice - 1]["clase"]),
            "archivo": str(metadatos.iloc[indice - 1]["archivo"]),
            "distribucion": distribucion,
            "representacion": representacion,
            "numero_capas": len(rangos),
            "inicializacion_onmf": "nndsvd",
            "error_final": resultado.error_relativo_final,
            "forma_h_final": f"{resultado.h_final.shape[0]}x{resultado.h_final.shape[1]}",
            "segundos_extraccion_deep": time.perf_counter() - inicio,
        }
        for capa in resultado.capas:
            fila[f"capa_{capa.indice}_rango"] = capa.rango
            fila[f"capa_{capa.indice}_error"] = capa.error_relativo
            fila[f"capa_{capa.indice}_ortogonalidad_h"] = capa.ortogonalidad_media
            fila[f"capa_{capa.indice}_segundos"] = capa.segundos
        filas_auditoria.append(fila)
        if indice == 1 or indice % 100 == 0 or indice == len(stfts):
            print(f"[DeepONMF {base} {distribucion}] {indice}/{len(stfts)}")

    matriz = np.stack(matrices_h).astype(np.float32)
    np.save(ruta_matriz, matriz)
    metadatos.to_csv(ruta_meta, index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_auditoria).to_csv(
        carpeta / "auditoria_deep_onmf.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ruta_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return matriz, metadatos
