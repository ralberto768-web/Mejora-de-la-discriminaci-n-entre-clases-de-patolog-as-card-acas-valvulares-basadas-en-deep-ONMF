from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.fftpack import dct

from .configuracion import ConfiguracionExperimento, REPRESENTACIONES
from .datos import RegistroAudio, dividir_en_tramas_pcg, leer_wav_normalizado
from .onmf import deep_onmf


@dataclass(frozen=True)
class Representaciones:
    matrices: dict[str, np.ndarray]
    metadatos: pd.DataFrame
    auditoria_onmf: pd.DataFrame


def espectrograma_trama(trama: np.ndarray, config: ConfiguracionExperimento) -> np.ndarray:
    ventana = np.hamming(config.ventana_stft_muestras)
    inicios = np.arange(
        0,
        len(trama) - config.ventana_stft_muestras + 1,
        config.salto_stft_muestras,
    )
    segmentos = np.stack([trama[i : i + config.ventana_stft_muestras] for i in inicios], axis=0)
    segmentos = segmentos * ventana[None, :]
    espectro = np.fft.rfft(segmentos, n=config.puntos_fft, axis=1)
    magnitud = np.abs(espectro).T
    return np.maximum(magnitud, 1e-12).astype(np.float64)


def espectrograma_fijo_audio(
    registro: RegistroAudio,
    config: ConfiguracionExperimento,
) -> tuple[np.ndarray, bool, int]:
    senal, frecuencia = leer_wav_normalizado(registro.ruta)
    if frecuencia != config.frecuencia_objetivo_hz:
        raise ValueError(f"{registro.ruta} tiene {frecuencia} Hz; se esperaban {config.frecuencia_objetivo_hz} Hz")
    tramas, rellenado = dividir_en_tramas_pcg(senal, config)
    espectrogramas = [espectrograma_trama(trama, config) for trama in tramas]
    spec = np.mean(np.stack(espectrogramas, axis=0), axis=0)
    spec = spec / max(float(np.sum(spec)), 1e-12)
    return spec, rellenado, len(tramas)


def _hz_a_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_a_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def banco_mel(config: ConfiguracionExperimento) -> np.ndarray:
    frecuencias = np.linspace(0.0, config.frecuencia_objetivo_hz / 2.0, config.bins_frecuencia)
    puntos_mel = np.linspace(_hz_a_mel(0.0), _hz_a_mel(config.frecuencia_objetivo_hz / 2.0), config.bandas_mel + 2)
    puntos_hz = _mel_a_hz(puntos_mel)
    banco = np.zeros((config.bandas_mel, config.bins_frecuencia), dtype=np.float64)
    for banda in range(config.bandas_mel):
        izquierda, centro, derecha = puntos_hz[banda], puntos_hz[banda + 1], puntos_hz[banda + 2]
        subida = (frecuencias - izquierda) / max(centro - izquierda, 1e-12)
        bajada = (derecha - frecuencias) / max(derecha - centro, 1e-12)
        banco[banda] = np.maximum(0.0, np.minimum(subida, bajada))
    return banco


def _guardar_npz_representacion(nombre: str, x: np.ndarray, metadatos: pd.DataFrame, carpeta: Path) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(carpeta / f"{nombre}.npz", x=x.astype(np.float32), representacion=nombre)
    metadatos.to_csv(carpeta / "metadatos.csv", index=False, encoding="utf-8-sig")


def extraer_representaciones(
    registros: list[RegistroAudio],
    config: ConfiguracionExperimento,
    carpeta_salida: Path,
) -> Representaciones:
    matrices: dict[str, list[np.ndarray]] = {nombre: [] for nombre in REPRESENTACIONES}
    filas_meta: list[dict[str, object]] = []
    filas_onmf: list[dict[str, object]] = []
    mel_bank = banco_mel(config)

    for posicion, registro in enumerate(registros, start=1):
        inicio_audio = time.perf_counter()
        stft, rellenado, tramas_pcg = espectrograma_fijo_audio(registro, config)
        mel = np.maximum(mel_bank @ stft, 1e-12)
        logmel = np.log(mel)
        mfcc = dct(logmel, type=2, axis=0, norm="ortho")[: config.coeficientes_mfcc]
        resultado = deep_onmf(
            stft,
            rangos=config.rangos_deep_onmf,
            iteraciones=config.iteraciones_onmf,
            penalizacion_ortogonal=config.penalizacion_ortogonal,
            semilla=config.semilla + posicion * 37,
        )

        matrices["STFT"].append(stft)
        matrices["MFCC"].append(mfcc)
        matrices["MelSpectrogram"].append(mel)
        matrices["LogMelSpectrogram"].append(logmel)
        matrices["DeepONMF_W"].append(resultado.w_final)
        matrices["DeepONMF_H3"].append(resultado.h3)
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
                "segundos_extraccion": time.perf_counter() - inicio_audio,
            }
        )
        fila_onmf = {
            "indice_interno": posicion - 1,
            "clase": registro.clase,
            "archivo": registro.archivo,
            "error_final": resultado.error_relativo_final,
        }
        for capa in resultado.capas:
            fila_onmf[f"capa_{capa.indice}_rango"] = capa.rango
            fila_onmf[f"capa_{capa.indice}_error"] = capa.error_relativo
            fila_onmf[f"capa_{capa.indice}_ortogonalidad_h"] = capa.ortogonalidad_media
            fila_onmf[f"capa_{capa.indice}_segundos"] = capa.segundos
        filas_onmf.append(fila_onmf)
        if posicion == 1 or posicion % 25 == 0 or posicion == len(registros):
            print(f"[representaciones] {posicion}/{len(registros)} audios procesados")

    metadatos = pd.DataFrame(filas_meta)
    auditoria_onmf = pd.DataFrame(filas_onmf)
    matrices_np = {nombre: np.stack(valores, axis=0).astype(np.float32) for nombre, valores in matrices.items()}

    for nombre, x in matrices_np.items():
        _guardar_npz_representacion(nombre, x, metadatos, carpeta_salida / "representaciones" / nombre)
    auditoria_onmf.to_csv(carpeta_salida / "representaciones" / "auditoria_deep_onmf.csv", index=False, encoding="utf-8-sig")
    return Representaciones(
        matrices=matrices_np,
        metadatos=metadatos,
        auditoria_onmf=auditoria_onmf,
    )


def vectorizar_para_clasicos(x: np.ndarray) -> np.ndarray:
    return x.reshape((x.shape[0], -1)).astype(np.float32)
