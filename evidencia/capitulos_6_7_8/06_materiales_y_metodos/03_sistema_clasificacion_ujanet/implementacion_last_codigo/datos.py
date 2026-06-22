from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np
import pandas as pd

from .configuracion import CLASES, CLASES_ANOMALAS, ConfiguracionExperimento


@dataclass(frozen=True)
class RegistroAudio:
    indice: int
    clase: str
    etiqueta_binaria: str
    ruta: Path
    frecuencia_hz: int
    canales: int
    bytes_muestra: int
    muestras: int
    duracion_s: float

    @property
    def archivo(self) -> str:
        return self.ruta.name


def inferir_clase(nombre_carpeta: str) -> str | None:
    for clase in ("MVP", "AS", "MR", "MS", "N"):
        if nombre_carpeta.startswith(f"{clase}_"):
            return clase
    return None


def descubrir_audios(carpeta_datos: Path) -> list[RegistroAudio]:
    registros: list[RegistroAudio] = []
    indice = 0
    for carpeta in sorted(carpeta_datos.iterdir()):
        if not carpeta.is_dir() or carpeta.name.lower() == "code":
            continue
        clase = inferir_clase(carpeta.name)
        if clase is None or clase not in CLASES:
            continue
        for ruta in sorted(carpeta.glob("*.wav")):
            with wave.open(str(ruta), "rb") as wav:
                frecuencia = wav.getframerate()
                muestras = wav.getnframes()
                indice += 1
                registros.append(
                    RegistroAudio(
                        indice=indice - 1,
                        clase=clase,
                        etiqueta_binaria="anomalo" if clase in CLASES_ANOMALAS else "normal",
                        ruta=ruta,
                        frecuencia_hz=frecuencia,
                        canales=wav.getnchannels(),
                        bytes_muestra=wav.getsampwidth(),
                        muestras=muestras,
                        duracion_s=muestras / max(frecuencia, 1),
                    )
                )
    return sorted(registros, key=lambda r: (CLASES.index(r.clase), r.ruta.name))


def limitar_por_clase(registros: list[RegistroAudio], limite: int) -> list[RegistroAudio]:
    if limite <= 0:
        return registros
    conteos = {clase: 0 for clase in CLASES}
    seleccionados: list[RegistroAudio] = []
    for registro in registros:
        if conteos[registro.clase] < limite:
            seleccionados.append(registro)
            conteos[registro.clase] += 1
    return seleccionados


def leer_wav_normalizado(ruta: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(ruta), "rb") as wav:
        canales = wav.getnchannels()
        frecuencia = wav.getframerate()
        bytes_muestra = wav.getsampwidth()
        bruto = wav.readframes(wav.getnframes())

    if bytes_muestra == 1:
        datos = np.frombuffer(bruto, dtype=np.uint8).astype(np.float64)
        datos = (datos - 128.0) / 128.0
    elif bytes_muestra == 2:
        datos = np.frombuffer(bruto, dtype=np.int16).astype(np.float64) / 32768.0
    elif bytes_muestra == 4:
        datos = np.frombuffer(bruto, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Formato WAV no soportado en {ruta}: {bytes_muestra} bytes por muestra")

    if canales > 1:
        datos = datos.reshape(-1, canales).mean(axis=1)
    return datos.astype(np.float64, copy=False), frecuencia


def dividir_en_tramas_pcg(
    senal: np.ndarray,
    config: ConfiguracionExperimento,
) -> tuple[list[np.ndarray], bool]:
    longitud = config.muestras_trama_pcg
    salto = config.salto_trama_pcg
    rellenado = False
    if len(senal) < longitud:
        senal = np.pad(senal, (0, longitud - len(senal)), mode="constant")
        rellenado = True
    tramas = [senal[inicio : inicio + longitud] for inicio in range(0, len(senal) - longitud + 1, salto)]
    if not tramas:
        tramas = [senal[:longitud]]
    return tramas, rellenado


def tabla_auditoria(registros: list[RegistroAudio], config: ConfiguracionExperimento) -> pd.DataFrame:
    filas = []
    for clase in CLASES:
        grupo = [r for r in registros if r.clase == clase]
        duraciones = np.array([r.duracion_s for r in grupo], dtype=float)
        filas.append(
            {
                "clase": clase,
                "audios": len(grupo),
                "duracion_min_s": float(np.min(duraciones)) if len(duraciones) else 0.0,
                "duracion_media_s": float(np.mean(duraciones)) if len(duraciones) else 0.0,
                "duracion_max_s": float(np.max(duraciones)) if len(duraciones) else 0.0,
                "audios_rellenados_menores_2s": int(np.sum(duraciones < config.duracion_trama_pcg_s)),
                "frecuencia_ok": all(r.frecuencia_hz == config.frecuencia_objetivo_hz for r in grupo),
            }
        )
    return pd.DataFrame(filas)


def metadatos_registros(registros: list[RegistroAudio]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "indice": r.indice,
                "clase": r.clase,
                "etiqueta_binaria": r.etiqueta_binaria,
                "archivo": r.archivo,
                "ruta": str(r.ruta),
                "duracion_s": r.duracion_s,
                "frecuencia_hz": r.frecuencia_hz,
            }
            for r in registros
        ]
    )

