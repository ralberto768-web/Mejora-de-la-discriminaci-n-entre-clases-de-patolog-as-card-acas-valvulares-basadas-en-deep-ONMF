from __future__ import annotations

import csv
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RegistroAudio:
    ruta: Path
    clase: str


def detectar_clase(ruta: Path, clases: tuple[str, ...]) -> str | None:
    """Detecta la clase a partir del nombre del archivo o de sus carpetas."""

    texto = " ".join([ruta.stem, *(parte for parte in ruta.parts)]).upper()
    for clase in clases:
        if f"_{clase}_" in texto or f"_{clase}-" in texto or f"NEW_{clase}" in texto:
            return clase
        if ruta.parent.name.upper().startswith(clase):
            return clase
        if ruta.stem.upper().startswith(clase + "-"):
            return clase
    return None


def descubrir_audios(carpeta: Path, clases: tuple[str, ...]) -> list[RegistroAudio]:
    """Busca archivos WAV y los devuelve ordenados por clase y nombre."""

    registros: list[RegistroAudio] = []
    for ruta in sorted(carpeta.rglob("*.wav")):
        if "CODE" in [parte.upper() for parte in ruta.parts]:
            continue
        clase = detectar_clase(ruta, clases)
        if clase is not None:
            registros.append(RegistroAudio(ruta=ruta, clase=clase))
    return sorted(registros, key=lambda r: (r.clase, str(r.ruta).lower()))


def leer_wav_mono(ruta: Path) -> tuple[np.ndarray, int]:
    """Lee un WAV con la libreria estandar y lo normaliza a mono en [-1, 1]."""

    with wave.open(str(ruta), "rb") as wav:
        canales = wav.getnchannels()
        fs = wav.getframerate()
        ancho = wav.getsampwidth()
        tramas = wav.getnframes()
        crudo = wav.readframes(tramas)

    if ancho == 1:
        datos = np.frombuffer(crudo, dtype=np.uint8).astype(np.float32)
        datos = (datos - 128.0) / 128.0
    elif ancho == 2:
        datos = np.frombuffer(crudo, dtype="<i2").astype(np.float32) / 32768.0
    elif ancho == 4:
        datos = np.frombuffer(crudo, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Ancho de muestra no soportado ({ancho} bytes): {ruta}")

    if canales > 1:
        datos = datos.reshape(-1, canales).mean(axis=1)

    return datos.astype(np.float32), fs


def guardar_wav_mono(ruta: Path, senal: np.ndarray, fs: int) -> None:
    """Guarda una senal mono normalizada como WAV de 16 bits."""

    ruta.parent.mkdir(parents=True, exist_ok=True)
    senal = np.asarray(senal, dtype=np.float32)
    senal = np.clip(senal, -1.0, 1.0)
    enteros = (senal * 32767.0).astype("<i2")
    with wave.open(str(ruta), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(fs)
        wav.writeframes(enteros.tobytes())


def remuestrear_lineal(senal: np.ndarray, fs_original: int, fs_objetivo: int) -> np.ndarray:
    """Remuestrea con interpolacion lineal para no depender de scipy."""

    if fs_original == fs_objetivo:
        return senal.astype(np.float32)
    duracion = len(senal) / float(fs_original)
    muestras_objetivo = max(1, int(round(duracion * fs_objetivo)))
    eje_original = np.linspace(0.0, duracion, num=len(senal), endpoint=False)
    eje_objetivo = np.linspace(0.0, duracion, num=muestras_objetivo, endpoint=False)
    return np.interp(eje_objetivo, eje_original, senal).astype(np.float32)


def ajustar_a_duracion(senal: np.ndarray, fs: int, duracion_segundos: float) -> np.ndarray:
    """Recorta o repite la senal para obtener una duracion fija."""

    muestras = int(round(fs * duracion_segundos))
    if len(senal) == muestras:
        return senal.astype(np.float32)
    if len(senal) > muestras:
        return senal[:muestras].astype(np.float32)

    repeticiones = int(math.ceil(muestras / max(1, len(senal))))
    extendida = np.tile(senal, repeticiones)
    return extendida[:muestras].astype(np.float32)


def guardar_metadatos_csv(ruta: Path, filas: list[dict[str, str]]) -> None:
    """Guarda un CSV legible con los audios preparados."""

    ruta.parent.mkdir(parents=True, exist_ok=True)
    campos = ["clase", "archivo_origen", "archivo_preparado"]
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


def dividir_estratificado(etiquetas: np.ndarray, proporcion_entrenamiento: float, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    """Divide indices manteniendo la proporcion de clases en entrenamiento y prueba."""

    rng = np.random.default_rng(semilla)
    entrenamiento: list[int] = []
    prueba: list[int] = []
    for etiqueta in sorted(set(etiquetas.tolist())):
        indices = np.where(etiquetas == etiqueta)[0]
        rng.shuffle(indices)
        corte = int(round(len(indices) * proporcion_entrenamiento))
        entrenamiento.extend(indices[:corte].tolist())
        prueba.extend(indices[corte:].tolist())
    rng.shuffle(entrenamiento)
    rng.shuffle(prueba)
    return np.array(entrenamiento, dtype=np.int64), np.array(prueba, dtype=np.int64)

