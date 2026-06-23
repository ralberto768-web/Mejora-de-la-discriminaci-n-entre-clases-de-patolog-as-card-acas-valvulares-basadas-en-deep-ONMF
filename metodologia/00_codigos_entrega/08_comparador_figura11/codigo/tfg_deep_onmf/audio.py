from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np

from .configuracion import Configuracion


@dataclass(frozen=True)
class RegistroAudio:
    clase: str
    ruta: Path
    frecuencia_hz: int
    canales: int
    bytes_muestra: int
    muestras: int
    duracion_s: float


@dataclass
class DatosClase:
    clase: str
    matriz: np.ndarray
    audios_usados: list[RegistroAudio]
    audios_descartados: list[RegistroAudio]
    rangos_columnas: dict[str, tuple[int, int]]


def inferir_clase(nombre_carpeta: str) -> str | None:
    prefijos = {
        "N_": "N",
        "AS_": "AS",
        "MR_": "MR",
        "MS_": "MS",
        "MVP_": "MVP",
    }
    for prefijo, clase in prefijos.items():
        if nombre_carpeta.startswith(prefijo):
            return clase
    return None


def descubrir_audios(carpeta_base: Path, clases: tuple[str, ...]) -> list[RegistroAudio]:
    registros: list[RegistroAudio] = []
    for carpeta in sorted(carpeta_base.iterdir()):
        if not carpeta.is_dir() or carpeta.name == "Code":
            continue
        clase = inferir_clase(carpeta.name)
        if clase is None or clase not in clases:
            continue
        for ruta in sorted(carpeta.glob("*.wav")):
            with wave.open(str(ruta), "rb") as wav:
                frecuencia = wav.getframerate()
                muestras = wav.getnframes()
                registros.append(
                    RegistroAudio(
                        clase=clase,
                        ruta=ruta,
                        frecuencia_hz=frecuencia,
                        canales=wav.getnchannels(),
                        bytes_muestra=wav.getsampwidth(),
                        muestras=muestras,
                        duracion_s=muestras / frecuencia,
                    )
                )
    return registros


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
    return datos, frecuencia


def dividir_en_tramas(senal: np.ndarray, configuracion: Configuracion) -> list[np.ndarray]:
    longitud = configuracion.muestras_trama
    salto = configuracion.salto_trama
    if len(senal) < longitud:
        if configuracion.rellenar_audios_cortos:
            senal = np.pad(senal, (0, longitud - len(senal)), mode="constant")
            return [senal]
        return []
    tramas = []
    for inicio in range(0, len(senal) - longitud + 1, salto):
        tramas.append(senal[inicio : inicio + longitud])
    return tramas


def espectrograma_magnitud(trama: np.ndarray, configuracion: Configuracion) -> np.ndarray:
    ventana = np.hamming(configuracion.longitud_ventana)
    inicios = np.arange(0, len(trama) - configuracion.longitud_ventana + 1, configuracion.salto_ventana)
    segmentos = np.stack([trama[i : i + configuracion.longitud_ventana] for i in inicios], axis=0)
    segmentos = segmentos * ventana[None, :]
    espectro = np.fft.rfft(segmentos, n=configuracion.puntos_fft, axis=1)
    magnitud = np.abs(espectro).T
    return np.maximum(magnitud, 1e-12).astype(np.float64)


def construir_matriz_clase(
    clase: str,
    registros: list[RegistroAudio],
    configuracion: Configuracion,
) -> DatosClase:
    espectrogramas: list[np.ndarray] = []
    audios_usados: list[RegistroAudio] = []
    audios_descartados: list[RegistroAudio] = []
    rangos: dict[str, tuple[int, int]] = {}
    columna_actual = 0

    for registro in registros:
        if registro.clase != clase:
            continue
        senal, frecuencia = leer_wav_normalizado(registro.ruta)
        if frecuencia != configuracion.frecuencia_esperada_hz:
            raise ValueError(
                f"{registro.ruta} tiene {frecuencia} Hz, pero se esperaban "
                f"{configuracion.frecuencia_esperada_hz} Hz"
            )

        tramas = dividir_en_tramas(senal, configuracion)
        if not tramas:
            audios_descartados.append(registro)
            continue

        inicio_audio = columna_actual
        for trama in tramas:
            matriz_trama = espectrograma_magnitud(trama, configuracion)
            espectrogramas.append(matriz_trama)
            columna_actual += matriz_trama.shape[1]
        rangos[str(registro.ruta)] = (inicio_audio, columna_actual)
        audios_usados.append(registro)

    if not espectrogramas:
        raise ValueError(f"No se han podido crear espectrogramas para la clase {clase}")

    matriz = np.concatenate(espectrogramas, axis=1)
    return DatosClase(
        clase=clase,
        matriz=matriz,
        audios_usados=audios_usados,
        audios_descartados=audios_descartados,
        rangos_columnas=rangos,
    )


def construir_matriz_audio(registro: RegistroAudio, configuracion: Configuracion) -> np.ndarray:
    senal, frecuencia = leer_wav_normalizado(registro.ruta)
    if frecuencia != configuracion.frecuencia_esperada_hz:
        raise ValueError(
            f"{registro.ruta} tiene {frecuencia} Hz, pero se esperaban "
            f"{configuracion.frecuencia_esperada_hz} Hz"
        )
    tramas = dividir_en_tramas(senal, configuracion)
    espectrogramas = [espectrograma_magnitud(trama, configuracion) for trama in tramas]
    return np.concatenate(espectrogramas, axis=1)
