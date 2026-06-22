from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import wave

import numpy as np
import pandas as pd

from .configuracion import BASE_ORIGINAL, CLASES, SNR_OBJETIVOS, nombre_base_snr


MAX_PCM16 = 32767
ESCALA_PCM16 = 32768.0


def _semilla_audio(semilla: int, ruta_relativa: Path, snr_db: float) -> int:
    texto = f"{semilla}|{ruta_relativa.as_posix()}|{snr_db:+.1f}".encode("utf-8")
    resumen = hashlib.sha256(texto).digest()
    return int.from_bytes(resumen[:8], "little", signed=False)


def _leer_pcm16_mono(ruta: Path) -> tuple[np.ndarray, dict[str, int]]:
    with wave.open(str(ruta), "rb") as wav:
        parametros = {
            "canales": wav.getnchannels(),
            "frecuencia_hz": wav.getframerate(),
            "bytes_muestra": wav.getsampwidth(),
            "muestras": wav.getnframes(),
        }
        bruto = wav.readframes(wav.getnframes())
    if parametros["canales"] != 1 or parametros["bytes_muestra"] != 2:
        raise ValueError(f"Solo se admite WAV mono PCM16: {ruta}")
    senal = np.frombuffer(bruto, dtype="<i2").astype(np.float64) / ESCALA_PCM16
    return senal, parametros


def _guardar_pcm16_mono(ruta: Path, senal: np.ndarray, frecuencia_hz: int) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.rint(senal * ESCALA_PCM16)
    if np.max(pcm) > MAX_PCM16 or np.min(pcm) < -ESCALA_PCM16:
        raise AssertionError(f"La senal excede PCM16 antes de guardarse: {ruta}")
    with wave.open(str(ruta), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(frecuencia_hz)
        wav.writeframes(pcm.astype("<i2").tobytes())


def anadir_awgn(
    senal: np.ndarray,
    snr_db: float,
    semilla: int,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    senal = np.asarray(senal, dtype=np.float64)
    potencia_senal = float(np.mean(senal**2))
    if potencia_senal <= 0.0:
        raise ValueError("No se puede definir la SNR de una senal de potencia nula")

    rng = np.random.default_rng(semilla)
    ruido = rng.standard_normal(senal.shape)
    ruido -= float(np.mean(ruido))
    potencia_ruido = float(np.mean(ruido**2))
    potencia_objetivo = potencia_senal / (10.0 ** (snr_db / 10.0))
    ruido *= np.sqrt(potencia_objetivo / max(potencia_ruido, 1e-12))

    mezcla = senal + ruido
    limite = MAX_PCM16 / ESCALA_PCM16
    maximo = float(np.max(np.abs(mezcla)))
    factor = min(1.0, limite / maximo) if maximo > 0.0 else 1.0
    mezcla *= factor
    referencia = senal * factor
    pcm_sin_recortar = np.rint(mezcla * ESCALA_PCM16)
    recortes = int(np.sum((pcm_sin_recortar < -ESCALA_PCM16) | (pcm_sin_recortar > MAX_PCM16)))
    salida = np.clip(pcm_sin_recortar, -ESCALA_PCM16, MAX_PCM16) / ESCALA_PCM16
    ruido_efectivo = salida - referencia
    snr_real = float(
        10.0
        * np.log10(
            max(float(np.mean(referencia**2)), 1e-12)
            / max(float(np.mean(ruido_efectivo**2)), 1e-12)
        )
    )
    return salida, {
        "semilla_ruido": int(semilla),
        "snr_objetivo_db": float(snr_db),
        "snr_real_db": snr_real,
        "error_snr_db": snr_real - float(snr_db),
        "potencia_senal": potencia_senal,
        "potencia_ruido_objetivo": potencia_objetivo,
        "factor_antirecorte": factor,
        "escalado_antirecorte": factor < 1.0,
        "muestras_saturadas": recortes,
    }


def _clase_de_carpeta(nombre: str) -> str | None:
    for clase in ("MVP", "AS", "MR", "MS", "N"):
        if nombre.startswith(f"{clase}_"):
            return clase
    return None


def _archivos_origen(origen: Path, limite_por_clase: int) -> list[tuple[str, Path, Path]]:
    archivos: list[tuple[str, Path, Path]] = []
    conteos = {clase: 0 for clase in CLASES}
    for carpeta in sorted(origen.iterdir()):
        if not carpeta.is_dir():
            continue
        clase = _clase_de_carpeta(carpeta.name)
        if clase is None:
            continue
        for ruta in sorted(carpeta.glob("*.wav")):
            if limite_por_clase and conteos[clase] >= limite_por_clase:
                continue
            archivos.append((clase, ruta, ruta.relative_to(origen)))
            conteos[clase] += 1
    return archivos


def preparar_bases(
    origen: Path,
    destino: Path,
    semilla: int,
    limite_por_clase: int = 0,
) -> pd.DataFrame:
    archivos = _archivos_origen(origen, limite_por_clase)
    ruta_auditoria = destino / "auditoria_generacion_bases.csv"
    bases_esperadas = [BASE_ORIGINAL, *[nombre_base_snr(snr) for snr in SNR_OBJETIVOS]]
    if ruta_auditoria.exists():
        auditoria = pd.read_csv(ruta_auditoria, encoding="utf-8-sig")
        if (
            len(auditoria) == len(archivos) * len(bases_esperadas)
            and set(auditoria["base"].astype(str)) == set(bases_esperadas)
        ):
            return auditoria

    filas: list[dict[str, object]] = []
    for posicion, (clase, ruta_origen, relativa) in enumerate(archivos, start=1):
        senal, parametros = _leer_pcm16_mono(ruta_origen)
        ruta_original = destino / BASE_ORIGINAL / relativa
        ruta_original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ruta_origen, ruta_original)
        filas.append(
            {
                "base": BASE_ORIGINAL,
                "clase": clase,
                "archivo": ruta_origen.name,
                "ruta_relativa": relativa.as_posix(),
                "frecuencia_hz": parametros["frecuencia_hz"],
                "muestras": parametros["muestras"],
                "snr_objetivo_db": np.nan,
                "snr_real_db": np.nan,
                "error_snr_db": np.nan,
                "factor_antirecorte": 1.0,
                "escalado_antirecorte": False,
                "muestras_saturadas": 0,
            }
        )
        for snr_db in SNR_OBJETIVOS:
            base = nombre_base_snr(snr_db)
            semilla_audio = _semilla_audio(semilla, relativa, snr_db)
            ruidosa, auditoria = anadir_awgn(senal, snr_db, semilla_audio)
            ruta_destino = destino / base / relativa
            _guardar_pcm16_mono(ruta_destino, ruidosa, parametros["frecuencia_hz"])
            filas.append(
                {
                    "base": base,
                    "clase": clase,
                    "archivo": ruta_origen.name,
                    "ruta_relativa": relativa.as_posix(),
                    "frecuencia_hz": parametros["frecuencia_hz"],
                    "muestras": parametros["muestras"],
                    **auditoria,
                }
            )
        if posicion == 1 or posicion % 100 == 0 or posicion == len(archivos):
            print(f"[bases] {posicion}/{len(archivos)} audios originales")

    destino.mkdir(parents=True, exist_ok=True)
    auditoria = pd.DataFrame(filas)
    auditoria.to_csv(ruta_auditoria, index=False, encoding="utf-8-sig")
    return auditoria
