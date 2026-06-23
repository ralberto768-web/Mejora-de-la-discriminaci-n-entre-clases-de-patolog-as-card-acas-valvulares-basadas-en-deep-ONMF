from __future__ import annotations

import numpy as np

from .configuracion import Configuracion


def _hz_a_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_a_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def crear_banco_mel(fs: int, longitud_fft: int, bandas_mel: int, frecuencia_maxima: float) -> np.ndarray:
    """Crea un banco de filtros mel triangular equivalente al usado en el articulo."""

    minimo_mel = _hz_a_mel(np.array([0.0]))[0]
    maximo_mel = _hz_a_mel(np.array([frecuencia_maxima]))[0]
    puntos_mel = np.linspace(minimo_mel, maximo_mel, bandas_mel + 2)
    puntos_hz = _mel_a_hz(puntos_mel)
    bins = np.floor((longitud_fft + 1) * puntos_hz / fs).astype(int)

    banco = np.zeros((bandas_mel, longitud_fft // 2 + 1), dtype=np.float32)
    for banda in range(1, bandas_mel + 1):
        izquierda, centro, derecha = bins[banda - 1], bins[banda], bins[banda + 1]
        if centro == izquierda:
            centro += 1
        if derecha == centro:
            derecha += 1
        for k in range(izquierda, centro):
            if 0 <= k < banco.shape[1]:
                banco[banda - 1, k] = (k - izquierda) / max(1, centro - izquierda)
        for k in range(centro, derecha):
            if 0 <= k < banco.shape[1]:
                banco[banda - 1, k] = (derecha - k) / max(1, derecha - centro)
    return banco


def espectrograma_log_mel(senal: np.ndarray, cfg: Configuracion) -> np.ndarray:
    """Convierte una senal preparada en un espectrograma log-mel."""

    longitud_trama = int(round(cfg.duracion_trama * cfg.fs_objetivo))
    salto = int(round(cfg.salto_trama * cfg.fs_objetivo))
    numero_tramas = int(np.ceil((cfg.duracion_segmento - cfg.duracion_trama) / cfg.salto_trama))
    ventana = np.hanning(longitud_trama).astype(np.float32)
    banco_mel = crear_banco_mel(cfg.fs_objetivo, cfg.longitud_fft, cfg.bandas_mel, cfg.fs_objetivo / 2)

    potencia = np.zeros((cfg.longitud_fft // 2 + 1, numero_tramas), dtype=np.float32)
    for indice in range(numero_tramas):
        inicio = indice * salto
        fin = inicio + longitud_trama
        fragmento = np.zeros(longitud_trama, dtype=np.float32)
        disponible = senal[inicio:fin]
        fragmento[: len(disponible)] = disponible
        fft = np.fft.rfft(fragmento * ventana, n=cfg.longitud_fft)
        potencia[:, indice] = (np.abs(fft) ** 2).astype(np.float32)

    mel = banco_mel @ potencia
    return np.log10(mel + cfg.epsilon_log).astype(np.float32)

