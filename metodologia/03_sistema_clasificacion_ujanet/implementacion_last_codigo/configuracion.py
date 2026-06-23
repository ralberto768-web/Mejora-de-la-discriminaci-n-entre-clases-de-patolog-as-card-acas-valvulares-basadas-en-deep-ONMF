from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any


CLASES: tuple[str, ...] = ("N", "AS", "MR", "MS", "MVP")
CLASES_ANOMALAS: tuple[str, ...] = ("AS", "MR", "MS", "MVP")
REPRESENTACIONES: tuple[str, ...] = (
    "STFT",
    "MFCC",
    "MelSpectrogram",
    "LogMelSpectrogram",
    "DeepONMF_W",
    "DeepONMF_H3",
)


@dataclass(frozen=True)
class ConfiguracionExperimento:
    """Parametros reproducibles del experimento final.

    Todos los valores importantes quedan concentrados aqui para que el informe
    pueda explicar exactamente como se ha generado cada resultado.
    """

    semilla: int = 42
    frecuencia_objetivo_hz: int = 8000
    duracion_trama_pcg_s: float = 2.0
    solape_trama_pcg_s: float = 1.0
    ventana_stft_muestras: int = 150
    salto_stft_muestras: int = 75
    puntos_fft: int = 250
    bandas_mel: int = 40
    coeficientes_mfcc: int = 13
    rangos_deep_onmf: tuple[int, int, int] = (9, 8, 7)
    iteraciones_onmf: int = 60
    penalizacion_ortogonal: float = 0.05
    folds: int = 5
    pca_componentes_max: int = 128
    svm_c: float = 1.0
    knn_vecinos: int = 5
    ujanet_epocas: int = 30
    ujanet_paciencia: int = 10
    ujanet_lote: int = 16
    ujanet_lr: float = 0.001

    @property
    def muestras_trama_pcg(self) -> int:
        return int(round(self.duracion_trama_pcg_s * self.frecuencia_objetivo_hz))

    @property
    def salto_trama_pcg(self) -> int:
        avance_s = self.duracion_trama_pcg_s - self.solape_trama_pcg_s
        return int(round(avance_s * self.frecuencia_objetivo_hz))

    @property
    def bins_frecuencia(self) -> int:
        return self.puntos_fft // 2 + 1

    @property
    def tramas_stft_por_segmento(self) -> int:
        return 1 + (self.muestras_trama_pcg - self.ventana_stft_muestras) // self.salto_stft_muestras

    def como_diccionario(self) -> dict[str, Any]:
        datos = asdict(self)
        datos["clases"] = list(CLASES)
        datos["clases_anomalas"] = list(CLASES_ANOMALAS)
        datos["representaciones"] = list(REPRESENTACIONES)
        datos["muestras_trama_pcg"] = self.muestras_trama_pcg
        datos["salto_trama_pcg"] = self.salto_trama_pcg
        datos["bins_frecuencia"] = self.bins_frecuencia
        datos["tramas_stft_por_segmento"] = self.tramas_stft_por_segmento
        return datos


def cargar_configuracion(ruta: Path | None = None) -> ConfiguracionExperimento:
    if ruta is None or not ruta.exists():
        return ConfiguracionExperimento()
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    permitidos = {campo.name for campo in ConfiguracionExperimento.__dataclass_fields__.values()}
    filtrados = {clave: valor for clave, valor in datos.items() if clave in permitidos}
    if "rangos_deep_onmf" in filtrados:
        filtrados["rangos_deep_onmf"] = tuple(filtrados["rangos_deep_onmf"])
    return ConfiguracionExperimento(**filtrados)


def guardar_configuracion(config: ConfiguracionExperimento, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(config.como_diccionario(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def aplicar_modo_rapido(config: ConfiguracionExperimento) -> ConfiguracionExperimento:
    """Reduce el coste para pruebas sin cambiar el contrato de salida."""

    return ConfiguracionExperimento(
        semilla=config.semilla,
        frecuencia_objetivo_hz=config.frecuencia_objetivo_hz,
        duracion_trama_pcg_s=config.duracion_trama_pcg_s,
        solape_trama_pcg_s=config.solape_trama_pcg_s,
        ventana_stft_muestras=config.ventana_stft_muestras,
        salto_stft_muestras=config.salto_stft_muestras,
        puntos_fft=config.puntos_fft,
        bandas_mel=config.bandas_mel,
        coeficientes_mfcc=config.coeficientes_mfcc,
        rangos_deep_onmf=config.rangos_deep_onmf,
        iteraciones_onmf=4,
        penalizacion_ortogonal=config.penalizacion_ortogonal,
        folds=config.folds,
        pca_componentes_max=16,
        svm_c=config.svm_c,
        knn_vecinos=config.knn_vecinos,
        ujanet_epocas=2,
        ujanet_paciencia=1,
        ujanet_lote=4,
        ujanet_lr=config.ujanet_lr,
    )
