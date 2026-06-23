from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Configuracion:
    """Parámetros reproducibles extraídos del artículo objetivo."""

    raiz: Path
    carpeta_base_datos: Path = field(init=False)
    carpeta_resultados: Path = field(init=False)
    frecuencia_esperada_hz: int = 8000
    duracion_trama_s: float = 2.0
    solape_trama_s: float = 1.0
    longitud_ventana: int = 150
    salto_ventana: int = 75
    puntos_fft: int = 250
    rangos_onmf: tuple[int, int, int] = (9, 8, 7)
    iteraciones_onmf: int = 120
    penalizacion_ortogonal: float = 0.05
    semilla: int = 42
    rellenar_audios_cortos: bool = True
    clases: tuple[str, ...] = ("N", "AS", "MR", "MS", "MVP")

    def __post_init__(self) -> None:
        object.__setattr__(self, "carpeta_base_datos", self.raiz / "Bases de Datos")
        object.__setattr__(self, "carpeta_resultados", self.raiz / "resultados")

    @property
    def muestras_trama(self) -> int:
        return int(round(self.duracion_trama_s * self.frecuencia_esperada_hz))

    @property
    def salto_trama(self) -> int:
        return int(round((self.duracion_trama_s - self.solape_trama_s) * self.frecuencia_esperada_hz))

    @property
    def bins_frecuencia(self) -> int:
        return self.puntos_fft // 2 + 1

    def como_diccionario(self) -> dict[str, object]:
        datos = asdict(self)
        datos["raiz"] = str(self.raiz)
        datos["carpeta_base_datos"] = str(self.carpeta_base_datos)
        datos["carpeta_resultados"] = str(self.carpeta_resultados)
        datos["muestras_trama"] = self.muestras_trama
        datos["salto_trama"] = self.salto_trama
        datos["bins_frecuencia"] = self.bins_frecuencia
        return datos
