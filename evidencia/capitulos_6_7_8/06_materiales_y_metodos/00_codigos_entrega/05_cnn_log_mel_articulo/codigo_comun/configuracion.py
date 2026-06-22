from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Configuracion:
    """Parametros principales del experimento descrito en el articulo."""

    raiz_implementacion: Path
    raiz_proyecto: Path
    carpeta_bases_datos: Path
    carpeta_codigo_original: Path
    fs_objetivo: int = 8000
    duracion_segmento: float = 2.0
    duracion_trama: float = 0.025
    salto_trama: float = 0.01
    bandas_mel: int = 40
    longitud_fft: int = 512
    epsilon_log: float = 1e-6
    proporcion_entrenamiento: float = 0.70
    semilla: int = 42
    clases: tuple[str, ...] = ("AS", "MR", "MS", "MVP", "N")
    carpeta_resultados_personalizada: Path | None = None

    @property
    def etiqueta_duracion(self) -> str:
        return str(self.duracion_segmento).replace(".", "_")

    @property
    def carpeta_datos_preparados(self) -> Path:
        return self.carpeta_resultados / "datos_preparados" / f"segmentos_{self.etiqueta_duracion}s"

    @property
    def carpeta_resultados(self) -> Path:
        if self.carpeta_resultados_personalizada is not None:
            return self.carpeta_resultados_personalizada
        return self.raiz_implementacion / "resultados"

    @property
    def archivo_caracteristicas(self) -> Path:
        return self.carpeta_resultados / f"caracteristicas_log_mel_{self.etiqueta_duracion}s.npz"


def _resolver_bases_datos(raiz_implementacion: Path, raiz_proyecto: Path) -> Path:
    env = os.environ.get("TFG_DATOS")
    if env:
        return Path(env).expanduser().resolve()
    for base in [Path.cwd(), raiz_implementacion, raiz_proyecto, *raiz_implementacion.parents]:
        candidata = base / "Bases de Datos"
        if candidata.exists():
            return candidata.resolve()
    return raiz_proyecto / "Bases de Datos"


def _resolver_salida(raiz_implementacion: Path) -> Path | None:
    env = os.environ.get("TFG_SALIDA")
    if env:
        return Path(env).expanduser().resolve()
    return None


def crear_configuracion(duracion_segmento: float = 2.0, semilla: int = 42) -> Configuracion:
    """Crea la configuracion usando rutas relativas a la carpeta Implementacion."""

    raiz_implementacion = Path(__file__).resolve().parents[1]
    raiz_proyecto = raiz_implementacion.parent
    return Configuracion(
        raiz_implementacion=raiz_implementacion,
        raiz_proyecto=raiz_proyecto,
        carpeta_bases_datos=_resolver_bases_datos(raiz_implementacion, raiz_proyecto),
        carpeta_codigo_original=raiz_proyecto / "Heart-sound-classification-main",
        duracion_segmento=duracion_segmento,
        semilla=semilla,
        carpeta_resultados_personalizada=_resolver_salida(raiz_implementacion),
    )
