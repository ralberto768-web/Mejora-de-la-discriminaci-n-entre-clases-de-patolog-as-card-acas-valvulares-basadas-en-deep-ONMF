from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Iterable

from codigo.configuracion import ConfiguracionExperimento


CLASIFICADORES = ("SVM", "KNN", "UjaNet")
METRICAS = ("Accuracy", "Sensitivity", "Specificity", "Precision", "Score")
ARQUITECTURAS_HISTORICAS = (
    (9, 8, 7),
    (15, 10, 5),
    (10, 6, 4),
    (8, 5, 3),
)

# La primera ronda cubre distintas profundidades y niveles de compresion.
ARQUITECTURAS_INICIALES = (
    (20, 10),
    (18, 9),
    (15, 7),
    (12, 6),
    (10, 5),
    (8, 4),
    (6, 3),
    (20, 12, 6),
    (18, 12, 7),
    (16, 10, 5),
    (14, 9, 5),
    (12, 8, 4),
    (11, 7, 4),
    (10, 8, 6),
    (7, 5, 3),
    (20, 15, 10, 5),
    (18, 14, 9, 5),
    (16, 12, 8, 4),
    (14, 11, 8, 5),
    (12, 10, 7, 4),
    (10, 9, 7, 5),
    (10, 8, 6, 4),
    (9, 7, 5, 3),
    (20, 16, 12, 8, 4),
    (18, 14, 10, 7, 4),
    (15, 12, 9, 6, 3),
    (14, 11, 8, 6, 4),
    (12, 10, 8, 6, 4),
    (10, 9, 7, 5, 3),
)


@dataclass(frozen=True)
class ConfiguracionBusqueda:
    capas_minimas: int = 2
    capas_maximas: int = 5
    base_minima: int = 3
    base_maxima: int = 20
    rondas_sin_mejora: int = 3
    numero_finalistas: int = 5
    padres_por_ronda: int = 5
    mejora_minima: float = 1e-12


def etiqueta(rangos: Iterable[int]) -> str:
    return "-".join(str(int(valor)) for valor in rangos)


def clave_carpeta(rangos: Iterable[int]) -> str:
    return "_".join(str(int(valor)) for valor in rangos)


def nombre_h(numero_capas: int) -> str:
    return f"DeepONMF_H{numero_capas}"


def arquitectura_valida(
    rangos: tuple[int, ...],
    config: ConfiguracionBusqueda,
) -> bool:
    return (
        config.capas_minimas <= len(rangos) <= config.capas_maximas
        and all(config.base_minima <= valor <= config.base_maxima for valor in rangos)
        and all(a > b for a, b in zip(rangos, rangos[1:]))
    )


def configurar_experimento(
    base: ConfiguracionExperimento,
    rangos: tuple[int, ...],
    rapido: bool = False,
) -> ConfiguracionExperimento:
    cambios: dict[str, object] = {"rangos_deep_onmf": rangos}
    if rapido:
        cambios.update(
            {
                "iteraciones_onmf": 4,
                "pca_componentes_max": 16,
                "ujanet_epocas": 2,
                "ujanet_paciencia": 1,
                "ujanet_lote": 4,
            }
        )
    return replace(base, **cambios)


def guardar_configuracion(
    config_experimento: ConfiguracionExperimento,
    config_busqueda: ConfiguracionBusqueda,
    ruta: Path,
    rapido: bool,
) -> None:
    datos = config_experimento.como_diccionario()
    datos.update(
        {
            "modo_rapido_no_entregable": rapido,
            "numero_capas": len(config_experimento.rangos_deep_onmf),
            "matriz_temporal": nombre_h(len(config_experimento.rangos_deep_onmf)),
            "unico_parametro_metodologico_modificado": (
                "numero_de_capas_y_rangos_deep_onmf"
            ),
            "limites_busqueda": {
                "capas": [
                    config_busqueda.capas_minimas,
                    config_busqueda.capas_maximas,
                ],
                "bases": [
                    config_busqueda.base_minima,
                    config_busqueda.base_maxima,
                ],
            },
        }
    )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(datos, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

