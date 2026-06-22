from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from codigo.configuracion import ConfiguracionExperimento


DISTRIBUCIONES: dict[str, tuple[int, int, int]] = {
    "15_10_5": (15, 10, 5),
    "10_6_4": (10, 6, 4),
    "8_5_3": (8, 5, 3),
}

DISTRIBUCION_REFERENCIA = (9, 8, 7)
REPRESENTACIONES_DEEP = ("DeepONMF_W", "DeepONMF_H3")
CLASIFICADORES = ("SVM", "KNN", "UjaNet")


def etiqueta_distribucion(rangos: tuple[int, int, int]) -> str:
    return "-".join(str(valor) for valor in rangos)


def configurar_rangos(
    base: ConfiguracionExperimento,
    rangos: tuple[int, int, int],
    rapido: bool = False,
) -> ConfiguracionExperimento:
    """Copia la configuración original cambiando solo los rangos.

    El modo rápido reduce iteraciones y épocas exclusivamente para comprobar
    que el flujo técnico funciona; sus resultados nunca se mezclan con los
    del experimento completo.
    """

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


def guardar_configuracion_prueba(
    config: ConfiguracionExperimento,
    ruta: Path,
    modo_rapido: bool,
) -> None:
    datos = config.como_diccionario()
    datos["modo_rapido_no_entregable"] = modo_rapido
    datos["unico_parametro_metodologico_modificado"] = "rangos_deep_onmf"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
