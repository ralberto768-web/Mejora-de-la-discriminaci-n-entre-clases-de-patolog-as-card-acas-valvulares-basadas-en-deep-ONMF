from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from codigo.clasificadores import crear_folds
from codigo.configuracion import CLASES, ConfiguracionExperimento
from codigo.datos import RegistroAudio, tabla_auditoria


def auditar_base(
    registros: list[RegistroAudio],
    config: ConfiguracionExperimento,
    carpeta: Path,
    exigir_base_completa: bool,
) -> pd.DataFrame:
    auditoria = tabla_auditoria(registros, config)
    carpeta.mkdir(parents=True, exist_ok=True)
    auditoria.to_csv(carpeta / "auditoria_base_datos.csv", index=False, encoding="utf-8-sig")
    if exigir_base_completa:
        if len(registros) != 1000:
            raise AssertionError(f"Se esperaban 1000 audios y se han detectado {len(registros)}")
        conteos = {clase: sum(r.clase == clase for r in registros) for clase in CLASES}
        if any(conteos[clase] != 200 for clase in CLASES):
            raise AssertionError(f"Distribución de clases incorrecta: {conteos}")
    return auditoria


def crear_y_auditar_folds(
    metadatos: pd.DataFrame,
    config: ConfiguracionExperimento,
    ruta_particiones_originales: Path,
    carpeta: Path,
    exigir_protocolo_completo: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y_multi = metadatos["clase"].map({clase: i for i, clase in enumerate(CLASES)}).to_numpy(dtype=int)
    folds = crear_folds(y_multi, config)
    filas: list[dict[str, object]] = []
    for numero, (idx_train, idx_test) in enumerate(folds, start=1):
        conteos_test = metadatos.iloc[idx_test]["clase"].value_counts()
        fila: dict[str, object] = {
            "fold": numero,
            "entrenamiento": len(idx_train),
            "test": len(idx_test),
        }
        for clase in CLASES:
            fila[f"test_{clase}"] = int(conteos_test.get(clase, 0))
        filas.append(fila)
    protocolo = pd.DataFrame(filas)

    identidad = pd.DataFrame()
    if exigir_protocolo_completo:
        if len(folds) != 5:
            raise AssertionError(f"Se esperaban cinco folds y se han creado {len(folds)}")
        if not ((protocolo["entrenamiento"] == 800) & (protocolo["test"] == 200)).all():
            raise AssertionError("Algún fold no contiene 800 audios de entrenamiento y 200 de test")
        for clase in CLASES:
            if not (protocolo[f"test_{clase}"] == 40).all():
                raise AssertionError(f"Los folds no contienen 40 audios de test de la clase {clase}")
        identidad = comprobar_identidad_folds(folds, metadatos, ruta_particiones_originales)
        if not identidad["coincide"].all():
            raise AssertionError("Los folds nuevos no coinciden con los folds del experimento original")

    carpeta.mkdir(parents=True, exist_ok=True)
    protocolo.to_csv(carpeta / "validacion_protocolo.csv", index=False, encoding="utf-8-sig")
    if not identidad.empty:
        identidad.to_csv(carpeta / "auditoria_identidad_folds.csv", index=False, encoding="utf-8-sig")
    return folds


def comprobar_identidad_folds(
    folds: list[tuple[np.ndarray, np.ndarray]],
    metadatos: pd.DataFrame,
    ruta_particiones_originales: Path,
) -> pd.DataFrame:
    original = pd.read_csv(ruta_particiones_originales, encoding="utf-8-sig")
    original["clave"] = original["clase"].astype(str) + "/" + original["archivo"].astype(str)
    filas = []
    for numero, (_, idx_test) in enumerate(folds, start=1):
        claves_nuevas = set(
            metadatos.iloc[idx_test]["clase"].astype(str)
            + "/"
            + metadatos.iloc[idx_test]["archivo"].astype(str)
        )
        claves_originales = set(
            original.loc[
                (original["fold"] == numero) & (original["particion"] == "test"),
                "clave",
            ]
        )
        filas.append(
            {
                "fold": numero,
                "test_nuevo": len(claves_nuevas),
                "test_original": len(claves_originales),
                "coincidencias": len(claves_nuevas & claves_originales),
                "solo_nuevo": len(claves_nuevas - claves_originales),
                "solo_original": len(claves_originales - claves_nuevas),
                "coincide": claves_nuevas == claves_originales,
            }
        )
    return pd.DataFrame(filas)
