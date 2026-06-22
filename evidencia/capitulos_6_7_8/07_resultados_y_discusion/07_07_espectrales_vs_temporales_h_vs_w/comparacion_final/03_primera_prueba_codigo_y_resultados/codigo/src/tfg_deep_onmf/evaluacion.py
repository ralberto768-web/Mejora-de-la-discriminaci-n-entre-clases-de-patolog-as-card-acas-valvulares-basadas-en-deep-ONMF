from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .audio import RegistroAudio, construir_matriz_audio
from .configuracion import Configuracion
from .onmf import proyectar_sobre_w


@dataclass(frozen=True)
class DivisionDatos:
    entrenamiento: list[RegistroAudio]
    test: list[RegistroAudio]


def dividir_entrenamiento_test(
    registros: list[RegistroAudio],
    clases: tuple[str, ...],
    porcentaje_entrenamiento: int,
    semilla: int,
) -> DivisionDatos:
    rng = np.random.default_rng(semilla + porcentaje_entrenamiento)
    entrenamiento: list[RegistroAudio] = []
    test: list[RegistroAudio] = []

    for clase in clases:
        registros_clase = sorted([r for r in registros if r.clase == clase], key=lambda r: r.ruta.name)
        indices = np.arange(len(registros_clase))
        rng.shuffle(indices)
        n_entrenamiento = int(round(len(registros_clase) * porcentaje_entrenamiento / 100))
        indices_entrenamiento = set(indices[:n_entrenamiento].tolist())

        for indice, registro in enumerate(registros_clase):
            if indice in indices_entrenamiento:
                entrenamiento.append(registro)
            else:
                test.append(registro)

    return DivisionDatos(entrenamiento=entrenamiento, test=test)


def caracteristicas_sbv_por_audio(
    registros: list[RegistroAudio],
    w_por_clase: dict[str, np.ndarray],
    configuracion: Configuracion,
    particion: str,
) -> pd.DataFrame:
    filas: list[dict[str, object]] = []
    for registro in registros:
        matriz = construir_matriz_audio(registro, configuracion)
        h, error = proyectar_sobre_w(matriz, w_por_clase[registro.clase])
        vector = h.mean(axis=1)
        fila: dict[str, object] = {
            "particion": particion,
            "clase": registro.clase,
            "archivo": registro.ruta.name,
            "ruta": str(registro.ruta),
            "duracion_s": registro.duracion_s,
            "rellenado_a_2s": registro.duracion_s < configuracion.duracion_trama_s,
            "error_base_clase_real": error,
        }
        for indice, valor in enumerate(vector, start=1):
            fila[f"SBV_{indice}"] = float(valor)
        filas.append(fila)
    return pd.DataFrame(filas)


def evaluar_por_reconstruccion(
    registros: list[RegistroAudio],
    w_por_clase: dict[str, np.ndarray],
    configuracion: Configuracion,
    particion: str,
) -> pd.DataFrame:
    filas: list[dict[str, object]] = []
    clases_modelo = tuple(w_por_clase.keys())
    for registro in registros:
        matriz = construir_matriz_audio(registro, configuracion)
        errores = {}
        for clase_modelo in clases_modelo:
            _, error = proyectar_sobre_w(matriz, w_por_clase[clase_modelo])
            errores[clase_modelo] = error
        predicha = min(errores, key=errores.get)
        fila: dict[str, object] = {
            "particion": particion,
            "archivo": registro.ruta.name,
            "clase_real": registro.clase,
            "clase_predicha": predicha,
            "correcto": predicha == registro.clase,
            "duracion_s": registro.duracion_s,
            "rellenado_a_2s": registro.duracion_s < configuracion.duracion_trama_s,
        }
        for clase_modelo, error in errores.items():
            fila[f"error_vs_{clase_modelo}"] = error
        filas.append(fila)
    return pd.DataFrame(filas)


def resumen_metricas(predicciones: pd.DataFrame, clases: tuple[str, ...]) -> pd.DataFrame:
    y_real = predicciones["clase_real"].to_numpy()
    y_pred = predicciones["clase_predicha"].to_numpy()
    precision, recall, f1, soporte = precision_recall_fscore_support(
        y_real,
        y_pred,
        labels=list(clases),
        zero_division=0,
    )
    filas: list[dict[str, object]] = []
    for clase, p, r, f, s in zip(clases, precision, recall, f1, soporte):
        filas.append(
            {
                "clase": clase,
                "precision": p,
                "sensibilidad_recall": r,
                "f1": f,
                "soporte": int(s),
            }
        )
    filas.append(
        {
            "clase": "TOTAL",
            "precision": float(np.mean(precision)),
            "sensibilidad_recall": float(np.mean(recall)),
            "f1": float(np.mean(f1)),
            "soporte": int(len(predicciones)),
        }
    )
    return pd.DataFrame(filas)


def matriz_confusion_df(predicciones: pd.DataFrame, clases: tuple[str, ...]) -> pd.DataFrame:
    matriz = confusion_matrix(
        predicciones["clase_real"],
        predicciones["clase_predicha"],
        labels=list(clases),
    )
    return pd.DataFrame(matriz, index=[f"real_{c}" for c in clases], columns=[f"pred_{c}" for c in clases])


def resumen_particion(
    registros: list[RegistroAudio],
    entrenamiento: list[RegistroAudio],
    test: list[RegistroAudio],
    clases: tuple[str, ...],
    duracion_trama_s: float,
) -> pd.DataFrame:
    filas = []
    for clase in clases:
        todos = [r for r in registros if r.clase == clase]
        train = [r for r in entrenamiento if r.clase == clase]
        test_clase = [r for r in test if r.clase == clase]
        filas.append(
            {
                "clase": clase,
                "total": len(todos),
                "entrenamiento": len(train),
                "test": len(test_clase),
                "rellenados_entrenamiento": sum(r.duracion_s < duracion_trama_s for r in train),
                "rellenados_test": sum(r.duracion_s < duracion_trama_s for r in test_clase),
            }
        )
    return pd.DataFrame(filas)
