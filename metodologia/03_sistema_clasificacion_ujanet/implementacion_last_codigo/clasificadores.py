from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .configuracion import (
    CLASES,
    REPRESENTACIONES,
    ConfiguracionExperimento,
)
from .metricas import (
    matriz_confusion_binaria,
    metricas_binarias,
    metricas_multiclase,
)
from .representaciones import vectorizar_para_clasicos


@dataclass(frozen=True)
class ResultadoClasificacion:
    resumen_binario: pd.DataFrame
    resumen_multiclase: pd.DataFrame


def etiquetas_numericas(metadatos: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_multi = metadatos["clase"].map({clase: i for i, clase in enumerate(CLASES)}).to_numpy(dtype=int)
    y_bin = (metadatos["etiqueta_binaria"] == "anomalo").to_numpy(dtype=int)
    return y_bin, y_multi


def crear_folds(y_multi: np.ndarray, config: ConfiguracionExperimento) -> list[tuple[np.ndarray, np.ndarray]]:
    conteos = np.bincount(y_multi, minlength=len(CLASES))
    n_splits = int(min(config.folds, max(2, conteos.min())))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.semilla)
    dummy = np.zeros(len(y_multi))
    return list(splitter.split(dummy, y_multi))


def guardar_particiones(
    folds: list[tuple[np.ndarray, np.ndarray]],
    metadatos: pd.DataFrame,
    carpeta: Path,
) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = []
    for fold, (idx_train, idx_test) in enumerate(folds, start=1):
        for particion, indices in [("entrenamiento", idx_train), ("test", idx_test)]:
            for indice in indices:
                fila = metadatos.iloc[int(indice)].to_dict()
                filas.append({"fold": fold, "particion": particion, **fila})
    pd.DataFrame(filas).to_csv(carpeta / "particiones_5fold.csv", index=False, encoding="utf-8-sig")


def _pipeline_svm(config: ConfiguracionExperimento, n_train: int, n_features: int) -> Pipeline:
    pasos: list[tuple[str, object]] = [("escalado", StandardScaler())]
    n_comp = min(config.pca_componentes_max, n_train - 1, n_features)
    if n_features > n_comp and n_comp >= 2:
        pasos.append(("pca", PCA(n_components=n_comp, random_state=config.semilla)))
    pasos.append(("svm", SVC(kernel="rbf", C=config.svm_c, gamma="scale", class_weight="balanced")))
    return Pipeline(pasos)


def _pipeline_knn(config: ConfiguracionExperimento, n_train: int, n_features: int) -> Pipeline:
    pasos: list[tuple[str, object]] = [("escalado", StandardScaler())]
    n_comp = min(config.pca_componentes_max, n_train - 1, n_features)
    if n_features > n_comp and n_comp >= 2:
        pasos.append(("pca", PCA(n_components=n_comp, random_state=config.semilla)))
    vecinos = max(1, min(config.knn_vecinos, n_train))
    pasos.append(("knn", KNeighborsClassifier(n_neighbors=vecinos, weights="distance")))
    return Pipeline(pasos)


def evaluar_clasificador_clasico(
    nombre_clasificador: str,
    matrices: dict[str, np.ndarray],
    metadatos: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: ConfiguracionExperimento,
    carpeta_salida: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    y_bin, y_multi = etiquetas_numericas(metadatos)
    filas_binarias: list[dict[str, object]] = []
    filas_multiclase: list[dict[str, object]] = []
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    for representacion in REPRESENTACIONES:
        x = vectorizar_para_clasicos(matrices[representacion])
        carpeta_repr = carpeta_salida / representacion
        carpeta_repr.mkdir(parents=True, exist_ok=True)
        for fold, (idx_train, idx_test) in enumerate(folds, start=1):
            creador = _pipeline_svm if nombre_clasificador == "SVM" else _pipeline_knn
            modelo_bin = creador(config, len(idx_train), x.shape[1])
            modelo_multi = creador(config, len(idx_train), x.shape[1])
            modelo_bin.fit(x[idx_train], y_bin[idx_train])
            modelo_multi.fit(x[idx_train], y_multi[idx_train])
            pred_bin = modelo_bin.predict(x[idx_test])
            pred_multi = modelo_multi.predict(x[idx_test])

            predicciones = metadatos.iloc[idx_test][["clase", "etiqueta_binaria", "archivo", "ruta"]].copy()
            predicciones["fold"] = fold
            predicciones["pred_binaria"] = np.where(pred_bin == 1, "anomalo", "normal")
            predicciones["pred_multiclase"] = [CLASES[int(i)] for i in pred_multi]
            predicciones.to_csv(carpeta_repr / f"fold_{fold}_predicciones.csv", index=False, encoding="utf-8-sig")
            matriz_confusion_binaria(y_bin[idx_test], pred_bin).to_csv(
                carpeta_repr / f"fold_{fold}_matriz_confusion_binaria.csv",
                encoding="utf-8-sig",
            )
            matriz_multi, metricas_multi = metricas_multiclase(y_multi[idx_test], pred_multi)
            matriz_multi.to_csv(carpeta_repr / f"fold_{fold}_matriz_confusion_multiclase.csv", encoding="utf-8-sig")
            metricas_multi.to_csv(carpeta_repr / f"fold_{fold}_metricas_multiclase.csv", index=False, encoding="utf-8-sig")

            filas_binarias.append(
                {
                    "clasificador": nombre_clasificador,
                    "representacion": representacion,
                    "fold": fold,
                    **metricas_binarias(y_bin[idx_test], pred_bin),
                }
            )
            fila_macro = metricas_multi.loc[metricas_multi["clase"] == "PROMEDIO_MACRO"].iloc[0].to_dict()
            filas_multiclase.append(
                {
                    "clasificador": nombre_clasificador,
                    "representacion": representacion,
                    "fold": fold,
                    **fila_macro,
                }
            )
        print(f"[{nombre_clasificador}] {representacion} completado")
    return filas_binarias, filas_multiclase


def _torch_disponible():
    try:
        import torch
        import torch.nn as nn
        import torch.utils.data as data
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch no esta instalado; no se puede entrenar UjaNet.") from exc
    return torch, nn, data


class _UjaNet:
    """Constructor diferido para evitar importar torch al cargar el modulo."""

    @staticmethod
    def crear(nn, forma_entrada: tuple[int, int], salidas: int):
        class UjaNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bloque = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=5, padding=2),
                    nn.LeakyReLU(0.01),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=5, padding=2),
                    nn.LeakyReLU(0.01),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                )
                with __import__("torch").no_grad():
                    dummy = __import__("torch").zeros(1, 1, *forma_entrada)
                    dimension = int(self.bloque(dummy).shape[1])
                self.clasificador = nn.Sequential(
                    nn.Linear(dimension, 100),
                    nn.LeakyReLU(0.01),
                    nn.Dropout(0.5),
                    nn.Linear(100, 50),
                    nn.LeakyReLU(0.01),
                    nn.Dropout(0.5),
                    nn.Linear(50, salidas),
                )

            def forward(self, x):
                return self.clasificador(self.bloque(x))

        return UjaNet()


def _indices_entrenamiento_validacion(indices: np.ndarray, y: np.ndarray, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    clases, conteos = np.unique(y[indices], return_counts=True)
    if len(indices) < 8 or np.min(conteos) < 2:
        return indices, indices
    return train_test_split(indices, test_size=0.25, random_state=semilla, stratify=y[indices])


def _entrenar_ujanet(
    x: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    config: ConfiguracionExperimento,
    binario: bool,
    ruta_modelo: Path,
) -> np.ndarray:
    torch, nn, data = _torch_disponible()
    torch.manual_seed(config.semilla)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    media = float(x[idx_train].mean())
    std = float(x[idx_train].std() + 1e-6)
    x_norm = ((x - media) / std).astype(np.float32)
    tensor_x = torch.tensor(x_norm[:, None, :, :], dtype=torch.float32)
    if binario:
        tensor_y = torch.tensor(y.astype(np.float32), dtype=torch.float32).view(-1, 1)
        salidas = 1
        criterio = nn.BCEWithLogitsLoss()
    else:
        tensor_y = torch.tensor(y.astype(np.int64), dtype=torch.long)
        salidas = len(CLASES)
        criterio = nn.CrossEntropyLoss()

    idx_ent, idx_val = _indices_entrenamiento_validacion(idx_train, y, config.semilla)
    modelo = _UjaNet.crear(nn, forma_entrada=tuple(x.shape[1:]), salidas=salidas).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=config.ujanet_lr)
    conjunto = data.TensorDataset(tensor_x[idx_ent], tensor_y[idx_ent])
    cargador = data.DataLoader(conjunto, batch_size=min(config.ujanet_lote, max(1, len(idx_ent))), shuffle=True)

    mejor_estado = None
    mejor_val = float("inf")
    sin_mejora = 0
    historial = []
    for epoca in range(1, config.ujanet_epocas + 1):
        modelo.train()
        perdidas = []
        for lote_x, lote_y in cargador:
            lote_x = lote_x.to(dispositivo)
            lote_y = lote_y.to(dispositivo)
            optimizador.zero_grad()
            salida = modelo(lote_x)
            perdida = criterio(salida, lote_y)
            perdida.backward()
            optimizador.step()
            perdidas.append(float(perdida.item()))

        modelo.eval()
        with torch.no_grad():
            val_x = tensor_x[idx_val].to(dispositivo)
            val_y = tensor_y[idx_val].to(dispositivo)
            val_loss = float(criterio(modelo(val_x), val_y).item())
        historial.append({"epoca": epoca, "loss_train": float(np.mean(perdidas)), "loss_val": val_loss})
        if val_loss < mejor_val:
            mejor_val = val_loss
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= config.ujanet_paciencia:
                break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(historial).to_csv(ruta_modelo.with_suffix(".historial.csv"), index=False, encoding="utf-8-sig")
    torch.save(
        {
            "estado": modelo.state_dict(),
            "media": media,
            "std": std,
            "forma_entrada": list(x.shape[1:]),
            "binario": binario,
        },
        ruta_modelo,
    )

    modelo.eval()
    with torch.no_grad():
        logits = modelo(tensor_x[idx_test].to(dispositivo)).cpu()
        if binario:
            pred = (torch.sigmoid(logits).numpy().ravel() >= 0.5).astype(int)
        else:
            pred = torch.argmax(logits, dim=1).numpy().astype(int)
    return pred


def evaluar_ujanet(
    matrices: dict[str, np.ndarray],
    metadatos: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: ConfiguracionExperimento,
    carpeta_salida: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    y_bin, y_multi = etiquetas_numericas(metadatos)
    filas_binarias: list[dict[str, object]] = []
    filas_multiclase: list[dict[str, object]] = []
    for representacion in REPRESENTACIONES:
        x = matrices[representacion].astype(np.float32)
        carpeta_repr_bin = carpeta_salida / "UjaNet" / "binaria" / representacion
        carpeta_repr_multi = carpeta_salida / "UjaNet" / "multiclase" / representacion
        carpeta_repr_bin.mkdir(parents=True, exist_ok=True)
        carpeta_repr_multi.mkdir(parents=True, exist_ok=True)
        for fold, (idx_train, idx_test) in enumerate(folds, start=1):
            pred_bin = _entrenar_ujanet(
                x,
                y_bin,
                idx_train,
                idx_test,
                config,
                binario=True,
                ruta_modelo=carpeta_repr_bin / f"fold_{fold}_modelo.pt",
            )
            pred_multi = _entrenar_ujanet(
                x,
                y_multi,
                idx_train,
                idx_test,
                config,
                binario=False,
                ruta_modelo=carpeta_repr_multi / f"fold_{fold}_modelo.pt",
            )
            predicciones = metadatos.iloc[idx_test][["clase", "etiqueta_binaria", "archivo", "ruta"]].copy()
            predicciones["fold"] = fold
            predicciones["pred_binaria"] = np.where(pred_bin == 1, "anomalo", "normal")
            predicciones["pred_multiclase"] = [CLASES[int(i)] for i in pred_multi]
            predicciones.to_csv(carpeta_repr_bin / f"fold_{fold}_predicciones.csv", index=False, encoding="utf-8-sig")
            predicciones.to_csv(carpeta_repr_multi / f"fold_{fold}_predicciones.csv", index=False, encoding="utf-8-sig")
            matriz_confusion_binaria(y_bin[idx_test], pred_bin).to_csv(
                carpeta_repr_bin / f"fold_{fold}_matriz_confusion_binaria.csv",
                encoding="utf-8-sig",
            )
            matriz_multi, metricas_multi = metricas_multiclase(y_multi[idx_test], pred_multi)
            matriz_multi.to_csv(carpeta_repr_multi / f"fold_{fold}_matriz_confusion_multiclase.csv", encoding="utf-8-sig")
            metricas_multi.to_csv(carpeta_repr_multi / f"fold_{fold}_metricas_multiclase.csv", index=False, encoding="utf-8-sig")
            filas_binarias.append(
                {
                    "clasificador": "UjaNet",
                    "representacion": representacion,
                    "fold": fold,
                    **metricas_binarias(y_bin[idx_test], pred_bin),
                }
            )
            fila_macro = metricas_multi.loc[metricas_multi["clase"] == "PROMEDIO_MACRO"].iloc[0].to_dict()
            filas_multiclase.append(
                {
                    "clasificador": "UjaNet",
                    "representacion": representacion,
                    "fold": fold,
                    **fila_macro,
                }
            )
        print(f"[UjaNet] {representacion} completado")
    return filas_binarias, filas_multiclase


def resumir_metricas(filas_binarias: list[dict[str, object]], filas_multiclase: list[dict[str, object]], carpeta: Path) -> ResultadoClasificacion:
    carpeta.mkdir(parents=True, exist_ok=True)
    binaria = pd.DataFrame(filas_binarias)
    multiclase = pd.DataFrame(filas_multiclase)
    binaria.to_csv(carpeta / "metricas_binarias_por_fold.csv", index=False, encoding="utf-8-sig")
    multiclase.to_csv(carpeta / "metricas_multiclase_por_fold.csv", index=False, encoding="utf-8-sig")
    resumen_bin = (
        binaria.groupby(["clasificador", "representacion"], as_index=False)
        .agg({col: ["mean", "std"] for col in ["Accuracy", "Sensitivity", "Specificity", "Precision", "Score"]})
    )
    resumen_bin.columns = ["_".join([c for c in col if c]).strip("_") for col in resumen_bin.columns.to_flat_index()]
    resumen_multi = (
        multiclase.groupby(["clasificador", "representacion"], as_index=False)
        .agg({col: ["mean", "std"] for col in ["Accuracy", "Sensitivity", "Specificity", "Precision", "Score"]})
    )
    resumen_multi.columns = ["_".join([c for c in col if c]).strip("_") for col in resumen_multi.columns.to_flat_index()]
    resumen_bin.to_csv(carpeta / "resumen_metricas_binarias.csv", index=False, encoding="utf-8-sig")
    resumen_multi.to_csv(carpeta / "resumen_metricas_multiclase.csv", index=False, encoding="utf-8-sig")
    return ResultadoClasificacion(resumen_binario=resumen_bin, resumen_multiclase=resumen_multi)


def guardar_resumen_ejecucion(carpeta: Path, datos: dict[str, object]) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "resumen_ejecucion.json").write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
