from __future__ import annotations

import json
import math
import shutil
import sys
import textwrap
import time
import warnings
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import fitz
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd


warnings.filterwarnings("ignore", category=UserWarning)

FINAL = Path(__file__).resolve().parent
ROOT = FINAL.parent
SRC = ROOT / "src"
RESULTADOS = FINAL / "RESULTADOS"
MEDIA = RESULTADOS / "fotos datos y graficas"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfg_deep_onmf.audio import (  # noqa: E402
    construir_matriz_audio,
    construir_matriz_clase,
    descubrir_audios,
    dividir_en_tramas,
    espectrograma_magnitud,
    leer_wav_normalizado,
)
from tfg_deep_onmf.configuracion import Configuracion  # noqa: E402
from tfg_deep_onmf.estadistica import (  # noqa: E402
    caracteristicas_por_audio,
    distancias_figura_7,
    resumen_auditoria,
    tabla_2_desde_w,
)
from tfg_deep_onmf.graficos import figura_5_sbv, figura_7_distancias, tabla_2_imagen  # noqa: E402
from tfg_deep_onmf.onmf import proyectar_sobre_w  # noqa: E402


CLASES = ("N", "AS", "MR", "MS", "MVP")
COLORES = {
    "N": "#1b9e77",
    "AS": "#d95f02",
    "MR": "#7570b3",
    "MS": "#e7298a",
    "MVP": "#66a61e",
}
SEED = 42
EPS = 1e-12


@dataclass
class CapaLocal:
    clase: str
    metodo: str
    capa: int
    rango: int
    forma_entrada: tuple[int, int]
    forma_w: tuple[int, int]
    forma_h: tuple[int, int]
    error_relativo: float
    ortogonalidad_media: float
    segundos: float


@dataclass
class ResultadoLocal:
    w_final: np.ndarray
    h_final: np.ndarray
    capas: list[CapaLocal]
    error_relativo_final: float


def preparar_carpetas() -> dict[str, Path]:
    if ROOT.name != "Programacion objetivo":
        raise RuntimeError(f"Raiz inesperada: {ROOT}")
    if RESULTADOS.exists():
        shutil.rmtree(RESULTADOS)
    MEDIA.mkdir(parents=True, exist_ok=True)
    carpetas = {
        "codigo": MEDIA / "01_codigo_generacion_final",
        "articulo": MEDIA / "02_implementacion_fiel_articulo",
        "comparativa": MEDIA / "03_comparativa_principal",
        "inicializaciones": MEDIA / "04_deep_onmf_mejorado_inicializaciones",
        "convergencia": MEDIA / "05_convergencia_0_300",
        "verificacion": MEDIA / "06_verificacion",
    }
    for carpeta in carpetas.values():
        carpeta.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), carpetas["codigo"] / "01_generar_entrega_final.py")
    return carpetas


def carpetas_existentes() -> dict[str, Path]:
    carpetas = {
        "codigo": MEDIA / "01_codigo_generacion_final",
        "articulo": MEDIA / "02_implementacion_fiel_articulo",
        "comparativa": MEDIA / "03_comparativa_principal",
        "inicializaciones": MEDIA / "04_deep_onmf_mejorado_inicializaciones",
        "convergencia": MEDIA / "05_convergencia_0_300",
        "verificacion": MEDIA / "06_verificacion",
    }
    for carpeta in carpetas.values():
        carpeta.mkdir(parents=True, exist_ok=True)
    return carpetas


def guardar_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def normalizar_columnas_w(w: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    escala = np.maximum(np.linalg.norm(w, axis=0), EPS)
    return w / escala[None, :], h * escala[:, None]


def ortogonalidad_media(h: np.ndarray) -> float:
    normas = np.maximum(np.linalg.norm(h, axis=1, keepdims=True), EPS)
    h_norm = h / normas
    gramo = h_norm @ h_norm.T
    mascara = ~np.eye(gramo.shape[0], dtype=bool)
    return float(np.mean(np.abs(gramo[mascara])))


def error_relativo(x: np.ndarray, w: np.ndarray, h: np.ndarray) -> float:
    return float(np.linalg.norm(x - w @ h, ord="fro") / max(np.linalg.norm(x, ord="fro"), EPS))


def nndsvd_inicializar(x: np.ndarray, rango: int, variante: str, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.maximum(x.astype(np.float64, copy=False), EPS)
    media = float(np.mean(x))
    u, s, vt = randomized_svd(x, n_components=rango, random_state=semilla)
    w = np.zeros((x.shape[0], rango), dtype=np.float64)
    h = np.zeros((rango, x.shape[1]), dtype=np.float64)

    w[:, 0] = math.sqrt(s[0]) * np.abs(u[:, 0])
    h[0, :] = math.sqrt(s[0]) * np.abs(vt[0, :])

    for j in range(1, rango):
        uj = u[:, j]
        vj = vt[j, :]
        uj_pos = np.maximum(uj, 0)
        uj_neg = np.maximum(-uj, 0)
        vj_pos = np.maximum(vj, 0)
        vj_neg = np.maximum(-vj, 0)

        norm_pos = np.linalg.norm(uj_pos) * np.linalg.norm(vj_pos)
        norm_neg = np.linalg.norm(uj_neg) * np.linalg.norm(vj_neg)
        if norm_pos >= norm_neg:
            uu = uj_pos / max(np.linalg.norm(uj_pos), EPS)
            vv = vj_pos / max(np.linalg.norm(vj_pos), EPS)
            sigma = norm_pos
        else:
            uu = uj_neg / max(np.linalg.norm(uj_neg), EPS)
            vv = vj_neg / max(np.linalg.norm(vj_neg), EPS)
            sigma = norm_neg
        escala = math.sqrt(s[j] * sigma)
        w[:, j] = escala * uu
        h[j, :] = escala * vv

    if variante == "nndsvda":
        w[w <= EPS] = media
        h[h <= EPS] = media
    elif variante == "nndsvdar":
        rng = np.random.default_rng(semilla)
        w[w <= EPS] = media * rng.random(np.count_nonzero(w <= EPS)) / 100.0
        h[h <= EPS] = media * rng.random(np.count_nonzero(h <= EPS)) / 100.0
    elif variante != "nndsvd":
        raise ValueError(f"Variante NNDSVD no soportada: {variante}")

    w = np.maximum(w, EPS)
    h = np.maximum(h, EPS)
    return normalizar_columnas_w(w, h)


def inicializar_factorizacion(
    x: np.ndarray,
    rango: int,
    metodo: str,
    semilla: int,
) -> tuple[np.ndarray, np.ndarray]:
    if metodo == "aleatoria":
        rng = np.random.default_rng(semilla)
        w = rng.random((x.shape[0], rango)) + EPS
        h = rng.random((rango, x.shape[1])) + EPS
        return normalizar_columnas_w(w, h)
    return nndsvd_inicializar(x, rango, metodo, semilla)


def factorizar_onmf_local(
    matriz: np.ndarray,
    rango: int,
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
    metodo_init: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.maximum(matriz, EPS).astype(np.float64, copy=False)
    w, h = inicializar_factorizacion(x, rango, metodo_init, semilla)
    for _ in range(iteraciones):
        w *= (x @ h.T) / (w @ (h @ h.T) + EPS)
        w = np.maximum(w, EPS)
        h *= (w.T @ x + penalizacion_ortogonal * h) / (
            (w.T @ w) @ h + penalizacion_ortogonal * ((h @ h.T) @ h) + EPS
        )
        h = np.maximum(h, EPS)
        w, h = normalizar_columnas_w(w, h)
    return w, h, error_relativo(x, w, h), ortogonalidad_media(h)


def deep_onmf_local(
    matriz: np.ndarray,
    clase: str,
    metodo: str,
    rangos: tuple[int, int, int],
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
) -> ResultadoLocal:
    entrada = np.maximum(matriz, EPS)
    matrices_w: list[np.ndarray] = []
    capas: list[CapaLocal] = []
    for indice, rango in enumerate(rangos, start=1):
        inicio = time.perf_counter()
        w, h, error, ort = factorizar_onmf_local(
            entrada,
            rango=rango,
            iteraciones=iteraciones,
            penalizacion_ortogonal=penalizacion_ortogonal,
            semilla=semilla + indice * 1000,
            metodo_init=metodo,
        )
        segundos = time.perf_counter() - inicio
        matrices_w.append(w)
        capas.append(
            CapaLocal(
                clase=clase,
                metodo=metodo,
                capa=indice,
                rango=rango,
                forma_entrada=entrada.shape,
                forma_w=w.shape,
                forma_h=h.shape,
                error_relativo=error,
                ortogonalidad_media=ort,
                segundos=segundos,
            )
        )
        entrada = h

    w_final = matrices_w[0] @ matrices_w[1] @ matrices_w[2]
    normas = np.maximum(np.linalg.norm(w_final, axis=0), EPS)
    w_final = w_final / normas[None, :]
    h_final = entrada * normas[:, None]
    return ResultadoLocal(
        w_final=w_final,
        h_final=h_final,
        capas=capas,
        error_relativo_final=error_relativo(np.maximum(matriz, EPS), w_final, h_final),
    )


def descubrir_y_preparar(config: Configuracion, carpetas: dict[str, Path]):
    registros = descubrir_audios(config.carpeta_base_datos, config.clases)
    datos_por_clase = {}
    for clase in config.clases:
        datos_por_clase[clase] = construir_matriz_clase(clase, registros, config)

    auditoria = resumen_auditoria(registros, datos_por_clase, config.clases)
    guardar_csv(auditoria, carpetas["articulo"] / "auditoria_articulo_fiel.csv")
    parametros = config.como_diccionario()
    parametros["nota"] = "Reproduccion fiel: audios menores de 2 s descartados, no rellenados."
    (carpetas["articulo"] / "parametros_articulo_fiel.json").write_text(
        json.dumps(parametros, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return registros, datos_por_clase, auditoria


def tabla_capas(resultados: dict[str, ResultadoLocal]) -> pd.DataFrame:
    filas = []
    for resultado in resultados.values():
        for capa in resultado.capas:
            filas.append(
                {
                    "metodo_init": capa.metodo,
                    "clase": capa.clase,
                    "capa": capa.capa,
                    "rango": capa.rango,
                    "entrada": f"{capa.forma_entrada[0]}x{capa.forma_entrada[1]}",
                    "W": f"{capa.forma_w[0]}x{capa.forma_w[1]}",
                    "H": f"{capa.forma_h[0]}x{capa.forma_h[1]}",
                    "error_relativo": capa.error_relativo,
                    "ortogonalidad_media": capa.ortogonalidad_media,
                    "segundos": capa.segundos,
                }
            )
    return pd.DataFrame(filas)


def entrenar_deep_onmf(
    datos_por_clase,
    config: Configuracion,
    metodo: str,
    carpeta: Path,
) -> dict[str, ResultadoLocal]:
    resultados: dict[str, ResultadoLocal] = {}
    for pos, clase in enumerate(config.clases):
        print(f"[deep-onmf] {metodo} clase {clase}")
        resultados[clase] = deep_onmf_local(
            datos_por_clase[clase].matriz,
            clase=clase,
            metodo=metodo,
            rangos=config.rangos_onmf,
            iteraciones=config.iteraciones_onmf,
            penalizacion_ortogonal=config.penalizacion_ortogonal,
            semilla=config.semilla + pos * 17,
        )
    guardar_csv(tabla_capas(resultados), carpeta / f"capas_deep_onmf_{metodo}.csv")
    np.savez_compressed(
        carpeta / f"matrices_w_finales_{metodo}.npz",
        **{clase: resultados[clase].w_final for clase in config.clases},
    )
    return resultados


def caracteristicas_normales_por_audio(datos_por_clase, resultados, clases) -> pd.DataFrame:
    h_por_clase = {clase: resultados[clase].h_final for clase in clases}
    return caracteristicas_por_audio(datos_por_clase, h_por_clase)


def caracteristicas_mejoradas_por_audio(registros, w_por_clase, config, nombre: str) -> pd.DataFrame:
    filas = []
    for i, registro in enumerate(registros, start=1):
        if i % 100 == 0:
            print(f"[afinidades] {nombre}: {i}/{len(registros)}")
        matriz = construir_matriz_audio(registro, config)
        errores = []
        fila: dict[str, object] = {
            "clase": registro.clase,
            "archivo": registro.ruta.name,
            "ruta": str(registro.ruta),
            "duracion_s": registro.duracion_s,
        }
        for clase_modelo in config.clases:
            _, err = proyectar_sobre_w(matriz, w_por_clase[clase_modelo], iteraciones=80)
            errores.append(err)
            fila[f"error_vs_{clase_modelo}"] = err
        errores_arr = np.asarray(errores, dtype=np.float64)
        afinidades = np.exp(-8.0 * errores_arr)
        afinidades = afinidades / np.maximum(afinidades.sum(), EPS)
        for clase_modelo, valor in zip(config.clases, afinidades):
            fila[f"afinidad_{clase_modelo}"] = float(valor)
        filas.append(fila)
    return pd.DataFrame(filas)


def leer_senal(registro) -> np.ndarray:
    senal, fs = leer_wav_normalizado(registro.ruta)
    if fs != 8000:
        raise ValueError(f"Frecuencia inesperada {fs} en {registro.ruta}")
    return senal


def espectrograma_audio_fijo(registro, config: Configuracion) -> np.ndarray:
    senal = leer_senal(registro)
    tramas = dividir_en_tramas(senal, config)
    matrices = [np.log1p(espectrograma_magnitud(trama, config)) for trama in tramas]
    return np.mean(np.stack(matrices, axis=0), axis=0)


def rasgos_stft(registros, config: Configuracion) -> pd.DataFrame:
    filas = []
    for registro in registros:
        spec = espectrograma_audio_fijo(registro, config)
        vector = np.concatenate([spec.mean(axis=1), spec.std(axis=1)])
        fila = {"clase": registro.clase, "archivo": registro.ruta.name}
        for i, valor in enumerate(vector, start=1):
            fila[f"stft_{i}"] = float(valor)
        filas.append(fila)
    return pd.DataFrame(filas)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def banco_mel_articulo(config: Configuracion, bandas: int = 40) -> np.ndarray:
    puntos_mel = np.linspace(hz_to_mel(0), hz_to_mel(config.frecuencia_esperada_hz / 2), bandas + 2)
    puntos_hz = mel_to_hz(puntos_mel)
    bins = np.floor((config.puntos_fft + 1) * puntos_hz / config.frecuencia_esperada_hz).astype(int)
    banco = np.zeros((bandas, config.bins_frecuencia), dtype=np.float64)
    for banda in range(1, bandas + 1):
        izq, centro, der = bins[banda - 1], bins[banda], bins[banda + 1]
        centro = max(centro, izq + 1)
        der = max(der, centro + 1)
        for k in range(izq, centro):
            if 0 <= k < banco.shape[1]:
                banco[banda - 1, k] = (k - izq) / max(1, centro - izq)
        for k in range(centro, der):
            if 0 <= k < banco.shape[1]:
                banco[banda - 1, k] = (der - k) / max(1, der - centro)
    return banco


def rasgos_mfcc(registros, config: Configuracion) -> pd.DataFrame:
    banco = banco_mel_articulo(config, bandas=40)
    filas = []
    for registro in registros:
        spec = espectrograma_audio_fijo(registro, config)
        mel = np.maximum(banco @ spec, EPS)
        log_mel = np.log(mel)
        coef = dct(log_mel, type=2, axis=0, norm="ortho")[:13]
        vector = coef.mean(axis=1)
        fila = {"clase": registro.clase, "archivo": registro.ruta.name}
        for i, valor in enumerate(vector, start=1):
            fila[f"mfcc_{i}"] = float(valor)
        filas.append(fila)
    return pd.DataFrame(filas)


def rasgos_dwt(registros, config: Configuracion) -> pd.DataFrame:
    filas = []
    for registro in registros:
        senal = leer_senal(registro)
        tramas = dividir_en_tramas(senal, config)
        vectores = []
        for trama in tramas:
            nivel = min(5, pywt.dwt_max_level(len(trama), pywt.Wavelet("coif5").dec_len))
            coeficientes = pywt.wavedec(trama, wavelet="coif5", level=nivel, mode="symmetric")
            rasgos = []
            for bloque in coeficientes:
                valores = np.asarray(bloque, dtype=np.float64)
                rasgos.extend([np.log1p(np.mean(valores**2)), np.mean(np.abs(valores)), np.std(valores)])
            vectores.append(np.asarray(rasgos, dtype=np.float64))
        vector = np.mean(np.stack(vectores, axis=0), axis=0)
        fila = {"clase": registro.clase, "archivo": registro.ruta.name}
        for i, valor in enumerate(vector, start=1):
            fila[f"dwt_{i}"] = float(valor)
        filas.append(fila)
    return pd.DataFrame(filas)


class CNNArticulo(nn.Module):
    def __init__(self, clases: int = 5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 6))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.clasificador = nn.Linear(64, clases)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clasificador(self.features(x))


def rasgos_cnn_articulo(registros, config: Configuracion, carpeta: Path) -> pd.DataFrame:
    print("[cnn] construyendo espectrogramas fijos")
    specs = np.stack([espectrograma_audio_fijo(r, config) for r in registros], axis=0).astype(np.float32)
    y = np.array([CLASES.index(r.clase) for r in registros], dtype=np.int64)
    media = specs.mean()
    std = specs.std() + 1e-6
    x = ((specs - media) / std)[:, None, :, :]

    train_idx, val_idx = train_test_split(
        np.arange(len(registros)),
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )
    torch.manual_seed(SEED)
    modelo = CNNArticulo(clases=len(CLASES))
    optim = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    perdida = nn.CrossEntropyLoss()
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)

    historial = []
    batch = 32
    for epoca in range(1, 13):
        modelo.train()
        orden = torch.tensor(np.random.default_rng(SEED + epoca).permutation(train_idx), dtype=torch.long)
        perdidas = []
        for inicio in range(0, len(orden), batch):
            idx = orden[inicio : inicio + batch]
            optim.zero_grad()
            logits = modelo(tensor_x[idx])
            loss = perdida(logits, tensor_y[idx])
            loss.backward()
            optim.step()
            perdidas.append(float(loss.item()))
        modelo.eval()
        with torch.no_grad():
            pred_train = modelo(tensor_x[train_idx]).argmax(dim=1)
            pred_val = modelo(tensor_x[val_idx]).argmax(dim=1)
        acc_train = float((pred_train == tensor_y[train_idx]).float().mean().item())
        acc_val = float((pred_val == tensor_y[val_idx]).float().mean().item())
        historial.append({"epoca": epoca, "loss": float(np.mean(perdidas)), "acc_train": acc_train, "acc_val": acc_val})
        print(f"[cnn] epoca {epoca:02d} loss={np.mean(perdidas):.4f} val={acc_val:.3f}")

    guardar_csv(pd.DataFrame(historial), carpeta / "entrenamiento_cnn_articulo.csv")
    torch.save(
        {
            "estado": modelo.state_dict(),
            "arquitectura": "Conv2d(1,32,3x3)+ReLU; Conv2d(32,64,3x3)+ReLU; MaxPool2d(1,6); GAP; Linear local",
            "media": float(media),
            "std": float(std),
        },
        carpeta / "cnn_articulo_entrenada.pt",
    )

    modelo.eval()
    with torch.no_grad():
        features = modelo.features(tensor_x).numpy()
    filas = []
    for registro, vector in zip(registros, features):
        fila = {"clase": registro.clase, "archivo": registro.ruta.name}
        for i, valor in enumerate(vector, start=1):
            fila[f"cnn_{i}"] = float(valor)
        filas.append(fila)
    return pd.DataFrame(filas)


def columnas_rasgos(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"clase", "archivo", "ruta", "duracion_s"} and not c.startswith("error_")]


def evaluar_metodo(nombre: str, df: pd.DataFrame, carpeta: Path) -> tuple[pd.Series, pd.DataFrame]:
    cols = columnas_rasgos(df)
    x_original = df[cols].to_numpy(dtype=np.float64)
    labels = df["clase"].to_numpy()
    x = StandardScaler().fit_transform(x_original)
    if x.shape[1] > 50:
        x_tsne = PCA(n_components=50, random_state=SEED).fit_transform(x)
        dim_tsne = 50
    else:
        x_tsne = x
        dim_tsne = x.shape[1]

    perplexity = min(30, max(5, (len(df) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=SEED,
        max_iter=1000,
    )
    coords = tsne.fit_transform(x_tsne)
    coords_df = df[["clase", "archivo"]].copy()
    coords_df["tsne_1"] = coords[:, 0]
    coords_df["tsne_2"] = coords[:, 1]
    guardar_csv(coords_df, carpeta / f"coordenadas_tsne_{slug(nombre)}.csv")

    metricas = pd.Series(
        {
            "metodo": nombre,
            "muestras": len(df),
            "rasgos_originales": len(cols),
            "rasgos_entrada_tsne": dim_tsne,
            "perplexity": perplexity,
            "silhouette_features": silhouette_score(x, labels),
            "davies_bouldin_features": davies_bouldin_score(x, labels),
            "silhouette_tsne": silhouette_score(coords, labels),
            "davies_bouldin_tsne": davies_bouldin_score(coords, labels),
        }
    )
    return metricas, coords_df


def slug(nombre: str) -> str:
    return (
        nombre.lower()
        .replace("+", "mas")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def scatter_en_eje(eje, coords: pd.DataFrame, titulo: str, metricas: pd.Series) -> None:
    for clase in CLASES:
        mask = coords["clase"] == clase
        eje.scatter(
            coords.loc[mask, "tsne_1"],
            coords.loc[mask, "tsne_2"],
            s=16,
            alpha=0.78,
            color=COLORES[clase],
            label=clase,
            edgecolors="none",
        )
    eje.set_title(
        f"{titulo}\nSil={metricas['silhouette_tsne']:.3f} | DB={metricas['davies_bouldin_tsne']:.3f}",
        fontsize=10,
    )
    eje.set_xlabel("t-SNE 1")
    eje.set_ylabel("t-SNE 2")
    eje.grid(True, alpha=0.18)


def figura_comparativa(metricas: pd.DataFrame, coords_por_metodo: dict[str, pd.DataFrame], orden: list[str], ruta: Path) -> None:
    filas = math.ceil(len(orden) / 3)
    fig, ejes = plt.subplots(filas, 3, figsize=(16, 5.2 * filas))
    ejes = np.asarray(ejes).reshape(-1)
    for eje, metodo in zip(ejes, orden):
        fila_metricas = metricas.loc[metricas["metodo"] == metodo].iloc[0]
        scatter_en_eje(eje, coords_por_metodo[metodo], metodo, fila_metricas)
    for eje in ejes[len(orden) :]:
        eje.axis("off")
    handles, labels = ejes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CLASES), title="Clase")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.suptitle("Comparativa t-SNE generada de nuevo desde codigo", fontsize=16)
    fig.savefig(ruta, dpi=220)
    plt.close(fig)


def figura_metricas(metricas: pd.DataFrame, ruta: Path, titulo: str) -> None:
    fig, ejes = plt.subplots(2, 2, figsize=(15, 9))
    columnas = [
        ("silhouette_features", "Silhouette rasgos (mayor es mejor)"),
        ("davies_bouldin_features", "Davies-Bouldin rasgos (menor es mejor)"),
        ("silhouette_tsne", "Silhouette t-SNE (mayor es mejor)"),
        ("davies_bouldin_tsne", "Davies-Bouldin t-SNE (menor es mejor)"),
    ]
    colores = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6"]
    for eje, (col, label) in zip(ejes.reshape(-1), columnas):
        eje.bar(metricas["metodo"], metricas[col], color=colores[: len(metricas)])
        eje.set_title(label)
        eje.tick_params(axis="x", rotation=35)
        eje.grid(True, axis="y", alpha=0.25)
    fig.suptitle(titulo, fontsize=15)
    fig.tight_layout()
    fig.savefig(ruta, dpi=220)
    plt.close(fig)


def generar_comparativas(registros, datos_por_clase, resultados_random, config, carpetas):
    registros_usados = [r for clase in config.clases for r in datos_por_clase[clase].audios_usados]
    print(f"[datos] audios usados tras descartar menores de 2 s: {len(registros_usados)}")

    w_random = {c: resultados_random[c].w_final for c in config.clases}
    normal = caracteristicas_normales_por_audio(datos_por_clase, resultados_random, config.clases)
    normal = normal[["clase", "archivo"] + [f"SBV_{i}" for i in range(1, 8)]]
    guardar_csv(normal, carpetas["comparativa"] / "rasgos_deep_onmf_normal.csv")

    mejorado = caracteristicas_mejoradas_por_audio(registros_usados, w_random, config, "Deep-ONMF mejorado")
    guardar_csv(mejorado, carpetas["comparativa"] / "rasgos_deep_onmf_mejorado.csv")

    stft = rasgos_stft(registros_usados, config)
    mfcc = rasgos_mfcc(registros_usados, config)
    dwt = rasgos_dwt(registros_usados, config)
    cnn = rasgos_cnn_articulo(registros_usados, config, carpetas["comparativa"])
    for nombre, df in {"STFT": stft, "MFCC": mfcc, "DWT": dwt, "CNN": cnn}.items():
        guardar_csv(df, carpetas["comparativa"] / f"rasgos_{slug(nombre)}.csv")

    metodos = {
        "CNN": cnn,
        "DWT": dwt,
        "MFCC": mfcc,
        "STFT": stft,
        "Deep-ONMF normal": normal,
        "Deep-ONMF mejorado": mejorado,
    }
    filas_metricas = []
    coords = {}
    for nombre, df in metodos.items():
        print(f"[metricas] {nombre}")
        fila, coord = evaluar_metodo(nombre, df, carpetas["comparativa"])
        filas_metricas.append(fila)
        coords[nombre] = coord
    metricas = pd.DataFrame(filas_metricas)
    guardar_csv(metricas, carpetas["comparativa"] / "metricas_comparativa_principal.csv")
    figura_comparativa(
        metricas,
        coords,
        ["CNN", "DWT", "MFCC", "STFT", "Deep-ONMF normal", "Deep-ONMF mejorado"],
        carpetas["comparativa"] / "comparativa_principal_tsne.png",
    )
    figura_metricas(metricas, carpetas["comparativa"] / "comparativa_principal_metricas.png", "Metricas de comparativa principal")
    return registros_usados, normal, mejorado, metricas, coords


def generar_figuras_articulo(datos_por_clase, resultados_random, normal_df, config, carpetas):
    w_por_clase = {c: resultados_random[c].w_final for c in config.clases}
    tabla_2 = tabla_2_desde_w(w_por_clase, config.clases)
    guardar_csv(tabla_2, carpetas["articulo"] / "tabla_2_sbv_articulo_fiel.csv")
    figura_5_sbv(w_por_clase, config.clases, config.frecuencia_esperada_hz, carpetas["articulo"] / "figura_5_sbv_por_clase.png")
    tabla_2_imagen(tabla_2, carpetas["articulo"] / "tabla_2_sbv_articulo_fiel.png")
    distancias = distancias_figura_7(normal_df, config.clases)
    for nombre, tabla in distancias.items():
        guardar_csv(tabla, carpetas["articulo"] / f"{nombre}.csv")
    figura_7_distancias(distancias, carpetas["articulo"] / "figura_7_distancias_sbv.png")


def generar_inicializaciones(registros_usados, datos_por_clase, normal, mejorado, metricas_principal, config, carpetas):
    variantes = {
        "Deep-ONMF normal": normal,
        "Deep-ONMF mejorado": mejorado,
    }
    for metodo_init, etiqueta in [
        ("nndsvd", "Deep-ONMF mejorado + NNDSVD"),
        ("nndsvda", "Deep-ONMF mejorado + NNDSVDa"),
        ("nndsvdar", "Deep-ONMF mejorado + NNDSVDar"),
    ]:
        resultados = entrenar_deep_onmf(datos_por_clase, config, metodo_init, carpetas["inicializaciones"])
        w = {c: resultados[c].w_final for c in config.clases}
        df = caracteristicas_mejoradas_por_audio(registros_usados, w, config, etiqueta)
        guardar_csv(df, carpetas["inicializaciones"] / f"rasgos_{slug(etiqueta)}.csv")
        variantes[etiqueta] = df

    filas = []
    coords = {}
    for nombre, df in variantes.items():
        fila, coord = evaluar_metodo(nombre, df, carpetas["inicializaciones"])
        filas.append(fila)
        coords[nombre] = coord
    metricas = pd.DataFrame(filas)
    guardar_csv(metricas, carpetas["inicializaciones"] / "metricas_deep_onmf_variantes.csv")
    orden = [
        "Deep-ONMF normal",
        "Deep-ONMF mejorado",
        "Deep-ONMF mejorado + NNDSVD",
        "Deep-ONMF mejorado + NNDSVDa",
        "Deep-ONMF mejorado + NNDSVDar",
    ]
    figura_comparativa(metricas, coords, orden, carpetas["inicializaciones"] / "comparativa_variantes_deep_onmf_tsne.png")
    figura_metricas(metricas, carpetas["inicializaciones"] / "comparativa_variantes_deep_onmf_metricas.png", "Deep-ONMF normal, mejorado e inicializaciones")
    return metricas


def factorizar_con_historial(
    matriz: np.ndarray,
    rango: int,
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    x = np.maximum(matriz, EPS)
    w, h = inicializar_factorizacion(x, rango, "aleatoria", semilla)
    historial = [error_relativo(x, w, h)]
    for _ in range(iteraciones):
        w *= (x @ h.T) / (w @ (h @ h.T) + EPS)
        w = np.maximum(w, EPS)
        h *= (w.T @ x + penalizacion_ortogonal * h) / (
            (w.T @ w) @ h + penalizacion_ortogonal * ((h @ h.T) @ h) + EPS
        )
        h = np.maximum(h, EPS)
        w, h = normalizar_columnas_w(w, h)
        historial.append(error_relativo(x, w, h))
    return w, h, historial


def estudio_convergencia(datos_por_clase, config, carpeta: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filas = []
    for clase in config.clases:
        entrada = datos_por_clase[clase].matriz
        for capa, rango in enumerate(config.rangos_onmf, start=1):
            print(f"[convergencia] clase {clase} capa {capa}")
            w, h, historial = factorizar_con_historial(
                entrada,
                rango=rango,
                iteraciones=300,
                penalizacion_ortogonal=config.penalizacion_ortogonal,
                semilla=config.semilla + capa * 1000,
            )
            for iteracion, err in enumerate(historial):
                filas.append({"clase": clase, "capa": capa, "rango": rango, "iteracion": iteracion, "error_relativo": err})
            entrada = h
    detalle = pd.DataFrame(filas)
    resumen = detalle.groupby(["capa", "rango", "iteracion"], as_index=False)["error_relativo"].mean()
    guardar_csv(detalle, carpeta / "convergencia_0_300_detalle.csv")
    guardar_csv(resumen, carpeta / "convergencia_0_300_resumen_medio.csv")

    filas_comp = []
    for (capa, rango), grupo in resumen.groupby(["capa", "rango"]):
        serie = grupo.sort_values("iteracion")["error_relativo"].to_numpy()
        e0, e120, e300 = serie[0], serie[120], serie[300]
        total = max(e0 - e300, EPS)
        filas_comp.append(
            {
                "capa": capa,
                "rango": rango,
                "error_iter_0": e0,
                "error_iter_120": e120,
                "error_iter_300": e300,
                "mejora_abs_0_a_120": e0 - e120,
                "mejora_pct_0_a_120": 100 * (e0 - e120) / max(e0, EPS),
                "mejora_abs_120_a_300": e120 - e300,
                "mejora_pct_120_a_300_sobre_120": 100 * (e120 - e300) / max(e120, EPS),
                "porcentaje_mejora_total_antes_120": 100 * (e0 - e120) / total,
            }
        )
    comparacion = pd.DataFrame(filas_comp)
    guardar_csv(comparacion, carpeta / "convergencia_0_120_vs_120_300.csv")

    mesetas = []
    for (capa, rango), grupo in resumen.groupby(["capa", "rango"]):
        grupo = grupo.sort_values("iteracion")
        serie = grupo["error_relativo"].to_numpy()
        iter_meseta = 300
        for i in range(10, len(serie)):
            mejora_10 = 100 * (serie[i - 10] - serie[i]) / max(serie[i - 10], EPS)
            if mejora_10 < 0.1:
                iter_meseta = i
                break
        mesetas.append({"capa": capa, "rango": rango, "iteracion_meseta_0p1pct_10_iter": iter_meseta})
    mesetas_df = pd.DataFrame(mesetas)
    guardar_csv(mesetas_df, carpeta / "convergencia_umbrales_meseta.csv")

    fig, eje = plt.subplots(figsize=(11, 6))
    for (capa, rango), grupo in resumen.groupby(["capa", "rango"]):
        eje.plot(grupo["iteracion"], grupo["error_relativo"], label=f"Capa {capa} rango {rango}")
    eje.axvline(120, color="#e45756", linestyle="--", label="120 iteraciones")
    eje.set_title("Convergencia Deep-ONMF de 0 a 300 iteraciones")
    eje.set_xlabel("Iteracion")
    eje.set_ylabel("Error relativo medio")
    eje.grid(True, alpha=0.25)
    eje.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "convergencia_error_0_300.png", dpi=220)
    plt.close(fig)

    fig, eje = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comparacion))
    eje.bar(x - 0.18, comparacion["mejora_abs_0_a_120"], width=0.36, label="0 a 120")
    eje.bar(x + 0.18, comparacion["mejora_abs_120_a_300"], width=0.36, label="120 a 300")
    eje.set_xticks(x)
    eje.set_xticklabels([f"Capa {c}" for c in comparacion["capa"]])
    eje.set_title("Mejora de error antes y despues de 120 iteraciones")
    eje.set_ylabel("Reduccion absoluta de error")
    eje.grid(True, axis="y", alpha=0.25)
    eje.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "convergencia_mejora_0_120_vs_120_300.png", dpi=220)
    plt.close(fig)
    return detalle, resumen, comparacion


def fmt(valor) -> str:
    if isinstance(valor, (float, np.floating)):
        return f"{float(valor):.4f}"
    return str(valor)


class PDF:
    def __init__(self, path: Path, titulo: str) -> None:
        self.path = path
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=595, height=842)
        self.landscape = False
        self.y = 54
        self.titulo(titulo)

    @property
    def left(self) -> float:
        return 42

    @property
    def right(self) -> float:
        return self.page.rect.width - 42

    @property
    def bottom(self) -> float:
        return self.page.rect.height - 42

    def nueva_pagina(self, landscape: bool | None = None) -> None:
        if landscape is None:
            landscape = self.landscape
        self.landscape = landscape
        self.page = self.doc.new_page(width=842 if landscape else 595, height=595 if landscape else 842)
        self.y = 44

    def asegurar(self, alto: float, landscape: bool | None = None) -> None:
        if self.y + alto > self.bottom:
            self.nueva_pagina(self.landscape if landscape is None else landscape)

    def titulo(self, texto: str) -> None:
        self.page.insert_textbox(
            fitz.Rect(self.left, self.y, self.right, self.y + 60),
            texto,
            fontsize=21,
            fontname="helv",
            color=(0.03, 0.10, 0.20),
        )
        self.y += 54

    def encabezado(self, texto: str, nivel: int = 2) -> None:
        size = 15 if nivel == 2 else 12
        self.asegurar(28)
        self.page.insert_text((self.left, self.y), texto, fontsize=size, fontname="helv", color=(0.03, 0.10, 0.20))
        self.y += size + 10

    def parrafo(self, texto: str, size: float = 9.5) -> None:
        ancho = self.right - self.left
        alto = max(22, 14 * (texto.count("\n") + max(1, len(texto) // (120 if self.landscape else 88))))
        rect = fitz.Rect(self.left, self.y, self.right, min(self.y + alto, self.bottom))
        resto = texto
        while resto:
            self.asegurar(36)
            rect = fitz.Rect(self.left, self.y, self.right, self.bottom)
            rc = self.page.insert_textbox(rect, resto, fontsize=size, fontname="helv", color=(0, 0, 0), lineheight=1.25)
            if rc >= 0:
                used = min(alto, self.bottom - self.y)
                self.y += used + 8
                break
            corte = max(250, int(len(resto) * 0.55))
            punto = resto.rfind(" ", 0, corte)
            if punto < 100:
                punto = corte
            parte, resto = resto[:punto], resto[punto:].lstrip()
            self.page.insert_textbox(rect, parte, fontsize=size, fontname="helv", color=(0, 0, 0), lineheight=1.25)
            self.nueva_pagina()

    def bullets(self, items: list[str]) -> None:
        for item in items:
            self.parrafo("- " + item, size=9.2)

    def imagen(self, path: Path, titulo: str, alto: float = 300, landscape: bool | None = None) -> None:
        if landscape is not None and landscape != self.landscape:
            self.nueva_pagina(landscape)
        self.asegurar(alto + 42, landscape=landscape)
        self.page.insert_text((self.left, self.y), titulo, fontsize=10.5, fontname="helv", color=(0.03, 0.10, 0.20))
        self.y += 16
        rect = fitz.Rect(self.left, self.y, self.right, self.y + alto)
        self.page.insert_image(rect, filename=str(path), keep_proportion=True)
        self.y += alto + 18

    def tabla(self, df: pd.DataFrame, titulo: str, max_rows: int | None = None) -> None:
        data = df.head(max_rows).copy() if max_rows else df.copy()
        data = data.fillna("")
        landscape = len(data.columns) > 5
        if landscape != self.landscape:
            self.nueva_pagina(landscape)
        self.encabezado(titulo, nivel=3)
        cols = list(data.columns)
        ancho = self.right - self.left
        pesos = []
        for col in cols:
            valores = data[col].astype(str).str.len()
            pesos.append(max(8, min(22, max(len(str(col)), int(valores.max()) if len(valores) else 8))))
        total = sum(pesos)
        col_w = [ancho * p / total for p in pesos]
        font = 6.6 if landscape else 7.4
        row_h = 38 if landscape else 42

        def celda(texto: str, rect: fitz.Rect, bold: bool = False, fill: tuple[float, float, float] | None = None) -> None:
            self.page.draw_rect(rect, color=(0.65, 0.67, 0.72), fill=fill, width=0.35)
            max_chars = max(5, int(rect.width / (font * 0.54)))
            texto = fmt(texto)
            lineas = textwrap.wrap(texto, width=max_chars)[:3]
            if not lineas:
                lineas = [""]
            y = rect.y0 + 8
            for linea in lineas:
                self.page.insert_text((rect.x0 + 3, y), linea, fontsize=font, fontname="helv", color=(0, 0, 0))
                y += font + 3

        def cabecera() -> None:
            x = self.left
            for i, col in enumerate(cols):
                celda(str(col), fitz.Rect(x, self.y, x + col_w[i], self.y + row_h), fill=(0.86, 0.91, 0.98))
                x += col_w[i]
            self.y += row_h

        self.asegurar(row_h * 2)
        cabecera()
        for row in data.itertuples(index=False, name=None):
            if self.y + row_h > self.bottom:
                self.nueva_pagina(landscape)
                self.encabezado(titulo + " (continua)", nivel=3)
                cabecera()
            x = self.left
            for i, value in enumerate(row):
                celda(value, fitz.Rect(x, self.y, x + col_w[i], self.y + row_h))
                x += col_w[i]
            self.y += row_h
        self.y += 12

    def guardar(self) -> None:
        self.doc.save(self.path, deflate=True, garbage=4, clean=True)
        self.doc.close()


def tabla_metricas_doc(metricas: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "metodo",
        "muestras",
        "rasgos_originales",
        "rasgos_entrada_tsne",
        "silhouette_features",
        "davies_bouldin_features",
        "silhouette_tsne",
        "davies_bouldin_tsne",
    ]
    return metricas[cols].copy()


def generar_documento_v1(auditoria, metricas, carpetas):
    pdf = PDF(RESULTADOS / "Documento explicativo v1.pdf", "Documento explicativo v1")
    pdf.encabezado("1. Objetivo de esta version")
    pdf.parrafo(
        "Este documento rehace la entrega final tomando como referencia el articulo objetivo y el documento de tareas. "
        "La idea central es separar tres cosas que antes estaban mezcladas: la implementacion fiel del articulo, la "
        "comparativa generada con codigo propio y la mejora encontrada para Deep-ONMF. Todas las figuras incluidas aqui "
        "proceden de CSV y matrices generadas en esta ejecucion; no se insertan recortes del paper como si fueran resultados propios."
    )
    pdf.parrafo(
        "El resultado se organiza como una memoria tecnica: primero se documenta exactamente que parte del articulo se ha "
        "implementado, despues se explica como se han generado las caracteristicas de cada metodo, y finalmente se comparan "
        "los espacios de rasgos y los mapas t-SNE. Esto es importante para defender el TFG porque evita presentar una figura "
        "bonita sin justificar de donde salen sus coordenadas, sus colores y sus metricas."
    )
    pdf.encabezado("2. Parametros fieles al articulo")
    pdf.bullets(
        [
            "Los audios se dividen en tramas de 2 segundos con solape de 1 segundo.",
            "Los audios menores de 2 segundos se descartan para mantener la consistencia que describe el articulo.",
            "Cada trama se transforma con STFT usando ventana Hamming de 150 muestras, salto de 75 muestras y FFT de 250 puntos.",
            "El espectrograma tiene 126 bins de frecuencia. Las matrices de cada clase se concatenan por columnas.",
            "Deep-ONMF usa tres capas con rangos 9, 8 y 7. El producto final W1 W2 W3 contiene 7 SBV por clase.",
        ]
    )
    pdf.tabla(auditoria, "Auditoria de datos usada en la reproduccion fiel")
    pipeline = pd.DataFrame(
        [
            {"fase": "Lectura WAV", "decision": "usar las cinco clases N, AS, MR, MS y MVP", "motivo": "mantener el escenario de Figura 11"},
            {"fase": "Duracion", "decision": "descartar audios menores de 2 s", "motivo": "reproducir el criterio textual del articulo"},
            {"fase": "Framing", "decision": "tramas de 2 s con solape de 1 s", "motivo": "cubrir ciclos cardiacos completos y continuidad temporal"},
            {"fase": "STFT", "decision": "Hamming 150, salto 75, FFT 250", "motivo": "parametros publicados en el articulo"},
            {"fase": "Deep-ONMF", "decision": "capas 9-8-7", "motivo": "extraer 7 SBV finales por clase"},
            {"fase": "Visualizacion", "decision": "t-SNE con semilla fija", "motivo": "comparar todos los metodos con el mismo criterio"},
        ]
    )
    pdf.tabla(pipeline, "Trazabilidad del flujo implementado")
    pdf.encabezado("3. Implementacion de CNN, MFCC, DWT y STFT")
    pdf.parrafo(
        "La CNN se ha corregido para que la arquitectura principal coincida con el texto del articulo: primera convolucion "
        "con 32 filtros de tamano 3x3 y ReLU, segunda convolucion con 64 filtros de tamano 3x3 y ReLU, y una capa de "
        "max-pooling de tamano 1x6. Para poder entrenarla en Python se anade una cabeza local de clasificacion despues "
        "del bloque publicado; las caracteristicas comparadas se extraen del bloque convolucional, no de un modelo previo guardado."
    )
    pdf.parrafo(
        "MFCC se implementa con 40 filtros Mel y 13 coeficientes finales. La senal se convierte primero en el espectrograma "
        "STFT de la misma reproduccion y despues se aplica el banco Mel, el logaritmo y la DCT. DWT usa la wavelet coif5, "
        "que es la wavelet indicada por el articulo; el nivel de descomposicion se fija como parametro local de reproduccion "
        "porque el texto no concreta ese nivel. STFT aparece como comparacion adicional local: el articulo la usa para formar "
        "la matriz tiempo-frecuencia de Deep-ONMF, pero la Figura 11 original no incluye un panel STFT independiente."
    )
    tabla_metodos = pd.DataFrame(
        [
            {
                "metodo": "CNN",
                "paper": "Conv1 32 filtros 3x3, Conv2 64 filtros 3x3, Pool 1x6",
                "codigo": "clase CNNArticulo con esas capas y una cabeza local de entrenamiento",
                "rasgos": "64 activaciones tras pooling global",
            },
            {
                "metodo": "MFCC",
                "paper": "40 filtros Mel y 13 coeficientes",
                "codigo": "banco Mel sobre espectrograma STFT, log y DCT",
                "rasgos": "13 medias cepstrales por audio",
            },
            {
                "metodo": "DWT",
                "paper": "wavelet coif5",
                "codigo": "pywt.wavedec con coif5 y nivel local reproducible",
                "rasgos": "energia log, media absoluta y desviacion por bloque",
            },
            {
                "metodo": "STFT",
                "paper": "base tiempo-frecuencia para Deep-ONMF",
                "codigo": "resumen directo de espectrograma",
                "rasgos": "media y desviacion de 126 bins",
            },
        ]
    )
    pdf.tabla(tabla_metodos, "Correspondencia entre articulo, codigo y rasgos")
    pdf.encabezado("4. Deep-ONMF normal frente a Deep-ONMF mejorado")
    pdf.parrafo(
        "Deep-ONMF normal resume cada audio mediante las activaciones medias asociadas a la base de su clase, produciendo "
        "siete rasgos SBV. La mejora mantiene el entrenamiento Deep-ONMF, pero cambia la representacion final: cada audio "
        "se reconstruye contra las cinco bases finales W_N, W_AS, W_MR, W_MS y W_MVP. Los cinco errores de reconstruccion "
        "se convierten mediante softmin en un perfil de afinidad por clase. Esa representacion responde a una pregunta "
        "mas discriminativa: que diccionario de clase explica mejor el audio."
    )
    diferencias = pd.DataFrame(
        [
            {"parte": "Entrada", "Deep-ONMF normal": "STFT por clase", "Deep-ONMF mejorado": "STFT por clase"},
            {"parte": "Factorizacion", "Deep-ONMF normal": "W1,H1; W2,H2; W3,H3", "Deep-ONMF mejorado": "Misma factorizacion"},
            {"parte": "Rasgos finales", "Deep-ONMF normal": "Media de H final: 7 SBV", "Deep-ONMF mejorado": "Afinidades por error: 5 rasgos"},
            {"parte": "Decision geometrica", "Deep-ONMF normal": "Activacion interna", "Deep-ONMF mejorado": "Reconstruccion frente a cada clase"},
            {"parte": "Motivo de mejora", "Deep-ONMF normal": "Puede perder contraste entre clases", "Deep-ONMF mejorado": "Aumenta separacion entre diccionarios"},
        ]
    )
    pdf.tabla(diferencias, "Cuadro de diferencias de codigo y representacion")
    pseudocodigo = pd.DataFrame(
        [
            {"bloque": "Normal", "operacion": "X_clase -> W1,H1 -> W2,H2 -> W3,H3", "salida": "H3 medio por audio"},
            {"bloque": "Mejorado", "operacion": "audio -> error contra W_N,W_AS,W_MR,W_MS,W_MVP", "salida": "perfil softmin de afinidad"},
            {"bloque": "Ventaja", "operacion": "comparar reconstruccion entre diccionarios", "salida": "rasgos directamente ligados a la clase"},
        ]
    )
    pdf.tabla(pseudocodigo, "Diferencia operativa resumida")
    pdf.encabezado("5. Comparativa principal")
    pdf.tabla(tabla_metricas_doc(metricas), "Metricas principales")
    pdf.imagen(carpetas["comparativa"] / "comparativa_principal_tsne.png", "Figura nueva: t-SNE de CNN, DWT, MFCC, STFT, Deep-ONMF normal y mejorado", alto=360, landscape=True)
    pdf.imagen(carpetas["comparativa"] / "comparativa_principal_metricas.png", "Resumen grafico de metricas", alto=310, landscape=True)
    pdf.encabezado("6. Como leer los valores")
    pdf.parrafo(
        "Silhouette mide si cada punto esta mas cerca de los puntos de su propia clase que de los puntos de otras clases. "
        "Por eso sube cuando las nubes del t-SNE se vuelven compactas y separadas, y baja cuando varias clases se mezclan "
        "o aparecen puentes entre grupos. Davies-Bouldin mide la relacion entre dispersion interna y distancia entre centros; "
        "en este caso interesa que sea bajo. Si dos clases quedan cerca o una clase se reparte en varias islas alejadas, "
        "Davies-Bouldin empeora aunque visualmente haya algunos grupos bonitos."
    )
    mejor_sil_tsne = metricas.sort_values("silhouette_tsne", ascending=False).iloc[0]
    mejor_db_tsne = metricas.sort_values("davies_bouldin_tsne", ascending=True).iloc[0]
    pdf.parrafo(
        f"En esta ejecucion, el mejor silhouette t-SNE corresponde a {mejor_sil_tsne['metodo']} con "
        f"{mejor_sil_tsne['silhouette_tsne']:.4f}. El menor Davies-Bouldin t-SNE corresponde a "
        f"{mejor_db_tsne['metodo']} con {mejor_db_tsne['davies_bouldin_tsne']:.4f}. La interpretacion no se basa en una "
        "sola cifra aislada: se mira simultaneamente compacidad, separacion y estabilidad visual de las cinco clases."
    )
    pdf.encabezado("7. Lectura metodo por metodo")
    for _, fila in metricas.iterrows():
        metodo = fila["metodo"]
        pdf.parrafo(
            f"{metodo}: utiliza {int(fila['rasgos_originales'])} rasgos y {int(fila['rasgos_entrada_tsne'])} dimensiones "
            f"de entrada para t-SNE. Su silhouette en rasgos es {fila['silhouette_features']:.4f} y su Davies-Bouldin en "
            f"rasgos es {fila['davies_bouldin_features']:.4f}. En la proyeccion t-SNE obtiene silhouette "
            f"{fila['silhouette_tsne']:.4f} y Davies-Bouldin {fila['davies_bouldin_tsne']:.4f}. Si el silhouette es bajo, "
            "significa que existen zonas de mezcla entre clases o que una clase queda repartida en regiones diferentes. "
            "Si el Davies-Bouldin es alto, la penalizacion viene de grupos dispersos o de centros demasiado cercanos."
        )
    pdf.encabezado("8. Por que la mejora cambia la grafica")
    pdf.parrafo(
        "En Deep-ONMF normal, el vector de un audio se obtiene desde activaciones internas. Esa informacion es util, pero "
        "no compara explicitamente el audio con las bases de las otras clases. En el mejorado, cada audio produce cinco "
        "errores de reconstruccion. Si un audio de MR se reconstruye claramente mejor con W_MR que con W_N, W_AS, W_MS o "
        "W_MVP, su afinidad MR sube y las demas bajan. Esa diferencia se convierte en una separacion geometrica mas directa. "
        "Por eso las nubes del t-SNE pueden moverse y compactarse: no se estan usando rasgos nuevos inventados, sino otra "
        "forma de leer las bases aprendidas por Deep-ONMF."
    )
    pdf.encabezado("9. Figuras del flujo del articulo")
    pdf.imagen(carpetas["articulo"] / "figura_5_sbv_por_clase.png", "Figura 5 reproducida: SBV por clase", alto=420)
    pdf.imagen(carpetas["articulo"] / "figura_7_distancias_sbv.png", "Figura 7 reproducida: distancias entre y dentro de clases", alto=420)
    pdf.guardar()


def generar_documento_v2(metricas_variantes, convergencia, carpetas):
    pdf = PDF(RESULTADOS / "Documento explicativo v2.pdf", "Documento explicativo v2")
    pdf.encabezado("1. Punto de partida: Deep-ONMF mejorado")
    pdf.parrafo(
        "La segunda version parte de la mejora defendible: no se sustituye Deep-ONMF por otro modelo, sino que se aprovecha "
        "mejor lo que Deep-ONMF aprende. Las bases W finales se usan como diccionarios de clase y cada audio se describe por "
        "su perfil de afinidad de reconstruccion. A partir de esa version se prueban las inicializaciones NNDSVD, NNDSVDa y "
        "NNDSVDar, siempre dentro del flujo mejorado."
    )
    pdf.parrafo(
        "En esta version se comparan cinco filas y no se incluye una fila adicional de arranque base: Deep-ONMF normal, "
        "Deep-ONMF mejorado, Deep-ONMF mejorado con NNDSVD, con NNDSVDa y con NNDSVDar. La fila normal se mantiene porque "
        "sirve como referencia del articulo; las tres inicializaciones se aplican sobre el flujo mejorado, es decir, despues "
        "se calculan errores de reconstruccion por clase y afinidades softmin."
    )
    esquema = pd.DataFrame(
        [
            {"variante": "Deep-ONMF normal", "entrenamiento": "ONMF 9-8-7", "rasgo final": "7 activaciones/SBV", "uso": "referencia base"},
            {"variante": "Deep-ONMF mejorado", "entrenamiento": "ONMF 9-8-7", "rasgo final": "5 afinidades softmin", "uso": "mejora principal"},
            {"variante": "Mejorado + NNDSVD", "entrenamiento": "ONMF iniciado por SVD no negativa", "rasgo final": "5 afinidades softmin", "uso": "estabilidad estructurada"},
            {"variante": "Mejorado + NNDSVDa", "entrenamiento": "NNDSVD con ceros rellenados con media", "rasgo final": "5 afinidades softmin", "uso": "evitar ceros muertos"},
            {"variante": "Mejorado + NNDSVDar", "entrenamiento": "NNDSVD con ruido pequeno en ceros", "rasgo final": "5 afinidades softmin", "uso": "variabilidad controlada"},
        ]
    )
    pdf.tabla(esquema, "Que se compara en v2")
    pdf.encabezado("2. Comparacion de variantes Deep-ONMF")
    pdf.tabla(tabla_metricas_doc(metricas_variantes), "Deep-ONMF normal, mejorado e inicializaciones")
    pdf.imagen(carpetas["inicializaciones"] / "comparativa_variantes_deep_onmf_tsne.png", "Figura nueva: variantes Deep-ONMF", alto=360, landscape=True)
    pdf.imagen(carpetas["inicializaciones"] / "comparativa_variantes_deep_onmf_metricas.png", "Metricas de variantes Deep-ONMF", alto=310, landscape=True)
    pdf.encabezado("3. Explicacion de NNDSVD, NNDSVDa y NNDSVDar")
    pdf.parrafo(
        "NNDSVD inicializa W y H usando la descomposicion SVD y separando partes positivas y negativas para obtener matrices "
        "no negativas. Su ventaja es que parte de una estructura alineada con la energia dominante de la matriz, en vez de "
        "empezar desde numeros arbitrarios. NNDSVDa rellena los ceros con la media de la matriz, lo que evita zonas muertas. "
        "NNDSVDar rellena esos ceros con ruido pequeno proporcional a la media, lo que introduce variabilidad controlada."
    )
    mejor = metricas_variantes.sort_values(["silhouette_tsne", "davies_bouldin_tsne"], ascending=[False, True]).iloc[0]
    pdf.parrafo(
        f"Segun la tabla generada, la variante con mayor lectura t-SNE combinada es {mejor['metodo']}. "
        f"Su silhouette t-SNE es {mejor['silhouette_tsne']:.4f} y su Davies-Bouldin t-SNE es "
        f"{mejor['davies_bouldin_tsne']:.4f}. Si una inicializacion mejora el error pero empeora silhouette o Davies-Bouldin, "
        "la conclusion correcta es que reconstruye numericamente bien, pero no necesariamente separa mejor las clases."
    )
    pdf.encabezado("4. Lectura de los resultados de variantes")
    for _, fila in metricas_variantes.iterrows():
        pdf.parrafo(
            f"{fila['metodo']}: trabaja con {int(fila['rasgos_originales'])} rasgos y alcanza silhouette t-SNE "
            f"{fila['silhouette_tsne']:.4f} con Davies-Bouldin t-SNE {fila['davies_bouldin_tsne']:.4f}. "
            "La lectura de la grafica depende de ambos valores: silhouette premia que cada punto quede rodeado por puntos "
            "de su clase, mientras que Davies-Bouldin castiga dispersion y proximidad entre centros. Cuando una variante "
            "aumenta silhouette pero tambien aumenta Davies-Bouldin, hay mejora local de vecindad pero no necesariamente "
            "mejora global de compacidad."
        )
    explicacion_valores = pd.DataFrame(
        [
            {"metrica": "silhouette_features", "sube cuando": "los rasgos originales separan clases", "baja cuando": "hay mezcla antes del t-SNE"},
            {"metrica": "davies_bouldin_features", "sube cuando": "los grupos son dispersos o cercanos", "baja cuando": "cada clase es compacta y distante"},
            {"metrica": "silhouette_tsne", "sube cuando": "el mapa 2D conserva vecindades de clase", "baja cuando": "aparecen puentes o solapes"},
            {"metrica": "davies_bouldin_tsne", "sube cuando": "el mapa tiene islas dispersas o centros proximos", "baja cuando": "las nubes 2D son compactas"},
        ]
    )
    pdf.tabla(explicacion_valores, "Por que cambian los valores de las graficas")
    pdf.encabezado("5. Convergencia de 0 a 300 iteraciones")
    pdf.parrafo(
        "El estudio de convergencia no se limita a justificar 120 iteraciones por conveniencia. Se calcula el error relativo "
        "desde la iteracion 0 hasta la 300 y se compara la mejora acumulada antes y despues de 120. La clave defensiva es "
        "ver cuanto mejora realmente el metodo tras 120 iteraciones: si la mejora de 120 a 300 es pequena frente a la mejora "
        "de 0 a 120, entonces 120 es un punto razonable de coste-beneficio."
    )
    pdf.tabla(convergencia, "Mejora de convergencia 0-120 frente a 120-300")
    for _, fila in convergencia.iterrows():
        pdf.parrafo(
            f"Capa {int(fila['capa'])}, rango {int(fila['rango'])}: el error pasa de {fila['error_iter_0']:.4f} "
            f"a {fila['error_iter_120']:.4f} en 120 iteraciones y termina en {fila['error_iter_300']:.4f} a las 300. "
            f"Antes de 120 se obtiene el {fila['porcentaje_mejora_total_antes_120']:.2f}% de la mejora total observada. "
            f"Despues de 120 la mejora adicional es {fila['mejora_abs_120_a_300']:.4f}, que equivale a "
            f"{fila['mejora_pct_120_a_300_sobre_120']:.2f}% respecto al error que quedaba en la iteracion 120."
        )
    pdf.imagen(carpetas["convergencia"] / "convergencia_error_0_300.png", "Curvas de error relativo medio de 0 a 300", alto=330)
    pdf.imagen(carpetas["convergencia"] / "convergencia_mejora_0_120_vs_120_300.png", "Reduccion de error antes y despues de 120 iteraciones", alto=300)
    pdf.encabezado("6. Defensa ante tribunal")
    pdf.parrafo(
        "La defensa debe explicar que se ha respetado el flujo del articulo y despues se ha probado una mejora concreta y medible. "
        "El cambio no consiste en dibujar una grafica mas favorable, sino en cambiar la representacion final de cada audio: de "
        "activaciones internas a afinidades por reconstruccion frente a las cinco bases de clase. Por eso los valores de las "
        "graficas cambian: cuando una clase se reconstruye claramente mejor con su propio diccionario, su punto se aleja de las "
        "otras clases en el espacio de rasgos y en el t-SNE; cuando dos patologias comparten patrones espectrales, aparecen "
        "zonas proximas, baja silhouette y sube Davies-Bouldin."
    )
    pdf.guardar()


def verificar_pdfs(carpetas) -> None:
    filas = []
    for pdf_path in [RESULTADOS / "Documento explicativo v1.pdf", RESULTADOS / "Documento explicativo v2.pdf"]:
        doc = fitz.open(pdf_path)
        textos = [len(page.get_text().strip()) for page in doc]
        imagenes = [len(page.get_images(full=True)) for page in doc]
        filas.append(
            {
                "pdf": pdf_path.name,
                "paginas": len(doc),
                "paginas_vacias": sum(1 for t, im in zip(textos, imagenes) if t == 0 and im == 0),
                "imagenes": sum(imagenes),
                "texto_min_pagina": min(textos) if textos else 0,
            }
        )
        for indice in [0, min(1, len(doc) - 1)]:
            pix = doc[indice].get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
            pix.save(carpetas["verificacion"] / f"preview_{pdf_path.stem}_pagina_{indice + 1}.png")
        doc.close()
    guardar_csv(pd.DataFrame(filas), carpetas["verificacion"] / "verificacion_pdfs.csv")


def main() -> None:
    np.random.seed(SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if "--solo-documentos" in sys.argv:
        carpetas = carpetas_existentes()
        shutil.copyfile(Path(__file__), carpetas["codigo"] / "01_generar_entrega_final.py")
        auditoria = pd.read_csv(carpetas["articulo"] / "auditoria_articulo_fiel.csv", encoding="utf-8-sig")
        metricas = pd.read_csv(carpetas["comparativa"] / "metricas_comparativa_principal.csv", encoding="utf-8-sig")
        metricas_variantes = pd.read_csv(
            carpetas["inicializaciones"] / "metricas_deep_onmf_variantes.csv",
            encoding="utf-8-sig",
        )
        conv_comp = pd.read_csv(carpetas["convergencia"] / "convergencia_0_120_vs_120_300.csv", encoding="utf-8-sig")
        for pdf in [RESULTADOS / "Documento explicativo v1.pdf", RESULTADOS / "Documento explicativo v2.pdf"]:
            if pdf.exists():
                pdf.unlink()
        generar_documento_v1(auditoria, metricas, carpetas)
        generar_documento_v2(metricas_variantes, conv_comp, carpetas)
        verificar_pdfs(carpetas)
        print("[ok] documentos regenerados desde resultados existentes")
        return
    carpetas = preparar_carpetas()
    config = Configuracion(raiz=ROOT, rellenar_audios_cortos=False)

    registros, datos_por_clase, auditoria = descubrir_y_preparar(config, carpetas)
    resultados_random = entrenar_deep_onmf(datos_por_clase, config, "aleatoria", carpetas["articulo"])
    registros_usados, normal, mejorado, metricas, coords = generar_comparativas(
        registros,
        datos_por_clase,
        resultados_random,
        config,
        carpetas,
    )
    generar_figuras_articulo(datos_por_clase, resultados_random, normal, config, carpetas)
    metricas_variantes = generar_inicializaciones(
        registros_usados,
        datos_por_clase,
        normal,
        mejorado,
        metricas,
        config,
        carpetas,
    )
    _, _, conv_comp = estudio_convergencia(datos_por_clase, config, carpetas["convergencia"])
    generar_documento_v1(auditoria, metricas, carpetas)
    generar_documento_v2(metricas_variantes, conv_comp, carpetas)
    verificar_pdfs(carpetas)
    print("[ok] entrega final generada en", RESULTADOS)


if __name__ == "__main__":
    main()
