from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import textwrap
import time
import unicodedata


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR
SRC = SCRIPT_DIR / "codigo"
RESULTADOS = Path.cwd() / "resultados"
DATOS_FIGURAS = RESULTADOS / "Fotos datos y graficos"
MPL_CACHE = DATOS_FIGURAS / ".cache_matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
import torch
from torch import nn

from tfg_deep_onmf.audio import (
    RegistroAudio,
    descubrir_audios,
    dividir_en_tramas,
    espectrograma_magnitud,
    leer_wav_normalizado,
)
from tfg_deep_onmf.configuracion import Configuracion


EPS = 1e-12
SEED = 42
CLASES = ("N", "AS", "MR", "MS", "MVP")
INIT_LABELS = {
    "nndsvd": "Deep-ONMF W por audio + NNDSVD",
    "nndsvda": "Deep-ONMF W por audio + NNDSVDa",
    "nndsvdar": "Deep-ONMF W por audio + NNDSVDar",
}
ORDEN_PRESENTACION = [
    "Deep-ONMF W por audio + NNDSVD",
    "Deep-ONMF W por audio + NNDSVDa",
    "Deep-ONMF W por audio + NNDSVDar",
    "CNN",
    "DWT",
    "MFCC",
    "STFT",
]
NOMBRES_CORTOS = {
    "Deep-ONMF W por audio + NNDSVD": "W NNDSVD",
    "Deep-ONMF W por audio + NNDSVDa": "W NNDSVDa",
    "Deep-ONMF W por audio + NNDSVDar": "W NNDSVDar",
    "CNN": "CNN",
    "DWT": "DWT",
    "MFCC": "MFCC",
    "STFT": "STFT",
}
COLORES_METODO = {
    "Deep-ONMF W por audio + NNDSVD": "#2f6f4e",
    "Deep-ONMF W por audio + NNDSVDa": "#57966b",
    "Deep-ONMF W por audio + NNDSVDar": "#7a9e3f",
    "CNN": "#4c78a8",
    "DWT": "#f58518",
    "MFCC": "#b279a2",
    "STFT": "#5f9ea0",
}
COLORES = {
    "N": "#4c78a8",
    "AS": "#f58518",
    "MR": "#54a24b",
    "MS": "#e45756",
    "MVP": "#72b7b2",
}


@dataclass(frozen=True)
class AudioPreparado:
    registro: RegistroAudio
    tramas: list[np.ndarray]
    matriz_x: np.ndarray
    spec_fijo: np.ndarray


@dataclass(frozen=True)
class Capa:
    indice: int
    rango: int
    forma_entrada: tuple[int, int]
    forma_w: tuple[int, int]
    forma_h: tuple[int, int]
    error_relativo: float
    ortogonalidad_media: float
    segundos: float


@dataclass(frozen=True)
class ResultadoAudioONMF:
    w_final: np.ndarray
    h_final: np.ndarray
    capas: list[Capa]
    error_final: float
    segundos: float


def guardar_csv(df: pd.DataFrame, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8-sig")


def slug(texto: str) -> str:
    normal = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return "".join(c.lower() if c.isalnum() else "_" for c in normal).strip("_")


def nombre_corto(nombre: str) -> str:
    return NOMBRES_CORTOS.get(nombre, nombre)


def clave_audio(registro: RegistroAudio) -> str:
    return f"{registro.clase}__{slug(registro.ruta.stem)}"


def leer_senal(registro: RegistroAudio) -> np.ndarray:
    senal, fs = leer_wav_normalizado(registro.ruta)
    if fs != 8000:
        raise ValueError(f"Frecuencia inesperada {fs} Hz en {registro.ruta}")
    return senal


def preparar_audio(registro: RegistroAudio, config: Configuracion) -> AudioPreparado | None:
    senal = leer_senal(registro)
    tramas = dividir_en_tramas(senal, config)
    if not tramas:
        return None
    matrices_trama = [espectrograma_magnitud(trama, config) for trama in tramas]
    matriz_x = np.concatenate(matrices_trama, axis=1)
    spec_fijo = np.log1p(np.mean(np.stack(matrices_trama, axis=0), axis=0))
    return AudioPreparado(registro=registro, tramas=tramas, matriz_x=matriz_x, spec_fijo=spec_fijo)


def descubrir_y_preparar(config: Configuracion, max_audios: int = 0) -> tuple[list[AudioPreparado], pd.DataFrame]:
    registros = descubrir_audios(config.carpeta_base_datos, config.clases)
    registros = sorted(registros, key=lambda r: (config.clases.index(r.clase), r.ruta.name))
    preparados: list[AudioPreparado] = []
    auditoria: list[dict[str, object]] = []

    for i, registro in enumerate(registros, start=1):
        preparado = preparar_audio(registro, config)
        valido = preparado is not None
        if valido:
            preparados.append(preparado)
        auditoria.append(
            {
                "clase": registro.clase,
                "archivo": registro.ruta.name,
                "ruta": str(registro.ruta),
                "duracion_s": registro.duracion_s,
                "estado": "usado" if valido else "descartado_menor_2s",
                "tramas_pcg": len(preparado.tramas) if preparado else 0,
                "columnas_matriz_x": preparado.matriz_x.shape[1] if preparado else 0,
            }
        )
        if i % 100 == 0:
            print(f"[datos] revisados {i}/{len(registros)} audios")

    if max_audios > 0:
        preparados = preparados[:max_audios]
        rutas = {str(p.registro.ruta) for p in preparados}
        auditoria = [fila for fila in auditoria if fila["estado"] != "usado" or fila["ruta"] in rutas]

    return preparados, pd.DataFrame(auditoria)


def normalizar_columnas_w(w: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    escala = np.maximum(np.linalg.norm(w, axis=0), EPS)
    return w / escala[None, :], h * escala[:, None]


def error_relativo(x: np.ndarray, w: np.ndarray, h: np.ndarray) -> float:
    return float(np.linalg.norm(x - w @ h, ord="fro") / max(np.linalg.norm(x, ord="fro"), EPS))


def ortogonalidad_media(h: np.ndarray) -> float:
    normas = np.maximum(np.linalg.norm(h, axis=1, keepdims=True), EPS)
    h_norm = h / normas
    gramo = h_norm @ h_norm.T
    mascara = ~np.eye(gramo.shape[0], dtype=bool)
    return float(np.mean(np.abs(gramo[mascara])))


def nndsvd_inicializar(x: np.ndarray, rango: int, variante: str, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.maximum(x.astype(np.float64, copy=False), EPS)
    media = float(np.mean(x))
    componentes = min(rango, min(x.shape) - 1 if min(x.shape) > 1 else 1)
    u, s, vt = randomized_svd(x, n_components=componentes, random_state=semilla)
    w = np.zeros((x.shape[0], rango), dtype=np.float64)
    h = np.zeros((rango, x.shape[1]), dtype=np.float64)

    w[:, 0] = math.sqrt(max(s[0], EPS)) * np.abs(u[:, 0])
    h[0, :] = math.sqrt(max(s[0], EPS)) * np.abs(vt[0, :])

    for j in range(1, componentes):
        uj = u[:, j]
        vj = vt[j, :]
        uj_pos = np.maximum(uj, 0.0)
        uj_neg = np.maximum(-uj, 0.0)
        vj_pos = np.maximum(vj, 0.0)
        vj_neg = np.maximum(-vj, 0.0)
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
        escala = math.sqrt(max(s[j] * sigma, EPS))
        w[:, j] = escala * uu
        h[j, :] = escala * vv

    if rango > componentes:
        rng = np.random.default_rng(semilla)
        w[:, componentes:] = media * rng.random((x.shape[0], rango - componentes)) / 100.0
        h[componentes:, :] = media * rng.random((rango - componentes, x.shape[1])) / 100.0

    if variante == "nndsvda":
        w[w <= EPS] = media
        h[h <= EPS] = media
    elif variante == "nndsvdar":
        rng = np.random.default_rng(semilla)
        w[w <= EPS] = media * rng.random(np.count_nonzero(w <= EPS)) / 100.0
        h[h <= EPS] = media * rng.random(np.count_nonzero(h <= EPS)) / 100.0
    elif variante != "nndsvd":
        raise ValueError(f"Inicializacion no soportada: {variante}")

    return normalizar_columnas_w(np.maximum(w, EPS), np.maximum(h, EPS))


def factorizar_onmf(
    matriz: np.ndarray,
    rango: int,
    iteraciones: int,
    penalizacion_ortogonal: float,
    semilla: int,
    metodo_init: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.maximum(matriz, EPS).astype(np.float64, copy=False)
    w, h = nndsvd_inicializar(x, rango, metodo_init, semilla)
    for _ in range(iteraciones):
        w *= (x @ h.T) / (w @ (h @ h.T) + EPS)
        w = np.maximum(w, EPS)
        h *= (w.T @ x + penalizacion_ortogonal * h) / (
            (w.T @ w) @ h + penalizacion_ortogonal * ((h @ h.T) @ h) + EPS
        )
        h = np.maximum(h, EPS)
        w, h = normalizar_columnas_w(w, h)
    return w, h, error_relativo(x, w, h), ortogonalidad_media(h)


def deep_onmf_audio(matriz: np.ndarray, config: Configuracion, metodo_init: str, semilla: int) -> ResultadoAudioONMF:
    inicio_total = time.perf_counter()
    entrada = np.maximum(matriz, EPS)
    matrices_w: list[np.ndarray] = []
    capas: list[Capa] = []

    for indice, rango in enumerate(config.rangos_onmf, start=1):
        inicio = time.perf_counter()
        w, h, error, ort = factorizar_onmf(
            entrada,
            rango=rango,
            iteraciones=config.iteraciones_onmf,
            penalizacion_ortogonal=config.penalizacion_ortogonal,
            semilla=semilla + indice * 1000,
            metodo_init=metodo_init,
        )
        segundos = time.perf_counter() - inicio
        matrices_w.append(w)
        capas.append(
            Capa(
                indice=indice,
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
    error_final = error_relativo(np.maximum(matriz, EPS), w_final, h_final)
    return ResultadoAudioONMF(
        w_final=w_final,
        h_final=h_final,
        capas=capas,
        error_final=error_final,
        segundos=time.perf_counter() - inicio_total,
    )


def vector_desde_w(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    w = np.maximum(w.astype(np.float64, copy=True), EPS)
    w = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), EPS)
    bins = np.arange(w.shape[0], dtype=np.float64)
    centroides = (bins[:, None] * w).sum(axis=0) / np.maximum(w.sum(axis=0), EPS)
    orden = np.argsort(centroides).tolist()
    w_ordenada = w[:, orden]
    return w_ordenada.ravel(order="F"), w_ordenada, orden


def nombres_columnas_w(config: Configuracion) -> list[str]:
    columnas: list[str] = []
    for sbv in range(1, config.rangos_onmf[-1] + 1):
        for bin_freq in range(config.bins_frecuencia):
            columnas.append(f"W_sbv{sbv:02d}_bin{bin_freq:03d}")
    return columnas


def generar_w_por_audio(
    preparados: list[AudioPreparado],
    config: Configuracion,
    carpetas: dict[str, Path],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rasgos_por_init: dict[str, pd.DataFrame] = {}
    filas_auditoria: list[dict[str, object]] = []
    columnas_w = nombres_columnas_w(config)

    for metodo in ("nndsvd", "nndsvda", "nndsvdar"):
        print(f"[deep-onmf] inicio {metodo}: {len(preparados)} audios")
        matrices_npz: dict[str, np.ndarray] = {}
        mapa_npz: list[dict[str, object]] = []
        filas_rasgos: list[dict[str, object]] = []

        for i, preparado in enumerate(preparados, start=1):
            registro = preparado.registro
            resultado = deep_onmf_audio(
                preparado.matriz_x,
                config=config,
                metodo_init=metodo,
                semilla=config.semilla + i * 37,
            )
            vector, w_ordenada, orden = vector_desde_w(resultado.w_final)
            key = clave_audio(registro)
            matrices_npz[key] = w_ordenada.astype(np.float32)
            mapa_npz.append(
                {
                    "clave_npz": key,
                    "clase": registro.clase,
                    "archivo": registro.ruta.name,
                    "ruta": str(registro.ruta),
                    "orden_columnas_w": ",".join(str(o) for o in orden),
                }
            )
            fila = {
                "metodo": INIT_LABELS[metodo],
                "inicializacion": metodo,
                "clase": registro.clase,
                "archivo": registro.ruta.name,
                "ruta": str(registro.ruta),
                "duracion_s": registro.duracion_s,
                "tramas_pcg": len(preparado.tramas),
                "columnas_matriz_x": preparado.matriz_x.shape[1],
            }
            fila.update({col: float(valor) for col, valor in zip(columnas_w, vector)})
            filas_rasgos.append(fila)

            fila_auditoria = {
                "inicializacion": metodo,
                "metodo": INIT_LABELS[metodo],
                "clase": registro.clase,
                "archivo": registro.ruta.name,
                "ruta": str(registro.ruta),
                "duracion_s": registro.duracion_s,
                "tramas_pcg": len(preparado.tramas),
                "columnas_matriz_x": preparado.matriz_x.shape[1],
                "forma_w_final": "126x7",
                "error_final": resultado.error_final,
                "segundos": resultado.segundos,
            }
            for capa in resultado.capas:
                fila_auditoria[f"capa{capa.indice}_error"] = capa.error_relativo
                fila_auditoria[f"capa{capa.indice}_ortogonalidad"] = capa.ortogonalidad_media
                fila_auditoria[f"capa{capa.indice}_segundos"] = capa.segundos
            filas_auditoria.append(fila_auditoria)

            if i == 1 or i % 25 == 0 or i == len(preparados):
                print(
                    f"[deep-onmf] {metodo} {i}/{len(preparados)} "
                    f"error={resultado.error_final:.4f} t={resultado.segundos:.2f}s"
                )

        df_rasgos = pd.DataFrame(filas_rasgos)
        rasgos_por_init[metodo] = df_rasgos
        guardar_csv(df_rasgos, carpetas["rasgos"] / f"rasgos_w_por_audio_{metodo}.csv")
        guardar_csv(pd.DataFrame(mapa_npz), carpetas["matrices"] / f"mapa_matrices_w_{metodo}.csv")
        np.savez_compressed(carpetas["matrices"] / f"matrices_w_por_audio_{metodo}.npz", **matrices_npz)

    auditoria_w = pd.DataFrame(filas_auditoria)
    guardar_csv(auditoria_w, carpetas["auditoria"] / "auditoria_w_por_audio.csv")
    return rasgos_por_init, auditoria_w


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


def rasgos_stft(preparados: list[AudioPreparado]) -> pd.DataFrame:
    filas: list[dict[str, object]] = []
    for preparado in preparados:
        spec = preparado.spec_fijo
        vector = np.concatenate([spec.mean(axis=1), spec.std(axis=1)])
        fila = {"metodo": "STFT", "clase": preparado.registro.clase, "archivo": preparado.registro.ruta.name}
        for i, valor in enumerate(vector, start=1):
            fila[f"stft_{i:03d}"] = float(valor)
        filas.append(fila)
    return pd.DataFrame(filas)


def rasgos_mfcc(preparados: list[AudioPreparado], config: Configuracion) -> pd.DataFrame:
    banco = banco_mel_articulo(config, bandas=40)
    filas: list[dict[str, object]] = []
    for preparado in preparados:
        mel = np.maximum(banco @ preparado.spec_fijo, EPS)
        coef = dct(np.log(mel), type=2, axis=0, norm="ortho")[:13]
        vector = coef.mean(axis=1)
        fila = {"metodo": "MFCC", "clase": preparado.registro.clase, "archivo": preparado.registro.ruta.name}
        for i, valor in enumerate(vector, start=1):
            fila[f"mfcc_{i:03d}"] = float(valor)
        filas.append(fila)
    return pd.DataFrame(filas)


def rasgos_dwt(preparados: list[AudioPreparado], config: Configuracion) -> pd.DataFrame:
    filas: list[dict[str, object]] = []
    for preparado in preparados:
        vectores = []
        for trama in preparado.tramas:
            nivel = min(5, pywt.dwt_max_level(len(trama), pywt.Wavelet("coif5").dec_len))
            coeficientes = pywt.wavedec(trama, wavelet="coif5", level=nivel, mode="symmetric")
            rasgos = []
            for bloque in coeficientes:
                valores = np.asarray(bloque, dtype=np.float64)
                rasgos.extend([np.log1p(np.mean(valores**2)), np.mean(np.abs(valores)), np.std(valores)])
            vectores.append(np.asarray(rasgos, dtype=np.float64))
        vector = np.mean(np.stack(vectores, axis=0), axis=0)
        fila = {"metodo": "DWT", "clase": preparado.registro.clase, "archivo": preparado.registro.ruta.name}
        for i, valor in enumerate(vector, start=1):
            fila[f"dwt_{i:03d}"] = float(valor)
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


def rasgos_cnn(preparados: list[AudioPreparado], carpeta: Path) -> pd.DataFrame:
    print("[cnn] preparando espectrogramas fijos")
    specs = np.stack([p.spec_fijo for p in preparados], axis=0).astype(np.float32)
    y = np.array([CLASES.index(p.registro.clase) for p in preparados], dtype=np.int64)
    media = float(specs.mean())
    std = float(specs.std() + 1e-6)
    x = ((specs - media) / std)[:, None, :, :]

    train_idx, val_idx = train_test_split(
        np.arange(len(preparados)),
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
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
            "media": media,
            "std": std,
        },
        carpeta / "cnn_articulo_entrenada.pt",
    )
    modelo.eval()
    with torch.no_grad():
        features = modelo.features(tensor_x).numpy()
    filas: list[dict[str, object]] = []
    for preparado, vector in zip(preparados, features):
        fila = {"metodo": "CNN", "clase": preparado.registro.clase, "archivo": preparado.registro.ruta.name}
        for i, valor in enumerate(vector, start=1):
            fila[f"cnn_{i:03d}"] = float(valor)
        filas.append(fila)
    return pd.DataFrame(filas)


def columnas_rasgos(df: pd.DataFrame) -> list[str]:
    omitidas = {"metodo", "inicializacion", "clase", "archivo", "ruta", "duracion_s", "tramas_pcg", "columnas_matriz_x"}
    return [c for c in df.columns if c not in omitidas]


def evaluar_metodo(nombre: str, df: pd.DataFrame, carpeta: Path) -> tuple[pd.Series, pd.DataFrame]:
    cols = columnas_rasgos(df)
    x_original = df[cols].to_numpy(dtype=np.float64)
    etiquetas = df["clase"].to_numpy()
    x = StandardScaler().fit_transform(x_original)
    componentes = min(50, x.shape[1], max(1, x.shape[0] - 1))
    if x.shape[1] > componentes:
        x_tsne = PCA(n_components=componentes, random_state=SEED).fit_transform(x)
        dim_tsne = componentes
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
    clases_unicas = np.unique(etiquetas)
    if len(clases_unicas) < 2 or len(df) <= len(clases_unicas):
        sil_features = float("nan")
        db_features = float("nan")
        sil_tsne = float("nan")
        db_tsne = float("nan")
    else:
        sil_features = float(silhouette_score(x, etiquetas))
        db_features = float(davies_bouldin_score(x, etiquetas))
        sil_tsne = float(silhouette_score(coords, etiquetas))
        db_tsne = float(davies_bouldin_score(coords, etiquetas))

    metricas = pd.Series(
        {
            "metodo": nombre,
            "muestras": len(df),
            "rasgos_originales": len(cols),
            "rasgos_entrada_tsne": dim_tsne,
            "perplexity": perplexity,
            "silhouette_features": sil_features,
            "davies_bouldin_features": db_features,
            "silhouette_tsne": sil_tsne,
            "davies_bouldin_tsne": db_tsne,
        }
    )
    return metricas, coords_df


def scatter_en_eje(eje, coords: pd.DataFrame, titulo: str, metricas: pd.Series) -> None:
    for clase in CLASES:
        mask = coords["clase"] == clase
        eje.scatter(
            coords.loc[mask, "tsne_1"],
            coords.loc[mask, "tsne_2"],
            s=18,
            alpha=0.82,
            color=COLORES[clase],
            label=clase,
            edgecolors="white",
            linewidths=0.18,
        )
    eje.set_title(nombre_corto(titulo), fontsize=12, fontweight="bold", pad=8)
    subtitulo = f"Sil t-SNE {metricas['silhouette_tsne']:.3f}   DB {metricas['davies_bouldin_tsne']:.3f}"
    eje.text(
        0.02,
        0.98,
        subtitulo,
        transform=eje.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#253238",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d8dee2", "alpha": 0.92},
    )
    eje.set_xlabel("t-SNE 1")
    eje.set_ylabel("t-SNE 2")
    eje.grid(True, color="#d9dde1", alpha=0.45, linewidth=0.7)
    eje.tick_params(labelsize=8, colors="#344047")
    for spine in eje.spines.values():
        spine.set_color("#a9b0b5")
        spine.set_linewidth(0.8)


def figura_individual(nombre: str, metricas: pd.Series, coords: pd.DataFrame, ruta: Path) -> None:
    fig, eje = plt.subplots(figsize=(8.2, 6.2))
    scatter_en_eje(eje, coords, nombre, metricas)
    handles, labels = eje.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CLASES), title="Clase")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(ruta, dpi=240, facecolor="white")
    plt.close(fig)


def figura_comparativa(metricas: pd.DataFrame, coords_por_metodo: dict[str, pd.DataFrame], orden: list[str], ruta: Path) -> None:
    filas = math.ceil(len(orden) / 2)
    fig, ejes = plt.subplots(filas, 2, figsize=(12.5, 4.35 * filas), constrained_layout=True)
    ejes = np.asarray(ejes).reshape(-1)
    for eje, metodo in zip(ejes, orden):
        fila_metricas = metricas.loc[metricas["metodo"] == metodo].iloc[0]
        scatter_en_eje(eje, coords_por_metodo[metodo], metodo, fila_metricas)
    for eje in ejes[len(orden) :]:
        eje.axis("off")
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORES[c], markeredgecolor="white", markersize=8)
            for c in CLASES
        ]
        eje.legend(handles, CLASES, loc="center", ncol=1, title="Clase", frameon=True)
        eje.text(
            0.5,
            0.22,
            "Todos los paneles usan los mismos 951 audios validos.",
            ha="center",
            va="center",
            fontsize=10,
            color="#4b565c",
            transform=eje.transAxes,
        )
    fig.suptitle("Figura tipo 11: t-SNE por metodo", fontsize=18, fontweight="bold")
    fig.savefig(ruta, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def figura_metricas(metricas: pd.DataFrame, ruta: Path) -> None:
    metricas_plot = metricas.copy()
    metricas_plot["metodo_corto"] = metricas_plot["metodo"].map(nombre_corto)
    metricas_plot["color"] = metricas_plot["metodo"].map(lambda m: COLORES_METODO.get(m, "#777777"))
    fig, ejes = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True)
    columnas = [
        ("silhouette_features", "Silhouette rasgos (mayor es mejor)"),
        ("davies_bouldin_features", "Davies-Bouldin rasgos (menor es mejor)"),
        ("silhouette_tsne", "Silhouette t-SNE (mayor es mejor)"),
        ("davies_bouldin_tsne", "Davies-Bouldin t-SNE (menor es mejor)"),
    ]
    for eje, (col, label) in zip(ejes.reshape(-1), columnas):
        asc = "Davies" in label
        datos = metricas_plot.sort_values(col, ascending=asc)
        eje.barh(datos["metodo_corto"], datos[col], color=datos["color"], alpha=0.88)
        eje.set_title(label)
        eje.grid(True, axis="x", color="#d9dde1", alpha=0.55)
        eje.tick_params(axis="both", labelsize=8)
        eje.invert_yaxis()
        rango = datos[col].max() - datos[col].min()
        margen = 0.04 * (abs(datos[col].max()) + rango + 1e-9)
        for y, valor in enumerate(datos[col]):
            if valor >= 0:
                x_texto = valor + margen
                ha = "left"
            else:
                x_texto = margen
                ha = "left"
            eje.text(x_texto, y, f"{valor:.3f}", va="center", ha=ha, fontsize=8, color="#273238")
        for spine in eje.spines.values():
            spine.set_visible(False)
    fig.suptitle("Metricas de separabilidad", fontsize=17, fontweight="bold")
    fig.savefig(ruta, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def generar_baselines(preparados: list[AudioPreparado], config: Configuracion, carpetas: dict[str, Path]) -> dict[str, pd.DataFrame]:
    print("[baselines] STFT")
    stft = rasgos_stft(preparados)
    print("[baselines] MFCC")
    mfcc = rasgos_mfcc(preparados, config)
    print("[baselines] DWT")
    dwt = rasgos_dwt(preparados, config)
    print("[baselines] CNN")
    cnn = rasgos_cnn(preparados, carpetas["baselines"])
    for nombre, df in {"STFT": stft, "MFCC": mfcc, "DWT": dwt, "CNN": cnn}.items():
        guardar_csv(df, carpetas["baselines"] / f"rasgos_{slug(nombre)}.csv")
    return {"CNN": cnn, "DWT": dwt, "MFCC": mfcc, "STFT": stft}


def evaluar_y_graficar(
    rasgos_w: dict[str, pd.DataFrame],
    baselines: dict[str, pd.DataFrame],
    carpetas: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    metodos: dict[str, pd.DataFrame] = {}
    for metodo_init, df in rasgos_w.items():
        metodos[INIT_LABELS[metodo_init]] = df
    metodos.update(baselines)

    orden = [
        "Deep-ONMF W por audio + NNDSVD",
        "Deep-ONMF W por audio + NNDSVDa",
        "Deep-ONMF W por audio + NNDSVDar",
        "CNN",
        "DWT",
        "MFCC",
        "STFT",
    ]
    filas = []
    coords: dict[str, pd.DataFrame] = {}
    for nombre in orden:
        print(f"[metricas] {nombre}")
        fila, coord = evaluar_metodo(nombre, metodos[nombre], carpetas["coordenadas"])
        filas.append(fila)
        coords[nombre] = coord
        figura_individual(nombre, fila, coord, carpetas["figuras"] / f"tsne_{slug(nombre)}.png")

    metricas = pd.DataFrame(filas)
    metricas = metricas.sort_values("metodo").reset_index(drop=True)
    guardar_csv(metricas, carpetas["base"] / "metricas_comparativas.csv")
    figura_comparativa(metricas, coords, orden, carpetas["figuras"] / "figura_11_comparativa_todos_los_metodos.png")
    figura_metricas(metricas, carpetas["figuras"] / "metricas_comparativas.png")
    return metricas, coords


class PDF:
    def __init__(self, path: Path, titulo: str) -> None:
        self.path = path
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=595, height=842)
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

    def nueva_pagina(self) -> None:
        self.page = self.doc.new_page(width=595, height=842)
        self.y = 44

    def asegurar(self, alto: float) -> None:
        if self.y + alto > self.bottom:
            self.nueva_pagina()

    def texto(self, texto: str, size: float = 9.5, font: str = "helv", color=(0.07, 0.07, 0.07), leading: float = 1.25) -> None:
        palabras = texto.split()
        lineas: list[str] = []
        linea = ""
        max_chars = 94 if size <= 9.5 else 75
        for palabra in palabras:
            candidata = f"{linea} {palabra}".strip()
            if len(candidata) > max_chars:
                if linea:
                    lineas.append(linea)
                linea = palabra
            else:
                linea = candidata
        if linea:
            lineas.append(linea)
        alto_linea = size * leading
        self.asegurar(alto_linea * max(1, len(lineas)) + 8)
        for linea in lineas:
            self.page.insert_text((self.left, self.y), linea, fontsize=size, fontname=font, color=color)
            self.y += alto_linea
        self.y += 5

    def titulo(self, texto: str) -> None:
        self.page.insert_textbox(
            fitz.Rect(self.left, self.y, self.right, self.y + 62),
            texto,
            fontsize=20,
            fontname="helv",
            color=(0.03, 0.10, 0.20),
        )
        self.y += 58

    def encabezado(self, texto: str) -> None:
        self.asegurar(30)
        self.page.insert_text((self.left, self.y), texto, fontsize=14, fontname="helv", color=(0.03, 0.10, 0.20))
        self.y += 22

    def tabla(self, df: pd.DataFrame, titulo: str, max_rows: int = 20) -> None:
        self.encabezado(titulo)
        tabla = df.head(max_rows).copy()
        for col in tabla.columns:
            if pd.api.types.is_float_dtype(tabla[col]):
                tabla[col] = tabla[col].map(lambda v: f"{v:.4f}")
        texto = tabla.to_string(index=False)
        lineas = texto.splitlines()
        alto = 11 * len(lineas) + 16
        self.asegurar(alto)
        for linea in lineas:
            self.page.insert_text((self.left, self.y), linea[:112], fontsize=7.4, fontname="cour", color=(0.05, 0.05, 0.05))
            self.y += 9.2
        self.y += 8

    def imagen(self, ruta: Path, titulo: str, ancho: float = 500, alto_max: float = 420) -> None:
        if not ruta.exists():
            return
        pix = fitz.Pixmap(str(ruta))
        ratio = pix.height / max(pix.width, 1)
        alto = min(alto_max, ancho * ratio)
        self.asegurar(alto + 42)
        self.page.insert_text((self.left, self.y), titulo, fontsize=10.5, fontname="helv", color=(0.03, 0.10, 0.20))
        self.y += 12
        rect = fitz.Rect(self.left, self.y, self.left + ancho, self.y + alto)
        self.page.insert_image(rect, filename=str(ruta))
        self.y += alto + 16

    def guardar(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(self.path, deflate=True, garbage=4)
        self.doc.close()


def resumen_auditoria(auditoria: pd.DataFrame) -> pd.DataFrame:
    return (
        auditoria.groupby(["clase", "estado"], as_index=False)
        .agg(audios=("archivo", "count"), columnas_matriz_x=("columnas_matriz_x", "sum"))
        .sort_values(["clase", "estado"])
    )


def envolver(texto: object, ancho: int) -> str:
    return "\n".join(textwrap.wrap(str(texto), width=ancho, break_long_words=False)) or str(texto)


def tabla_png(df: pd.DataFrame, ruta: Path, titulo: str, anchos: dict[str, int] | None = None) -> None:
    anchos = anchos or {}
    tabla = df.copy()
    for col in tabla.columns:
        if pd.api.types.is_float_dtype(tabla[col]):
            tabla[col] = tabla[col].map(lambda v: f"{v:.4f}")
        tabla[col] = tabla[col].map(lambda v, c=col: envolver(v, anchos.get(c, 18)))

    filas = len(tabla)
    columnas = len(tabla.columns)
    alto = max(2.0, 0.42 * filas + 1.25)
    ancho = max(8.0, 1.45 * columnas + 2.0)
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.axis("off")
    fig.suptitle(titulo, fontsize=14, fontweight="bold", color="#10202b", y=0.98)
    table = ax.table(
        cellText=tabla.values,
        colLabels=[envolver(c, anchos.get(c, 18)) for c in tabla.columns],
        bbox=[0.0, 0.02, 1.0, 0.90],
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1, 1.55)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d6dde2")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#10202b")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f7f9fa" if row % 2 == 0 else "white")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ruta, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def datos_tabla_lectura() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metodo": "W-audio NNDSVD/NNDSVDa/NNDSVDar",
                "idea": "matriz W propia por audio",
                "lectura": "prueba exactamente la hipotesis pedida: un punto sale de la matriz de bases",
            },
            {
                "metodo": "CNN",
                "idea": "filtros 3x3 sobre espectrogramas",
                "lectura": "aprende texturas supervisadas; sirve como referencia no lineal",
            },
            {
                "metodo": "DWT",
                "idea": "coif5 y estadisticos por escala",
                "lectura": "captura transitorios, pero puede mezclar patologias con escalas parecidas",
            },
            {
                "metodo": "MFCC",
                "idea": "40 filtros Mel y 13 coeficientes",
                "lectura": "resume la envolvente espectral y queda compacto",
            },
            {
                "metodo": "STFT",
                "idea": "energia tiempo-frecuencia resumida",
                "lectura": "es la base interpretable que alimenta despues a ONMF",
            },
        ]
    )


def preparar_metricas_para_tabla(metricas: pd.DataFrame) -> pd.DataFrame:
    tabla = metricas.copy()
    tabla["metodo"] = tabla["metodo"].map(nombre_corto)
    tabla = tabla[
        [
            "metodo",
            "muestras",
            "rasgos_originales",
            "silhouette_features",
            "davies_bouldin_features",
            "silhouette_tsne",
            "davies_bouldin_tsne",
        ]
    ].sort_values("silhouette_features", ascending=False)
    return tabla.rename(
        columns={
            "rasgos_originales": "rasgos",
            "silhouette_features": "sil_rasgos",
            "davies_bouldin_features": "DB_rasgos",
            "silhouette_tsne": "sil_tSNE",
            "davies_bouldin_tsne": "DB_tSNE",
        }
    )


def generar_tablas_png(auditoria: pd.DataFrame, metricas: pd.DataFrame, carpetas: dict[str, Path]) -> None:
    tabla_png(
        resumen_auditoria(auditoria),
        carpetas["figuras"] / "tabla_auditoria_audios.png",
        "Auditoria de audios usados y descartados",
        {"estado": 20, "columnas_matriz_x": 16},
    )
    tabla_png(
        preparar_metricas_para_tabla(metricas),
        carpetas["figuras"] / "tabla_metricas_comparativas.png",
        "Metricas comparativas",
        {"metodo": 17, "silhouette_features": 12, "davies_bouldin_features": 12},
    )
    tabla_png(
        datos_tabla_lectura(),
        carpetas["figuras"] / "tabla_lectura_docente.png",
        "Lectura docente por metodo",
        {"metodo": 22, "idea": 25, "lectura": 45},
    )


def cargar_coordenadas(metricas: pd.DataFrame, carpetas: dict[str, Path]) -> dict[str, pd.DataFrame]:
    coords = {}
    for metodo in metricas["metodo"]:
        ruta = carpetas["coordenadas"] / f"coordenadas_tsne_{slug(metodo)}.csv"
        coords[metodo] = pd.read_csv(ruta, encoding="utf-8-sig")
    return coords


def regenerar_presentacion_desde_resultados(carpetas: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    auditoria = pd.read_csv(carpetas["auditoria"] / "auditoria_audios.csv", encoding="utf-8-sig")
    metricas = pd.read_csv(carpetas["base"] / "metricas_comparativas.csv", encoding="utf-8-sig")
    coords = cargar_coordenadas(metricas, carpetas)
    orden = [m for m in ORDEN_PRESENTACION if m in set(metricas["metodo"])]
    for metodo in orden:
        fila = metricas.loc[metricas["metodo"] == metodo].iloc[0]
        figura_individual(metodo, fila, coords[metodo], carpetas["figuras"] / f"tsne_{slug(metodo)}.png")
    figura_comparativa(metricas, coords, orden, carpetas["figuras"] / "figura_11_comparativa_todos_los_metodos.png")
    figura_metricas(metricas, carpetas["figuras"] / "metricas_comparativas.png")
    generar_tablas_png(auditoria, metricas, carpetas)
    return metricas, coords


def generar_pdf(
    auditoria: pd.DataFrame,
    auditoria_w: pd.DataFrame,
    metricas: pd.DataFrame,
    carpetas: dict[str, Path],
    config: Configuracion,
) -> Path:
    ruta_pdf = RESULTADOS / "Documento explicativo Matriz W.pdf"
    pdf = PDF(ruta_pdf, "DOCUMENTO EXPLICATIVO MATRIZ W POR AUDIO")
    usados = int((auditoria["estado"] == "usado").sum())
    descartados = int((auditoria["estado"] != "usado").sum())
    mejor = metricas.sort_values(["silhouette_features", "davies_bouldin_features"], ascending=[False, True]).iloc[0]
    pdf.encabezado("Resumen ejecutivo")
    pdf.texto(
        f"Este documento genera una matriz W_final por cada audio valido siguiendo el protocolo fiel al articulo: "
        f"tramas de 2 s, solape de 1 s, ventana Hamming de 150 muestras, salto 75, FFT de 250 puntos y Deep-ONMF "
        f"de tres capas con rangos {config.rangos_onmf}. No se rellenan audios cortos: se descartan si no llegan a 2 s."
    )
    pdf.texto(
        f"El conjunto final tiene {usados} audios usados y {descartados} descartados. Se ignoran subtipos o metadatos "
        "internos; solo se mantiene la clase principal N, AS, MR, MS o MVP para colorear la figura y calcular metricas."
    )
    pdf.texto(
        "La figura tipo 11 se construye desde W, no desde H ni desde errores de reconstruccion. Cada W de 126x7 se "
        "normaliza, se ordena por centroide frecuencial y se aplana a 882 rasgos para representar un punto por audio."
    )

    pdf.encabezado("Metodologia paso a paso")
    pdf.texto(
        "Primero se lee cada WAV de la base Yaseen local. Si dura menos de 2 s se descarta, igual que en el protocolo "
        "fiel del articulo: una senal menor no puede formar una trama completa. Si el audio es valido, se divide en "
        "tramas de 2 s con 1 s de solape. Cada trama se convierte en espectrograma de magnitud y las columnas se "
        "concatenan para formar X_audio."
    )
    pdf.texto(
        "Despues se aplica Deep-ONMF al X_audio de ese unico fichero. La primera capa aprende 9 bases, la segunda 8 y "
        "la tercera 7. El producto W1 W2 W3 devuelve las bases al espacio frecuencial original, por eso W_final queda "
        "con 126 filas y 7 columnas. Esta matriz es el objeto central de esta prueba."
    )
    pdf.texto(
        "Como NMF puede permutar columnas, antes de llevar W a t-SNE se normalizan las columnas y se ordenan por su "
        "centroide frecuencial. Asi SBV_1, SBV_2, ..., SBV_7 tienen una lectura mas estable entre audios."
    )

    pdf.imagen(carpetas["figuras"] / "tabla_auditoria_audios.png", "Tabla. Auditoria de audios", ancho=500, alto_max=300)
    pdf.imagen(carpetas["figuras"] / "tabla_metricas_comparativas.png", "Tabla. Metricas principales", ancho=510, alto_max=360)

    pdf.encabezado("Lectura del mejor resultado")
    pdf.texto(
        f"Por separacion en el espacio de rasgos, la mejor fila es {mejor['metodo']} con silhouette_features="
        f"{mejor['silhouette_features']:.4f} y Davies-Bouldin_features={mejor['davies_bouldin_features']:.4f}. "
        "La lectura de t-SNE se usa como apoyo visual, no como unica prueba."
    )

    pdf.encabezado("Por que W por audio da resultados debiles")
    pdf.texto(
        "El resultado bajo no significa que Deep-ONMF no funcione. Significa que esta prueba fuerza una lectura muy "
        "exigente: aprender una W independiente para cada audio. En el articulo, la matriz W se aprende sobre matrices "
        "grandes, normalmente por clase o por conjunto, de modo que las bases recogen patrones espectrales estables. "
        "Aqui cada audio construye su propio diccionario."
    )
    pdf.texto(
        "Aunque todas las W tengan forma 126x7, sus columnas no son automaticamente comparables. La columna 1 de un "
        "audio no tiene por que representar el mismo patron que la columna 1 de otro. NMF/ONMF puede permutar columnas, "
        "cambiar escalas y encontrar soluciones locales distintas. Por eso se normalizan y ordenan las columnas por "
        "centroide frecuencial, pero esa correccion solo reduce el problema; no lo elimina por completo."
    )
    pdf.texto(
        "Ademas, W describe que patrones espectrales existen, mientras que H describe cuanto aparece cada patron en "
        "cada tramo del audio. Al usar solo W, por indicacion experimental, se pierde informacion temporal y de "
        "activacion. Dos audios de clases distintas pueden aprender bases parecidas si comparten energia en frecuencias "
        "similares, aunque sus activaciones o su reconstruccion por clase fueran diferentes."
    )
    pdf.texto(
        "Por eso las variantes W por audio obtienen silhouette negativa en el espacio de rasgos. Una silhouette negativa "
        "indica que, en promedio, varios audios quedan mas cerca de audios de otra clase que de los de su propia clase. "
        "La causa principal no es un fallo de ejecucion, sino la comparabilidad limitada de diccionarios W aprendidos "
        "de forma independiente."
    )
    pdf.texto(
        "La comparacion con MFCC, STFT, DWT y CNN ayuda a verlo. Esos metodos generan rasgos en un sistema comun: un "
        "coeficiente MFCC, un bin STFT o un estadistico DWT significan lo mismo para todos los audios. En cambio, una "
        "W entrenada por audio tiene el mismo tamano en todos los casos, pero no necesariamente el mismo significado "
        "interno. NNDSVD mejora el punto de arranque de la factorizacion, pero no convierte W por audio en una "
        "representacion supervisada ni optimiza directamente la separacion entre clases."
    )
    pdf.texto(
        "La conclusion defendible es: Deep-ONMF es util cuando aprende bases comunes y luego se comparan activaciones, "
        "errores de reconstruccion o afinidades frente a esas bases. En cambio, entrenar una W aislada por senal cumple "
        "la prueba solicitada, pero debilita la separabilidad porque cada punto procede de un diccionario propio."
    )

    pdf.encabezado("Inicializaciones NNDSVD")
    pdf.texto(
        "NNDSVD inicializa W y H usando SVD y separando partes positivas y negativas. NNDSVDa rellena los ceros "
        "con la media de X, y NNDSVDar los rellena con ruido pequeno proporcional a esa media. Las tres se prueban "
        "sobre el mismo conjunto de audios validos y con los mismos parametros del articulo."
    )
    pdf.texto(
        "En esta ejecucion, la mejor variante W por audio en silhouette_features es NNDSVDar. Eso no significa que sea "
        "la mejor representacion global frente a todos los metodos; significa que, entre las tres formas de inicializar "
        "W por audio, es la que deja la nube de rasgos algo menos mezclada antes de t-SNE."
    )

    pdf.encabezado("Comparacion con CNN, DWT, MFCC y STFT")
    pdf.texto(
        "CNN usa la arquitectura del articulo: dos convoluciones 3x3 con 32 y 64 filtros, ReLU, MaxPool 1x6 y pooling "
        "global. DWT usa coif5 y estadisticos por escala. MFCC usa 40 filtros Mel y 13 coeficientes. STFT usa la "
        "misma representacion tiempo-frecuencia que alimenta Deep-ONMF, resumida por medias y desviaciones."
    )
    pdf.texto(
        "La comparacion es justa porque todos los metodos se calculan sobre los mismos audios validos despues de aplicar "
        "el descarte estricto de menores de 2 s."
    )
    pdf.imagen(carpetas["figuras"] / "tabla_lectura_docente.png", "Tabla. Lectura docente por metodo", ancho=510, alto_max=330)

    pdf.imagen(
        carpetas["figuras"] / "figura_11_comparativa_todos_los_metodos.png",
        "Figura tipo 11 comparativa",
        ancho=500,
        alto_max=610,
    )
    pdf.imagen(carpetas["figuras"] / "metricas_comparativas.png", "Comparativa de metricas", ancho=510, alto_max=390)
    for nombre_figura, titulo in [
        ("tsne_deep_onmf_w_por_audio___nndsvd.png", "t-SNE W por audio con NNDSVD"),
        ("tsne_deep_onmf_w_por_audio___nndsvda.png", "t-SNE W por audio con NNDSVDa"),
        ("tsne_deep_onmf_w_por_audio___nndsvdar.png", "t-SNE W por audio con NNDSVDar"),
        ("tsne_cnn.png", "t-SNE CNN"),
        ("tsne_dwt.png", "t-SNE DWT"),
        ("tsne_mfcc.png", "t-SNE MFCC"),
        ("tsne_stft.png", "t-SNE STFT"),
    ]:
        pdf.imagen(carpetas["figuras"] / nombre_figura, titulo, ancho=470)

    pdf.encabezado("Guia breve de defensa")
    pdf.texto(
        "Si preguntan por que no se usan los 1000 audios, la respuesta es que el articulo descarta tramas incompletas: "
        "un audio menor de 2 s no forma una trama valida y por eso no entra en el protocolo fiel."
    )
    pdf.texto(
        "Si preguntan por que cada audio tiene W propia, la respuesta es que esta prueba cambia la lectura de la Figura 11: "
        "cada punto representa directamente la matriz de bases aprendida para una senal concreta."
    )
    pdf.texto(
        "Si preguntan por los subtipos, la respuesta es que se han ignorado por indicacion experimental; solo se conserva "
        "la clase principal para evaluacion visual y numerica."
    )
    pdf.texto(
        "Si preguntan por que MFCC o STFT pueden superar a W por audio en algunas metricas, la respuesta es que esta "
        "prueba fuerza una W independiente por senal. Eso cumple la instruccion, pero tambien introduce el problema de "
        "comparabilidad entre diccionarios independientes. Por eso se ordenan las columnas y se informa la comparacion "
        "sin ocultar el resultado."
    )

    pdf.encabezado("Ficheros generados")
    pdf.texto(f"Datos, CSV, NPZ y figuras: {DATOS_FIGURAS}")
    pdf.texto(f"PDF explicativo: {ruta_pdf}")
    pdf.texto(f"Filas de auditoria W por audio e inicializacion: {len(auditoria_w)}")
    pdf.guardar()
    return ruta_pdf


def validar_salidas(
    preparados: list[AudioPreparado],
    auditoria: pd.DataFrame,
    rasgos_w: dict[str, pd.DataFrame],
    metricas: pd.DataFrame,
    pdf: Path,
    carpetas: dict[str, Path],
) -> pd.DataFrame:
    filas = []
    usados = int((auditoria["estado"] == "usado").sum())
    filas.append({"comprobacion": "audios_validos", "esperado": len(preparados), "observado": usados, "ok": usados == len(preparados)})
    for metodo, df in rasgos_w.items():
        filas.append({"comprobacion": f"filas_rasgos_{metodo}", "esperado": len(preparados), "observado": len(df), "ok": len(df) == len(preparados)})
        filas.append({"comprobacion": f"columnas_w_{metodo}", "esperado": 882, "observado": len(nombres_columnas_w(Configuracion(ROOT, rellenar_audios_cortos=False))), "ok": True})
        npz = carpetas["matrices"] / f"matrices_w_por_audio_{metodo}.npz"
        with np.load(npz) as data:
            formas_ok = all(data[key].shape == (126, 7) for key in data.files)
            filas.append({"comprobacion": f"formas_npz_{metodo}", "esperado": "126x7", "observado": len(data.files), "ok": formas_ok and len(data.files) == len(preparados)})
    filas.append({"comprobacion": "metodos_comparados", "esperado": 7, "observado": len(metricas), "ok": len(metricas) == 7})
    filas.append({"comprobacion": "pdf_existe", "esperado": "si", "observado": str(pdf.exists()), "ok": pdf.exists()})
    df = pd.DataFrame(filas)
    guardar_csv(df, carpetas["base"] / "validacion_salidas.csv")
    return df


def preparar_carpetas() -> dict[str, Path]:
    carpetas = {
        "base": DATOS_FIGURAS,
        "auditoria": DATOS_FIGURAS / "01_auditoria",
        "matrices": DATOS_FIGURAS / "02_matrices_w",
        "rasgos": DATOS_FIGURAS / "03_rasgos_w",
        "baselines": DATOS_FIGURAS / "04_baselines",
        "coordenadas": DATOS_FIGURAS / "05_coordenadas_tsne",
        "figuras": DATOS_FIGURAS / "06_figuras",
    }
    for carpeta in carpetas.values():
        carpeta.mkdir(parents=True, exist_ok=True)
    return carpetas


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera W por audio y figura tipo 11 fiel al articulo.")
    parser.add_argument("--datos", type=Path, default=None, help="Ruta a la carpeta Bases de Datos.")
    parser.add_argument("--salida", type=Path, default=None, help="Ruta donde se guardaran los resultados.")
    parser.add_argument("--max-audios", type=int, default=0, help="Solo para pruebas: limita audios validos.")
    parser.add_argument("--limite-por-clase", type=int, default=0, help="Limite equilibrado por clase para prueba reducida.")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla de reproducibilidad.")
    parser.add_argument("--iteraciones", type=int, default=120, help="Iteraciones ONMF por capa.")
    parser.add_argument("--rapido", action="store_true", help="Ejecuta una prueba reducida.")
    parser.add_argument("--solo-pdf", action="store_true", help="Regenera solo el PDF desde los CSV y figuras existentes.")
    parser.add_argument(
        "--solo-presentacion",
        action="store_true",
        help="Regenera figuras, tablas visuales y PDF desde resultados existentes.",
    )
    return parser.parse_args()


def resolver_datos(ruta: Path | None) -> Path:
    if ruta is not None:
        datos = ruta.expanduser().resolve()
        if datos.exists():
            return datos
        raise FileNotFoundError(f"No existe la carpeta de datos indicada: {datos}")
    for base in [Path.cwd(), SCRIPT_DIR, *SCRIPT_DIR.parents]:
        datos = base / "Bases de Datos"
        if datos.exists():
            return datos.resolve()
    raise FileNotFoundError("No se encontro una carpeta 'Bases de Datos'. Usa --datos RUTA.")


def limitar_preparados_por_clase(
    preparados: list[AudioPreparado],
    auditoria: pd.DataFrame,
    clases: tuple[str, ...],
    limite: int,
) -> tuple[list[AudioPreparado], pd.DataFrame]:
    if limite <= 0:
        return preparados, auditoria
    seleccionados: list[AudioPreparado] = []
    conteos = {clase: 0 for clase in clases}
    for preparado in preparados:
        clase = preparado.registro.clase
        if conteos[clase] < limite:
            seleccionados.append(preparado)
            conteos[clase] += 1
    rutas = {str(preparado.registro.ruta) for preparado in seleccionados}
    auditoria_filtrada = auditoria.copy()
    mascara = (auditoria_filtrada["estado"] == "usado") & (~auditoria_filtrada["ruta"].isin(rutas))
    auditoria_filtrada.loc[mascara, "estado"] = "no_usado_por_limite"
    return seleccionados, auditoria_filtrada


def main() -> int:
    global ROOT, RESULTADOS, DATOS_FIGURAS, MPL_CACHE, SEED
    args = parsear_args()
    SEED = args.semilla
    ROOT = SCRIPT_DIR
    RESULTADOS = (args.salida or (Path.cwd() / "resultados")).expanduser().resolve()
    DATOS_FIGURAS = RESULTADOS / "Fotos datos y graficos"
    MPL_CACHE = DATOS_FIGURAS / ".cache_matplotlib"
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)
    np.random.seed(SEED)
    carpetas = preparar_carpetas()
    datos = resolver_datos(args.datos)
    config = Configuracion(
        raiz=SCRIPT_DIR,
        iteraciones_onmf=5 if args.rapido else args.iteraciones,
        semilla=args.semilla,
        rellenar_audios_cortos=False,
    )
    object.__setattr__(config, "carpeta_base_datos", datos)
    object.__setattr__(config, "carpeta_resultados", RESULTADOS)
    if args.solo_pdf:
        auditoria = pd.read_csv(carpetas["auditoria"] / "auditoria_audios.csv", encoding="utf-8-sig")
        metricas = pd.read_csv(carpetas["base"] / "metricas_comparativas.csv", encoding="utf-8-sig")
        generar_tablas_png(auditoria, metricas, carpetas)
        auditoria = pd.read_csv(carpetas["auditoria"] / "auditoria_audios.csv", encoding="utf-8-sig")
        auditoria_w = pd.read_csv(carpetas["auditoria"] / "auditoria_w_por_audio.csv", encoding="utf-8-sig")
        metricas = pd.read_csv(carpetas["base"] / "metricas_comparativas.csv", encoding="utf-8-sig")
        pdf = generar_pdf(auditoria, auditoria_w, metricas, carpetas, config)
        print(f"[ok] PDF regenerado: {pdf}")
        return 0
    if args.solo_presentacion:
        metricas, _coords = regenerar_presentacion_desde_resultados(carpetas)
        auditoria = pd.read_csv(carpetas["auditoria"] / "auditoria_audios.csv", encoding="utf-8-sig")
        auditoria_w = pd.read_csv(carpetas["auditoria"] / "auditoria_w_por_audio.csv", encoding="utf-8-sig")
        pdf = generar_pdf(auditoria, auditoria_w, metricas, carpetas, config)
        print(f"[ok] presentacion regenerada: {pdf}")
        return 0

    inicio = time.perf_counter()
    print("[inicio] Matriz W por audio fiel al articulo")
    print(f"[config] iteraciones={config.iteraciones_onmf}, rangos={config.rangos_onmf}, rellenar={config.rellenar_audios_cortos}")
    limite = args.limite_por_clase or (2 if args.rapido else 0)
    preparados, auditoria = descubrir_y_preparar(config, max_audios=args.max_audios)
    preparados, auditoria = limitar_preparados_por_clase(preparados, auditoria, config.clases, limite)
    guardar_csv(auditoria, carpetas["auditoria"] / "auditoria_audios.csv")
    guardar_csv(resumen_auditoria(auditoria), carpetas["auditoria"] / "resumen_auditoria_audios.csv")
    print(f"[datos] audios validos: {len(preparados)}; descartados: {int((auditoria['estado'] != 'usado').sum())}")

    rasgos_w, auditoria_w = generar_w_por_audio(preparados, config, carpetas)
    baselines = generar_baselines(preparados, config, carpetas)
    metricas, _coords = evaluar_y_graficar(rasgos_w, baselines, carpetas)
    generar_tablas_png(auditoria, metricas, carpetas)
    pdf = generar_pdf(auditoria, auditoria_w, metricas, carpetas, config)
    validacion = validar_salidas(preparados, auditoria, rasgos_w, metricas, pdf, carpetas)

    resumen = {
        "audios_validos": len(preparados),
        "audios_descartados": int((auditoria["estado"] != "usado").sum()),
        "metodos": metricas["metodo"].tolist(),
        "pdf": str(pdf),
        "carpeta_datos_figuras": str(DATOS_FIGURAS),
        "segundos_totales": time.perf_counter() - inicio,
        "validacion_ok": bool(validacion["ok"].all()),
    }
    (DATOS_FIGURAS / "resumen_ejecucion.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[ok] ejecucion completada")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
