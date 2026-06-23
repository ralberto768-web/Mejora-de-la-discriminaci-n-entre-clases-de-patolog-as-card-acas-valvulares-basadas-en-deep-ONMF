from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pywt
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA / "codigo"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(CARPETA / ".cache_matplotlib"))

from tfg_deep_onmf.audio import descubrir_audios, dividir_en_tramas, espectrograma_magnitud, leer_wav_normalizado
from tfg_deep_onmf.configuracion import Configuracion


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae baselines STFT, MFCC y DWT.")
    parser.add_argument("--datos", type=Path, default=None)
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--limite-por-clase", type=int, default=0)
    parser.add_argument("--rapido", action="store_true")
    return parser.parse_args()


def resolver_datos(ruta: Path | None) -> Path:
    if ruta is not None:
        datos = ruta.expanduser().resolve()
        if datos.exists():
            return datos
        raise FileNotFoundError(f"No existe la carpeta de datos indicada: {datos}")
    for base in [Path.cwd(), CARPETA, *CARPETA.parents]:
        datos = base / "Bases de Datos"
        if datos.exists():
            return datos.resolve()
    raise FileNotFoundError("No se encontro una carpeta 'Bases de Datos'. Usa --datos RUTA.")


def crear_configuracion(datos: Path, salida: Path, semilla: int) -> Configuracion:
    config = Configuracion(raiz=CARPETA, semilla=semilla, rellenar_audios_cortos=True)
    object.__setattr__(config, "carpeta_base_datos", datos)
    object.__setattr__(config, "carpeta_resultados", salida)
    return config


def limitar_registros(registros: list, clases: tuple[str, ...], limite: int) -> list:
    if limite <= 0:
        return registros
    seleccionados = []
    conteos = {clase: 0 for clase in clases}
    for registro in sorted(registros, key=lambda r: (clases.index(r.clase), r.ruta.name)):
        if conteos[registro.clase] < limite:
            seleccionados.append(registro)
            conteos[registro.clase] += 1
    return seleccionados


def matriz_audio(registro, config: Configuracion) -> np.ndarray:
    senal, fs = leer_wav_normalizado(registro.ruta)
    if fs != config.frecuencia_esperada_hz:
        raise ValueError(f"{registro.ruta} tiene {fs} Hz; se esperaban {config.frecuencia_esperada_hz} Hz")
    tramas = dividir_en_tramas(senal, config)
    return np.concatenate([espectrograma_magnitud(trama, config) for trama in tramas], axis=1)


def hz_a_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_a_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def banco_mel(config: Configuracion, bandas: int = 40) -> np.ndarray:
    freqs = np.linspace(0, config.frecuencia_esperada_hz / 2, config.bins_frecuencia)
    puntos_mel = np.linspace(hz_a_mel(np.array([0.0]))[0], hz_a_mel(np.array([config.frecuencia_esperada_hz / 2]))[0], bandas + 2)
    puntos_hz = mel_a_hz(puntos_mel)
    banco = np.zeros((bandas, len(freqs)))
    for i in range(bandas):
        izq, cen, der = puntos_hz[i], puntos_hz[i + 1], puntos_hz[i + 2]
        subida = (freqs - izq) / max(cen - izq, 1e-12)
        bajada = (der - freqs) / max(der - cen, 1e-12)
        banco[i] = np.maximum(0.0, np.minimum(subida, bajada))
    return banco


def fila_base(registro) -> dict[str, str]:
    return {"clase": registro.clase, "archivo": registro.ruta.name, "ruta": str(registro.ruta)}


def rasgos_stft(registros: list, config: Configuracion) -> pd.DataFrame:
    filas = []
    for registro in registros:
        matriz = np.log1p(matriz_audio(registro, config))
        vector = np.concatenate([matriz.mean(axis=1), matriz.std(axis=1)])
        fila = fila_base(registro)
        for i, valor in enumerate(vector, start=1):
            fila[f"stft_{i:03d}"] = float(valor)
        filas.append(fila)
    return pd.DataFrame(filas)


def rasgos_mfcc(registros: list, config: Configuracion) -> pd.DataFrame:
    banco = banco_mel(config, bandas=40)
    filas = []
    for registro in registros:
        matriz = matriz_audio(registro, config)
        mel = np.maximum(banco @ matriz, 1e-12)
        mfcc = dct(np.log(mel), type=2, axis=0, norm="ortho")[:13]
        vector = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        fila = fila_base(registro)
        for i, valor in enumerate(vector, start=1):
            fila[f"mfcc_{i:03d}"] = float(valor)
        filas.append(fila)
    return pd.DataFrame(filas)


def rasgos_dwt(registros: list, config: Configuracion) -> pd.DataFrame:
    filas = []
    for registro in registros:
        senal, fs = leer_wav_normalizado(registro.ruta)
        if fs != config.frecuencia_esperada_hz:
            raise ValueError(f"{registro.ruta} tiene {fs} Hz; se esperaban {config.frecuencia_esperada_hz} Hz")
        if len(senal) < config.muestras_trama:
            senal = np.pad(senal, (0, config.muestras_trama - len(senal)), mode="constant")
        else:
            senal = senal[: config.muestras_trama]
        coeficientes = pywt.wavedec(senal, "coif5", level=5)
        vector = []
        for coef in coeficientes:
            vector.extend([float(np.mean(coef)), float(np.std(coef)), float(np.mean(coef**2))])
        fila = fila_base(registro)
        for i, valor in enumerate(vector, start=1):
            fila[f"dwt_{i:03d}"] = valor
        filas.append(fila)
    return pd.DataFrame(filas)


def columnas_rasgos(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in {"clase", "archivo", "ruta"}]


def evaluar_metodo(nombre: str, df: pd.DataFrame, salida: Path, semilla: int) -> dict[str, float]:
    cols = columnas_rasgos(df)
    x = df[cols].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x)
    etiquetas = df["clase"].to_numpy()
    x = StandardScaler().fit_transform(x)
    componentes = min(30, x.shape[1], max(1, x.shape[0] - 1))
    x_tsne = PCA(n_components=componentes, random_state=semilla).fit_transform(x) if x.shape[1] > componentes else x
    perplexity = min(30, max(2, (len(df) - 1) // 3))
    perplexity = min(perplexity, max(1, len(df) - 1))
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=semilla, max_iter=1000).fit_transform(x_tsne)
    coords_df = df[["clase", "archivo", "ruta"]].copy()
    coords_df["tsne_1"] = coords[:, 0]
    coords_df["tsne_2"] = coords[:, 1]
    coords_df.to_csv(salida / f"coordenadas_tsne_{nombre.lower()}.csv", index=False, encoding="utf-8-sig")
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
    return {
        "metodo": nombre.upper(),
        "muestras": len(df),
        "rasgos": len(cols),
        "silhouette_features": sil_features,
        "davies_bouldin_features": db_features,
        "silhouette_tsne": sil_tsne,
        "davies_bouldin_tsne": db_tsne,
    }


def main() -> int:
    args = parsear_argumentos()
    datos = resolver_datos(args.datos)
    salida = (args.salida or (Path.cwd() / "resultados")).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)
    config = crear_configuracion(datos, salida, args.semilla)
    registros = descubrir_audios(config.carpeta_base_datos, config.clases)
    registros = limitar_registros(registros, config.clases, args.limite_por_clase or (2 if args.rapido else 0))
    if not registros:
        raise RuntimeError("No se encontraron WAV validos.")

    tablas = {
        "stft": rasgos_stft(registros, config),
        "mfcc": rasgos_mfcc(registros, config),
        "dwt": rasgos_dwt(registros, config),
    }
    metricas = []
    for nombre, df in tablas.items():
        df.to_csv(salida / f"rasgos_{nombre}.csv", index=False, encoding="utf-8-sig")
        metricas.append(evaluar_metodo(nombre, df, salida, args.semilla))
    metricas_df = pd.DataFrame(metricas)
    metricas_df.to_csv(salida / "metricas_baselines.csv", index=False, encoding="utf-8-sig")
    validacion = pd.DataFrame(
        [
            {"comprobacion": "metodos", "ok": set(metricas_df["metodo"]) == {"STFT", "MFCC", "DWT"}},
            {"comprobacion": "muestras", "ok": all(metricas_df["muestras"] == len(registros))},
            {"comprobacion": "sin_nan_rasgos", "ok": all(not df[columnas_rasgos(df)].isna().any().any() for df in tablas.values())},
        ]
    )
    validacion.to_csv(salida / "validacion.csv", index=False, encoding="utf-8-sig")
    print(f"Baselines completados. Resultados guardados en: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
