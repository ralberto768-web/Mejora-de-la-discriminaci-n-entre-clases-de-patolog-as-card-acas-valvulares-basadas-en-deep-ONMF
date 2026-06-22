from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

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
from sklearn.preprocessing import StandardScaler


CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA / "codigo"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(CARPETA / ".cache_matplotlib"))

from tfg_deep_onmf.audio import (
    construir_matriz_clase,
    descubrir_audios,
    dividir_en_tramas,
    espectrograma_magnitud,
    leer_wav_normalizado,
)
from tfg_deep_onmf.configuracion import Configuracion
from tfg_deep_onmf.estadistica import caracteristicas_por_audio
from tfg_deep_onmf.onmf import deep_onmf


COLORES = {"N": "#1b9e77", "AS": "#d95f02", "MR": "#7570b3", "MS": "#e7298a", "MVP": "#66a61e"}


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparador tipo Figura 11.")
    parser.add_argument("--datos", type=Path, default=None)
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--iteraciones", type=int, default=120)
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


def crear_configuracion(args: argparse.Namespace, datos: Path, salida: Path) -> Configuracion:
    config = Configuracion(
        raiz=CARPETA,
        semilla=args.semilla,
        iteraciones_onmf=5 if args.rapido else args.iteraciones,
        rellenar_audios_cortos=True,
    )
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
        banco[i] = np.maximum(0.0, np.minimum((freqs - izq) / max(cen - izq, 1e-12), (der - freqs) / max(der - cen, 1e-12)))
    return banco


def fila_base(registro) -> dict[str, str]:
    return {"clase": registro.clase, "archivo": registro.ruta.name, "ruta": str(registro.ruta)}


def rasgos_deep_onmf(registros: list, config: Configuracion) -> pd.DataFrame:
    datos_por_clase = {}
    h_por_clase = {}
    for posicion, clase in enumerate(config.clases, start=1):
        datos = construir_matriz_clase(clase, registros, config)
        resultado = deep_onmf(
            datos.matriz,
            rangos=config.rangos_onmf,
            iteraciones=config.iteraciones_onmf,
            penalizacion_ortogonal=config.penalizacion_ortogonal,
            semilla=config.semilla + posicion * 100,
        )
        datos_por_clase[clase] = datos
        h_por_clase[clase] = resultado.h_final
    df = caracteristicas_por_audio(datos_por_clase, h_por_clase)
    return df


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
    banco = banco_mel(config)
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
        vector = []
        for coef in pywt.wavedec(senal, "coif5", level=5):
            vector.extend([float(np.mean(coef)), float(np.std(coef)), float(np.mean(coef**2))])
        fila = fila_base(registro)
        for i, valor in enumerate(vector, start=1):
            fila[f"dwt_{i:03d}"] = valor
        filas.append(fila)
    return pd.DataFrame(filas)


def columnas_rasgos(df: pd.DataFrame) -> list[str]:
    omitidas = {"clase", "archivo", "ruta", "columnas_espectrograma", "duracion_s"}
    return [col for col in df.columns if col not in omitidas]


def evaluar_metodo(nombre: str, df: pd.DataFrame, salida: Path, semilla: int) -> tuple[dict[str, float], pd.DataFrame]:
    cols = columnas_rasgos(df)
    x = np.nan_to_num(df[cols].to_numpy(dtype=np.float64))
    etiquetas = df["clase"].to_numpy()
    x = StandardScaler().fit_transform(x)
    componentes = min(30, x.shape[1], max(1, x.shape[0] - 1))
    x_tsne = PCA(n_components=componentes, random_state=semilla).fit_transform(x) if x.shape[1] > componentes else x
    perplexity = min(30, max(2, (len(df) - 1) // 3))
    perplexity = min(perplexity, max(1, len(df) - 1))
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=semilla, max_iter=1000).fit_transform(x_tsne)
    coords_df = df[["clase", "archivo", "ruta"]].copy()
    coords_df["metodo"] = nombre
    coords_df["tsne_1"] = coords[:, 0]
    coords_df["tsne_2"] = coords[:, 1]
    coords_df.to_csv(salida / f"coordenadas_tsne_{nombre.lower().replace(' ', '_')}.csv", index=False, encoding="utf-8-sig")
    clases_unicas = np.unique(etiquetas)
    if len(clases_unicas) < 2 or len(df) <= len(clases_unicas):
        sil_features = db_features = sil_tsne = db_tsne = float("nan")
    else:
        sil_features = float(silhouette_score(x, etiquetas))
        db_features = float(davies_bouldin_score(x, etiquetas))
        sil_tsne = float(silhouette_score(coords, etiquetas))
        db_tsne = float(davies_bouldin_score(coords, etiquetas))
    return (
        {
            "metodo": nombre,
            "muestras": len(df),
            "rasgos": len(cols),
            "silhouette_features": sil_features,
            "davies_bouldin_features": db_features,
            "silhouette_tsne": sil_tsne,
            "davies_bouldin_tsne": db_tsne,
        },
        coords_df,
    )


def figura_comparativa(coords_por_metodo: dict[str, pd.DataFrame], ruta: Path) -> None:
    metodos = list(coords_por_metodo)
    fig, ejes = plt.subplots(1, len(metodos), figsize=(5 * len(metodos), 4), constrained_layout=True)
    if len(metodos) == 1:
        ejes = [ejes]
    for eje, metodo in zip(ejes, metodos):
        df = coords_por_metodo[metodo]
        for clase, grupo in df.groupby("clase"):
            eje.scatter(grupo["tsne_1"], grupo["tsne_2"], s=18, label=clase, alpha=0.85, color=COLORES.get(clase))
        eje.set_title(metodo)
        eje.set_xticks([])
        eje.set_yticks([])
    ejes[0].legend(loc="best", fontsize=8)
    fig.savefig(ruta, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parsear_argumentos()
    datos = resolver_datos(args.datos)
    salida = (args.salida or (Path.cwd() / "resultados")).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)
    config = crear_configuracion(args, datos, salida)
    registros = descubrir_audios(config.carpeta_base_datos, config.clases)
    registros = limitar_registros(registros, config.clases, args.limite_por_clase or (2 if args.rapido else 0))
    if not registros:
        raise RuntimeError("No se encontraron WAV validos.")

    tablas = {
        "Deep ONMF": rasgos_deep_onmf(registros, config),
        "STFT": rasgos_stft(registros, config),
        "MFCC": rasgos_mfcc(registros, config),
        "DWT": rasgos_dwt(registros, config),
    }
    metricas = []
    coords_por_metodo = {}
    for nombre, df in tablas.items():
        df.to_csv(salida / f"rasgos_{nombre.lower().replace(' ', '_')}.csv", index=False, encoding="utf-8-sig")
        fila, coords = evaluar_metodo(nombre, df, salida, args.semilla)
        metricas.append(fila)
        coords_por_metodo[nombre] = coords
    metricas_df = pd.DataFrame(metricas)
    metricas_df.to_csv(salida / "metricas_comparacion_figura11.csv", index=False, encoding="utf-8-sig")
    figura_comparativa(coords_por_metodo, salida / "figura_comparativa_tipo_11.png")
    validacion = pd.DataFrame(
        [
            {"comprobacion": "metodos", "ok": set(metricas_df["metodo"]) == {"Deep ONMF", "STFT", "MFCC", "DWT"}},
            {"comprobacion": "muestras", "ok": all(metricas_df["muestras"] == len(registros))},
            {"comprobacion": "figura", "ok": (salida / "figura_comparativa_tipo_11.png").exists()},
        ]
    )
    validacion.to_csv(salida / "validacion.csv", index=False, encoding="utf-8-sig")
    print(f"Comparacion completada. Resultados guardados en: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
