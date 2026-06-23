from __future__ import annotations

"""Comparacion final de la Figura 11 del articulo objetivo.

El script usa:
1. WAV reales de Programacion objetivo.
2. SBV por audio ya generados por deep ONMF en Programacion objetivo.
3. Checkpoint CNN de log-mel guardado por Programacion a implemenar.

Genera figuras t-SNE separadas, un PDF comparativo y datos de trazabilidad.
"""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import textwrap
from typing import Callable

CARPETA_COMPARACION = Path(__file__).resolve().parents[1]
MPL_CACHE = CARPETA_COMPARACION / "02_resultados" / ".cache_matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import pywt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Falta PyWavelets para DWT. Instala primero: "
        "python -m pip install -r 01_codigos_ordenados/requirements_comparacion.txt"
    ) from exc


RAIZ_OBJETIVO = CARPETA_COMPARACION.parent
RAIZ_TFG = RAIZ_OBJETIVO.parent
RAIZ_IMPLEMENTAR = RAIZ_TFG / "Programacion a implemenar"
RAIZ_IMPLEMENTACION = RAIZ_IMPLEMENTAR / "Implementacion"
SRC_OBJETIVO = RAIZ_OBJETIVO / "src"

if str(SRC_OBJETIVO) not in sys.path:
    sys.path.insert(0, str(SRC_OBJETIVO))

from tfg_deep_onmf.audio import dividir_en_tramas, espectrograma_magnitud, leer_wav_normalizado
from tfg_deep_onmf.configuracion import Configuracion


CLASES = ("N", "AS", "MR", "MS", "MVP")
COLORES = {
    "N": "#1b9e77",
    "AS": "#d95f02",
    "MR": "#7570b3",
    "MS": "#e7298a",
    "MVP": "#66a61e",
}
ORDEN_METODOS = ("CNN", "DWT", "MFCC", "Deep ONMF", "STFT")
NOMBRES_FIGURA = {
    "CNN": "01_Figura11A_CNN.png",
    "DWT": "02_Figura11B_DWT.png",
    "MFCC": "03_Figura11C_MFCC.png",
    "Deep ONMF": "04_Figura11D_Deep_ONMF.png",
    "STFT": "05_Figura11_extra_STFT.png",
}


@dataclass(frozen=True)
class RegistroComparacion:
    clase: str
    archivo: str
    ruta: Path


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera la comparacion t-SNE de CNN, DWT, MFCC, deep ONMF y STFT.")
    parser.add_argument(
        "--limite-por-clase",
        type=int,
        default=0,
        help="0 usa todos los audios deep ONMF disponibles; otro valor permite una prueba rapida.",
    )
    parser.add_argument("--semilla", type=int, default=42, help="Semilla usada por PCA y t-SNE.")
    parser.add_argument(
        "--ruta-sbv-deep-onmf",
        type=Path,
        default=None,
        help=(
            "CSV caracteristicas_sbv_por_audio.csv a comparar. Si no se indica, "
            "se prioriza el ultimo resultado Deep ONMF con tramas de 2 segundos."
        ),
    )
    return parser.parse_args()


def crear_carpetas_salida() -> dict[str, Path]:
    base = CARPETA_COMPARACION / "02_resultados"
    carpetas = {
        "base": base,
        "figuras": base / "01_figuras_separadas",
        "pdf": base / "02_pdf_comparativo",
        "datos": base / "03_datos_y_metricas",
    }
    for carpeta in carpetas.values():
        carpeta.mkdir(parents=True, exist_ok=True)
    return carpetas


def leer_parametros_deep_onmf(ruta_csv: Path) -> dict[str, object]:
    ruta_parametros = ruta_csv.parent / "parametros_configuracion.json"
    if not ruta_parametros.exists():
        return {}
    try:
        return json.loads(ruta_parametros.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def encontrar_caracteristicas_deep_onmf(ruta_forzada: Path | None) -> Path:
    if ruta_forzada is not None:
        ruta = ruta_forzada.expanduser().resolve()
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el CSV Deep ONMF indicado: {ruta}")
        return ruta

    candidatas = list(
        RAIZ_OBJETIVO.glob("resultados/resultado*/documentacion_tecnica/caracteristicas_sbv_por_audio.csv")
    )
    if not candidatas:
        raise FileNotFoundError(
            "No se ha encontrado caracteristicas_sbv_por_audio.csv en los resultados de Programacion objetivo."
        )

    candidatas_2s = []
    for ruta in candidatas:
        parametros = leer_parametros_deep_onmf(ruta)
        if float(parametros.get("duracion_trama_s", 0.0)) == 2.0:
            candidatas_2s.append(ruta)
    return max(candidatas_2s or candidatas, key=lambda ruta: ruta.stat().st_mtime)


def encontrar_checkpoint_cnn() -> Path:
    candidatas = list((RAIZ_IMPLEMENTACION / "resultados").glob("modelo_cnn_2_0s*.pt"))
    if not candidatas:
        raise FileNotFoundError(
            "No se ha encontrado un checkpoint modelo_cnn_2_0s*.pt en Programacion a implemenar."
        )
    return max(candidatas, key=lambda ruta: ruta.stat().st_mtime)


def seleccionar_deep_onmf(ruta_csv: Path, limite_por_clase: int) -> tuple[pd.DataFrame, list[RegistroComparacion]]:
    df = pd.read_csv(ruta_csv)
    columnas_sbv = [f"SBV_{indice}" for indice in range(1, 8)]
    columnas_necesarias = {"clase", "archivo", "ruta", *columnas_sbv}
    faltantes = columnas_necesarias.difference(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en {ruta_csv}: {sorted(faltantes)}")

    df = df[df["clase"].isin(CLASES)].copy()
    df = df.sort_values(["clase", "archivo"]).reset_index(drop=True)
    if limite_por_clase > 0:
        df = df.groupby("clase", sort=False).head(limite_por_clase).reset_index(drop=True)

    registros: list[RegistroComparacion] = []
    filas_validas: list[int] = []
    for indice, fila in df.iterrows():
        ruta = Path(str(fila["ruta"]))
        if ruta.exists():
            filas_validas.append(indice)
            registros.append(RegistroComparacion(clase=str(fila["clase"]), archivo=str(fila["archivo"]), ruta=ruta))

    if not registros:
        raise FileNotFoundError("Las rutas WAV del CSV deep ONMF no existen en esta copia de Programacion objetivo.")
    return df.loc[filas_validas].reset_index(drop=True), registros


def hz_a_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_a_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def banco_mel(fs: int, n_fft: int, bandas: int) -> np.ndarray:
    puntos_mel = np.linspace(hz_a_mel(np.array([0.0]))[0], hz_a_mel(np.array([fs / 2]))[0], bandas + 2)
    puntos_hz = mel_a_hz(puntos_mel)
    bins = np.floor((n_fft + 1) * puntos_hz / fs).astype(int)
    banco = np.zeros((bandas, n_fft // 2 + 1), dtype=np.float64)

    for banda in range(1, bandas + 1):
        izquierda, centro, derecha = bins[banda - 1], bins[banda], bins[banda + 1]
        centro = max(centro, izquierda + 1)
        derecha = max(derecha, centro + 1)
        for k in range(izquierda, centro):
            if 0 <= k < banco.shape[1]:
                banco[banda - 1, k] = (k - izquierda) / max(1, centro - izquierda)
        for k in range(centro, derecha):
            if 0 <= k < banco.shape[1]:
                banco[banda - 1, k] = (derecha - k) / max(1, derecha - centro)
    return banco


BANCO_MEL_CNN = banco_mel(fs=8000, n_fft=512, bandas=40)


def log_mel_segmento(trama: np.ndarray) -> np.ndarray:
    """Log-mel de 2 segundos compatible con la CNN de la otra programacion."""

    fs = 8000
    duracion_segmento = 2.0
    longitud_ventana = int(round(0.025 * fs))
    salto = int(round(0.01 * fs))
    n_fft = 512
    numero_tramas = int(np.ceil((duracion_segmento - 0.025) / 0.01))
    ventana = np.hanning(longitud_ventana)
    potencia = np.zeros((n_fft // 2 + 1, numero_tramas), dtype=np.float64)

    for indice in range(numero_tramas):
        inicio = indice * salto
        fragmento = np.zeros(longitud_ventana, dtype=np.float64)
        disponible = trama[inicio : inicio + longitud_ventana]
        fragmento[: len(disponible)] = disponible
        fft = np.fft.rfft(fragmento * ventana, n=n_fft)
        potencia[:, indice] = np.abs(fft) ** 2

    mel = BANCO_MEL_CNN @ potencia
    return np.log10(mel + 1e-6).astype(np.float32)


def promedio_tramas(tramas: list[np.ndarray], extraer: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return np.mean(np.stack([extraer(trama) for trama in tramas], axis=0), axis=0)


def caracteristicas_stft(trama: np.ndarray, cfg: Configuracion) -> np.ndarray:
    magnitud = np.log1p(espectrograma_magnitud(trama, cfg))
    return np.concatenate([magnitud.mean(axis=1), magnitud.std(axis=1)]).astype(np.float64)


def caracteristicas_mfcc(trama: np.ndarray) -> np.ndarray:
    mel = log_mel_segmento(trama)
    coeficientes = dct(mel, type=2, axis=0, norm="ortho")[:13]
    return coeficientes.mean(axis=1).astype(np.float64)


def caracteristicas_dwt(trama: np.ndarray) -> np.ndarray:
    coeficientes = pywt.wavedec(trama, wavelet="coif5", level=5, mode="symmetric")
    rasgos: list[float] = []
    for bloque in coeficientes:
        valores = np.asarray(bloque, dtype=np.float64)
        rasgos.extend(
            [
                float(np.log1p(np.mean(valores**2))),
                float(np.mean(np.abs(valores))),
                float(np.std(valores)),
            ]
        )
    return np.asarray(rasgos, dtype=np.float64)


def crear_cnn_y_extractor(ruta_checkpoint: Path):
    import torch
    import torch.nn as nn

    red = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 100),
        nn.ReLU(),
        nn.Linear(100, len(CLASES)),
    )
    checkpoint = torch.load(ruta_checkpoint, map_location="cpu", weights_only=False)
    red.load_state_dict(checkpoint["estado"])
    red.eval()
    extractor = nn.Sequential(*list(red.children())[:-1])
    extractor.eval()
    return torch, extractor


def caracteristicas_cnn(tramas: list[np.ndarray], torch, extractor) -> np.ndarray:
    entrada = np.stack([log_mel_segmento(trama) for trama in tramas], axis=0)
    tensor = torch.tensor(entrada[:, None, :, :], dtype=torch.float32)
    with torch.no_grad():
        vectores = extractor(tensor).cpu().numpy()
    return vectores.mean(axis=0).astype(np.float64)


def extraer_baselines(registros: list[RegistroComparacion], cfg: Configuracion, ruta_checkpoint: Path) -> dict[str, np.ndarray]:
    torch, extractor = crear_cnn_y_extractor(ruta_checkpoint)
    matrices: dict[str, list[np.ndarray]] = {"STFT": [], "MFCC": [], "DWT": [], "CNN": []}

    for posicion, registro in enumerate(registros, start=1):
        senal, frecuencia = leer_wav_normalizado(registro.ruta)
        if frecuencia != cfg.frecuencia_esperada_hz:
            raise ValueError(f"{registro.ruta} tiene {frecuencia} Hz y se esperaban {cfg.frecuencia_esperada_hz} Hz.")
        tramas = dividir_en_tramas(senal, cfg)
        if not tramas:
            raise ValueError(f"No se han podido construir tramas para {registro.ruta}.")

        matrices["STFT"].append(promedio_tramas(tramas, lambda trama: caracteristicas_stft(trama, cfg)))
        matrices["MFCC"].append(promedio_tramas(tramas, caracteristicas_mfcc))
        matrices["DWT"].append(promedio_tramas(tramas, caracteristicas_dwt))
        matrices["CNN"].append(caracteristicas_cnn(tramas, torch, extractor))

        if posicion == 1 or posicion % 50 == 0 or posicion == len(registros):
            print(f"Caracteristicas base extraidas: {posicion}/{len(registros)} audios")

    return {nombre: np.stack(vectores, axis=0) for nombre, vectores in matrices.items()}


def preparar_entrada_tsne(x: np.ndarray, semilla: int) -> tuple[np.ndarray, np.ndarray, int]:
    x_escalada = StandardScaler().fit_transform(x)
    componentes = min(50, x_escalada.shape[1], max(1, x_escalada.shape[0] - 1))
    if x_escalada.shape[1] > componentes:
        entrada = PCA(n_components=componentes, random_state=semilla).fit_transform(x_escalada)
    else:
        entrada = x_escalada
    return x_escalada, entrada, entrada.shape[1]


def calcular_metricas(x_escalada: np.ndarray, coordenadas: np.ndarray, etiquetas: np.ndarray) -> dict[str, float]:
    return {
        "silhouette_features": float(silhouette_score(x_escalada, etiquetas)),
        "davies_bouldin_features": float(davies_bouldin_score(x_escalada, etiquetas)),
        "silhouette_tsne": float(silhouette_score(coordenadas, etiquetas)),
        "davies_bouldin_tsne": float(davies_bouldin_score(coordenadas, etiquetas)),
    }


def ejecutar_tsne(nombre: str, x: np.ndarray, etiquetas: np.ndarray, semilla: int) -> tuple[np.ndarray, dict[str, float]]:
    x_escalada, entrada, componentes = preparar_entrada_tsne(x, semilla)
    perplejidad = min(30, max(5, (len(x) - 1) // 3))
    print(f"t-SNE {nombre}: {len(x)} puntos, {x.shape[1]} rasgos, entrada={componentes}, perplexity={perplejidad}")
    coordenadas = TSNE(
        n_components=2,
        perplexity=perplejidad,
        init="pca",
        learning_rate="auto",
        random_state=semilla,
        max_iter=1000,
    ).fit_transform(entrada)
    metricas = calcular_metricas(x_escalada, coordenadas, etiquetas)
    metricas.update(
        {
            "muestras": int(x.shape[0]),
            "rasgos_originales": int(x.shape[1]),
            "rasgos_entrada_tsne": int(componentes),
            "perplexity": int(perplejidad),
        }
    )
    return coordenadas, metricas


def guardar_matriz_rasgos(
    nombre: str,
    x: np.ndarray,
    registros: list[RegistroComparacion],
    carpeta_datos: Path,
) -> None:
    df = pd.DataFrame(x, columns=[f"F_{indice + 1:03d}" for indice in range(x.shape[1])])
    df.insert(0, "ruta", [str(registro.ruta) for registro in registros])
    df.insert(0, "archivo", [registro.archivo for registro in registros])
    df.insert(0, "clase", [registro.clase for registro in registros])
    df.to_csv(carpeta_datos / f"rasgos_{nombre.replace(' ', '_').lower()}.csv", index=False, encoding="utf-8-sig")


def guardar_coordenadas(
    nombre: str,
    coordenadas: np.ndarray,
    registros: list[RegistroComparacion],
    carpeta_datos: Path,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "clase": [registro.clase for registro in registros],
            "archivo": [registro.archivo for registro in registros],
            "ruta": [str(registro.ruta) for registro in registros],
            "tSNE_1": coordenadas[:, 0],
            "tSNE_2": coordenadas[:, 1],
        }
    )
    df.to_csv(carpeta_datos / f"coordenadas_tsne_{nombre.replace(' ', '_').lower()}.csv", index=False, encoding="utf-8-sig")
    return df


def dibujar_dispersion(eje, coordenadas: pd.DataFrame, nombre: str, subtitulo: str = "") -> None:
    for clase in CLASES:
        mascara = coordenadas["clase"] == clase
        eje.scatter(
            coordenadas.loc[mascara, "tSNE_1"],
            coordenadas.loc[mascara, "tSNE_2"],
            s=15,
            alpha=0.78,
            color=COLORES[clase],
            label=clase,
            edgecolors="none",
        )
    titulo = nombre if not subtitulo else f"{nombre}\n{subtitulo}"
    eje.set_title(titulo, fontsize=10)
    eje.set_xlabel("t-SNE 1")
    eje.set_ylabel("t-SNE 2")
    eje.grid(True, alpha=0.2)


def guardar_figura_individual(nombre: str, coordenadas: pd.DataFrame, metricas: dict[str, float], carpeta_figuras: Path) -> None:
    fig, eje = plt.subplots(figsize=(9, 7))
    subtitulo = (
        f"silhouette t-SNE={metricas['silhouette_tsne']:.3f} | "
        f"Davies-Bouldin t-SNE={metricas['davies_bouldin_tsne']:.3f}"
    )
    dibujar_dispersion(eje, coordenadas, nombre, subtitulo)
    eje.legend(title="Clase", loc="best")
    fig.suptitle("Comparacion final de la Figura 11", fontsize=14)
    fig.tight_layout()
    fig.savefig(carpeta_figuras / NOMBRES_FIGURA[nombre], dpi=320)
    plt.close(fig)


def texto_mejor_metodo(metricas: pd.DataFrame) -> str:
    mejor_sil = metricas.sort_values("silhouette_tsne", ascending=False).iloc[0]
    mejor_db = metricas.sort_values("davies_bouldin_tsne", ascending=True).iloc[0]
    if mejor_sil["metodo"] == "Deep ONMF" and mejor_db["metodo"] == "Deep ONMF":
        return (
            "En esta ejecucion deep ONMF obtiene la mejor separacion t-SNE segun las dos metricas "
            "de apoyo: silhouette mayor y Davies-Bouldin menor."
        )
    return (
        "La conclusion visual debe leerse junto a la tabla: esta ejecucion local no coloca a deep ONMF "
        "primero en las dos metricas t-SNE a la vez. El articulo objetivo si describe la Figura 11D "
        "como la separacion mas clara frente a CNN, DWT y MFCC."
    )


def fila_metricas(metricas: pd.DataFrame, metodo: str) -> pd.Series:
    return metricas.loc[metricas["metodo"] == metodo].iloc[0]


def envolver(texto: str, ancho: int = 92) -> str:
    lineas = []
    for parrafo in texto.splitlines():
        if not parrafo.strip():
            lineas.append("")
            continue
        lineas.append(textwrap.fill(parrafo, width=ancho))
    return "\n".join(lineas)


def analisis_metodo(metricas: pd.DataFrame, metodo: str) -> str:
    fila = fila_metricas(metricas, metodo)
    cifras = (
        f"silhouette rasgos={fila['silhouette_features']:.4f}; "
        f"DB rasgos={fila['davies_bouldin_features']:.4f}; "
        f"silhouette t-SNE={fila['silhouette_tsne']:.4f}; "
        f"DB t-SNE={fila['davies_bouldin_tsne']:.4f}."
    )
    observaciones = {
        "CNN": (
            "La imagen muestra varias islas compactas por clase. En esta ejecucion las agrupaciones "
            "de `N`, `MR` y `MS` quedan bastante reconocibles, aunque hay fragmentos y puntos aislados "
            "de otras clases. La tabla confirma que la representacion CNN es la referencia local mas "
            "fuerte de esta comparacion."
        ),
        "DWT": (
            "La transformada wavelet captura cambios temporales y de frecuencia, pero el mapa presenta "
            "islas pequenas mezcladas con outliers. La separacion visual parcial no se convierte en una "
            "buena compacidad global: su Davies-Bouldin t-SNE queda alto."
        ),
        "MFCC": (
            "Los MFCC condensan la envolvente espectral. Se ven grupos mejor definidos que en DWT para "
            "algunas clases, pero todavia aparecen regiones compartidas y puntos de clases distintas "
            "cerca entre si. Por eso mejora algunos indicadores sin reproducir la claridad del panel "
            "Deep ONMF del articulo."
        ),
        "Deep ONMF": (
            "Deep ONMF resume cada audio mediante siete SBV. En la foto local hay estructura visible: "
            "zonas compactas de `N`, `MR`, `MS` y `AS`, junto con cruces y muestras dispersas, sobre todo "
            "cuando `MVP` se aproxima a otras clases. Su Davies-Bouldin t-SNE es competitivo frente a "
            "DWT, MFCC y STFT, pero el silhouette local no supera a CNN."
        ),
        "STFT": (
            "STFT usa un resumen espectral de mayor dimension. El t-SNE encuentra trayectorias e islas "
            "estrechas, pero varias quedan fragmentadas y proximas a otras clases. Esa geometria explica "
            "que el silhouette t-SNE pueda parecer aceptable mientras Davies-Bouldin penaliza mucho la "
            "dispersion y la vecindad entre grupos."
        ),
    }
    return f"{metodo}. {observaciones[metodo]} {cifras}"


def texto_metricas(metricas: pd.DataFrame) -> str:
    mejor_sil_rasgos = metricas.sort_values("silhouette_features", ascending=False).iloc[0]
    mejor_db_rasgos = metricas.sort_values("davies_bouldin_features", ascending=True).iloc[0]
    mejor_sil_tsne = metricas.sort_values("silhouette_tsne", ascending=False).iloc[0]
    mejor_db_tsne = metricas.sort_values("davies_bouldin_tsne", ascending=True).iloc[0]
    return (
        "La tabla separa dos lecturas. Las columnas con sufijo `features` se calculan antes de reducir "
        "a dos dimensiones y describen los rasgos reales que alimentan t-SNE. Las columnas con sufijo "
        "`tsne` se calculan sobre las coordenadas dibujadas y ayudan a leer la figura, pero no son una "
        "accuracy de clasificacion.\n\n"
        "Silhouette debe subir: valores mayores indican que cada audio queda mas cerca de su clase que "
        "de las otras. Davies-Bouldin debe bajar: valores menores indican grupos mas compactos y mas "
        "separados entre si.\n\n"
        f"Mejor silhouette en rasgos: {mejor_sil_rasgos['metodo']} "
        f"({mejor_sil_rasgos['silhouette_features']:.4f}). Mejor Davies-Bouldin en rasgos: "
        f"{mejor_db_rasgos['metodo']} ({mejor_db_rasgos['davies_bouldin_features']:.4f}). "
        f"Mejor silhouette t-SNE: {mejor_sil_tsne['metodo']} "
        f"({mejor_sil_tsne['silhouette_tsne']:.4f}). Mejor Davies-Bouldin t-SNE: "
        f"{mejor_db_tsne['metodo']} ({mejor_db_tsne['davies_bouldin_tsne']:.4f})."
    )


def texto_como_explicarlo(metricas: pd.DataFrame) -> str:
    return (
        "1. Primero se mira la foto: un buen mapa t-SNE deja nubes compactas de un color y evita "
        "regiones donde muchos colores se pisan. Los ejes t-SNE no tienen unidades fisicas; importa "
        "la vecindad relativa de los puntos.\n\n"
        "2. Despues se contrasta con la tabla. Una foto atractiva no basta: silhouette y "
        "Davies-Bouldin ponen numeros a la separacion. En este informe se muestran tanto los rasgos "
        "originales como las coordenadas t-SNE para no confundir calidad de caracteristicas con una "
        "proyeccion de dos dimensiones.\n\n"
        "3. La Figura 11 del articulo es la referencia conceptual: su panel Deep ONMF muestra las "
        "clases `N` y `MR` muy diferenciadas y el bloque superior de patologias queda mas ordenado que "
        "en CNN, DWT y MFCC. La reproduccion local puede variar por semillas, normalizacion, duracion "
        "de trama, forma exacta de construir los SBV y detalles del t-SNE no publicados.\n\n"
        f"4. Conclusion de esta ejecucion: {texto_mejor_metodo(metricas)} "
        "Por eso los ajustes se guardan como pruebas reproducibles y no se fuerza una conclusion que "
        "la tabla no sostenga."
    )


def tabla_markdown(df: pd.DataFrame) -> str:
    columnas = [str(columna) for columna in df.columns]
    lineas = [
        "| " + " | ".join(columnas) + " |",
        "| " + " | ".join("---" for _ in columnas) + " |",
    ]
    for fila in df.itertuples(index=False, name=None):
        valores = []
        for valor in fila:
            if isinstance(valor, (float, np.floating)):
                valores.append(f"{float(valor):.4f}")
            else:
                valores.append(str(valor))
        lineas.append("| " + " | ".join(valores) + " |")
    return "\n".join(lineas)


def guardar_pdf_comparativo(
    coordenadas_por_metodo: dict[str, pd.DataFrame],
    metricas: pd.DataFrame,
    carpeta_pdf: Path,
) -> Path:
    ruta_pdf = carpeta_pdf / "Comparacion_Figura11_CNN_DWT_MFCC_Deep_ONMF_STFT.pdf"
    with PdfPages(ruta_pdf) as pdf:
        fig, ejes = plt.subplots(2, 3, figsize=(18, 11))
        for eje, nombre in zip(ejes.flat, ORDEN_METODOS):
            fila = metricas.loc[metricas["metodo"] == nombre].iloc[0]
            subtitulo = f"sil={fila['silhouette_tsne']:.3f} | DB={fila['davies_bouldin_tsne']:.3f}"
            dibujar_dispersion(eje, coordenadas_por_metodo[nombre], nombre, subtitulo)

        eje_texto = ejes.flat[-1]
        eje_texto.axis("off")
        eje_texto.text(
            0.02,
            0.98,
            "Lectura\n\n"
            "CNN, DWT, MFCC y deep ONMF\n"
            "corresponden a la Figura 11.\n\n"
            "STFT es la comparacion adicional\n"
            "solicitada.\n\n"
            "silhouette mas alto = mejor.\n"
            "Davies-Bouldin mas bajo = mejor.\n\n"
            f"{texto_mejor_metodo(metricas)}",
            va="top",
            ha="left",
            fontsize=10,
            wrap=True,
        )
        manejadores, etiquetas = ejes.flat[0].get_legend_handles_labels()
        fig.legend(manejadores, etiquetas, title="Clase", loc="lower center", ncol=len(CLASES))
        fig.suptitle("Comparacion final lado a lado de espacios t-SNE", fontsize=16)
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        pdf.savefig(fig)
        fig.savefig(carpeta_pdf / "Comparacion_Figura11_lado_a_lado.png", dpi=260)
        plt.close(fig)

        fig_tabla, eje_tabla = plt.subplots(figsize=(16, 8))
        eje_tabla.axis("off")
        columnas = [
            "metodo",
            "muestras",
            "rasgos_originales",
            "silhouette_features",
            "davies_bouldin_features",
            "silhouette_tsne",
            "davies_bouldin_tsne",
        ]
        tabla = metricas[columnas].copy()
        for columna in columnas[3:]:
            tabla[columna] = tabla[columna].map(lambda valor: f"{valor:.4f}")
        objeto = eje_tabla.table(cellText=tabla.values, colLabels=tabla.columns, cellLoc="center", loc="center")
        objeto.auto_set_font_size(False)
        objeto.set_fontsize(9)
        objeto.scale(1.0, 1.7)
        eje_tabla.set_title("Metricas de apoyo de la comparacion", fontsize=14, pad=20)
        eje_tabla.text(
            0.02,
            0.13,
            envolver(texto_metricas(metricas), ancho=150),
            transform=eje_tabla.transAxes,
            fontsize=9.5,
            va="bottom",
            wrap=True,
        )
        eje_tabla.text(
            0.02,
            0.04,
            envolver(texto_mejor_metodo(metricas), ancho=145),
            transform=eje_tabla.transAxes,
            fontsize=10,
            va="bottom",
            wrap=True,
        )
        fig_tabla.tight_layout()
        pdf.savefig(fig_tabla)
        plt.close(fig_tabla)

        fig_analisis = plt.figure(figsize=(16, 10))
        fig_analisis.patch.set_facecolor("white")
        fig_analisis.text(0.04, 0.95, "Analisis detallado de las figuras separadas", fontsize=18, weight="bold")
        fig_analisis.text(
            0.04,
            0.90,
            envolver(
                "Estas notas describen lo que se observa en las fotos generadas y lo conectan con la "
                "tabla. Son una guia para estudiar la comparacion antes de redactar conclusiones.",
                ancho=135,
            ),
            fontsize=10.5,
            va="top",
        )
        posiciones = {
            "CNN": (0.04, 0.81),
            "DWT": (0.04, 0.58),
            "MFCC": (0.04, 0.35),
            "Deep ONMF": (0.52, 0.81),
            "STFT": (0.52, 0.52),
        }
        for nombre, (x, y) in posiciones.items():
            fig_analisis.text(
                x,
                y,
                envolver(analisis_metodo(metricas, nombre), ancho=72),
                fontsize=10,
                va="top",
            )
        fig_analisis.text(
            0.52,
            0.23,
            envolver(
                "Regla de lectura: si una clase se reparte en varias islas o comparte una zona con "
                "otros colores, la separacion es menos estable que cuando forma una nube compacta "
                "alejada. Un outlier aislado no invalida por si solo el metodo, pero muchos outliers "
                "suben Davies-Bouldin y reducen silhouette.",
                ancho=72,
            ),
            fontsize=10,
            va="top",
        )
        pdf.savefig(fig_analisis)
        plt.close(fig_analisis)

        fig_guion = plt.figure(figsize=(16, 10))
        fig_guion.patch.set_facecolor("white")
        fig_guion.text(0.05, 0.94, "Guion para entender y explicar el resultado", fontsize=18, weight="bold")
        fig_guion.text(
            0.05,
            0.86,
            envolver(texto_como_explicarlo(metricas), ancho=128),
            fontsize=12,
            va="top",
            linespacing=1.35,
        )
        fig_guion.text(
            0.05,
            0.14,
            envolver(
                "Pregunta clave para la fase de mejora: si Deep ONMF no queda primero localmente, "
                "hay que comparar configuraciones reproducibles y comprobar si el cambio mejora "
                "tambien los rasgos, no solo la apariencia del t-SNE.",
                ancho=128,
            ),
            fontsize=11,
            va="top",
        )
        pdf.savefig(fig_guion)
        plt.close(fig_guion)
    return ruta_pdf


def guardar_informe(
    carpetas: dict[str, Path],
    metricas: pd.DataFrame,
    ruta_sbv: Path,
    ruta_checkpoint: Path,
    cantidad_audios: int,
    semilla: int,
) -> None:
    tabla = tabla_markdown(metricas[
        [
            "metodo",
            "muestras",
            "rasgos_originales",
            "rasgos_entrada_tsne",
            "silhouette_features",
            "davies_bouldin_features",
            "silhouette_tsne",
            "davies_bouldin_tsne",
        ]
    ])
    informe = f"""# Informe de resultados de la comparacion final

## Que se ha generado

- Figuras separadas en `01_figuras_separadas`.
- PDF comparativo lado a lado en `02_pdf_comparativo`.
- Rasgos, coordenadas t-SNE y metricas CSV en `03_datos_y_metricas`.

## Origen real de los datos

- Audios comparados: `{cantidad_audios}` WAV referenciados por deep ONMF.
- Caracteristicas deep ONMF: `{ruta_sbv}`.
- Checkpoint CNN log-mel: `{ruta_checkpoint}`.
- Semilla t-SNE: `{semilla}`.

## Tabla de metricas

{tabla}

## Como leer la tabla

{texto_metricas(metricas)}

## Analisis de cada figura

{chr(10).join(f"- {analisis_metodo(metricas, nombre)}" for nombre in ORDEN_METODOS)}

## Lectura final

{texto_mejor_metodo(metricas)}

Para la comparacion visual abre primero las fotos separadas y despues el PDF.
La primera pagina del PDF coloca los metodos lado a lado. Las paginas siguientes
explican la tabla, cada figura y un guion de defensa para estudiar el resultado.

## Guion para explicarlo

{texto_como_explicarlo(metricas)}

## Nota metodologica

La Figura 11 original del articulo objetivo contiene CNN, DWT, MFCC y deep ONMF.
STFT aparece aqui como comparacion adicional pedida. La CNN local procede de la
programacion del articulo de log-mel, mientras deep ONMF procede de la
programacion objetivo.
"""
    (carpetas["base"] / "00_INFORME_RESULTADOS.md").write_text(informe, encoding="utf-8")


def guardar_origenes(
    carpetas: dict[str, Path],
    ruta_sbv: Path,
    ruta_checkpoint: Path,
    cfg: Configuracion,
    args: argparse.Namespace,
) -> None:
    datos = {
        "articulo_objetivo": str(RAIZ_OBJETIVO / "articulo_objetivo.pdf"),
        "articulo_a_implementar": str(RAIZ_IMPLEMENTAR / "artículo_implementar.pdf"),
        "caracteristicas_deep_onmf": str(ruta_sbv),
        "parametros_deep_onmf_usado": leer_parametros_deep_onmf(ruta_sbv),
        "checkpoint_cnn": str(ruta_checkpoint),
        "parametros_stft_objetivo": cfg.como_diccionario(),
        "parametros_comparacion": {
            "clases": list(CLASES),
            "semilla": args.semilla,
            "limite_por_clase": args.limite_por_clase,
            "mfcc_bandas_mel": 40,
            "mfcc_coeficientes": 13,
            "dwt_wavelet": "coif5",
            "dwt_niveles": 5,
            "cnn_embedding": "salida de la capa ReLU previa al clasificador final",
        },
    }
    (carpetas["datos"] / "origenes_y_parametros.json").write_text(
        json.dumps(datos, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parsear_argumentos()
    carpetas = crear_carpetas_salida()
    cfg = Configuracion(raiz=RAIZ_OBJETIVO, rellenar_audios_cortos=True)
    ruta_sbv = encontrar_caracteristicas_deep_onmf(args.ruta_sbv_deep_onmf)
    ruta_checkpoint = encontrar_checkpoint_cnn()
    deep_df, registros = seleccionar_deep_onmf(ruta_sbv, args.limite_por_clase)
    etiquetas = np.asarray([registro.clase for registro in registros])

    print(f"Usando deep ONMF: {ruta_sbv}")
    print(f"Usando CNN: {ruta_checkpoint}")
    print(f"Audios de comparacion: {len(registros)}")

    matrices = extraer_baselines(registros, cfg, ruta_checkpoint)
    matrices["Deep ONMF"] = deep_df[[f"SBV_{indice}" for indice in range(1, 8)]].to_numpy(dtype=np.float64)

    coordenadas_por_metodo: dict[str, pd.DataFrame] = {}
    filas_metricas: list[dict[str, object]] = []
    for nombre in ORDEN_METODOS:
        x = matrices[nombre]
        guardar_matriz_rasgos(nombre, x, registros, carpetas["datos"])
        coordenadas, metricas = ejecutar_tsne(nombre, x, etiquetas, args.semilla)
        df_coords = guardar_coordenadas(nombre, coordenadas, registros, carpetas["datos"])
        coordenadas_por_metodo[nombre] = df_coords
        guardar_figura_individual(nombre, df_coords, metricas, carpetas["figuras"])
        filas_metricas.append({"metodo": nombre, **metricas})

    metricas_df = pd.DataFrame(filas_metricas)
    metricas_df.to_csv(carpetas["datos"] / "metricas_comparacion_figura11.csv", index=False, encoding="utf-8-sig")
    ruta_pdf = guardar_pdf_comparativo(coordenadas_por_metodo, metricas_df, carpetas["pdf"])
    guardar_informe(carpetas, metricas_df, ruta_sbv, ruta_checkpoint, len(registros), args.semilla)
    guardar_origenes(carpetas, ruta_sbv, ruta_checkpoint, cfg, args)

    print(f"PDF comparativo generado: {ruta_pdf}")
    print(f"Informe generado: {carpetas['base'] / '00_INFORME_RESULTADOS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
